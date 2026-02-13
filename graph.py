# Here you go — a **single, self-contained Python file** that bundles everything we built:
#
# * Dataset + collate + batched GCN→GRU (**multi-horizon**) model + Trainer
# * **Optuna** tuning with **GPU-aware scheduling** (file-locks) + pruning + Journal storage
# * **Visualization** block to render Optuna plots (HTML + optional PNG)
#
# It exposes **three subcommands**:
#
# * `demo` — quick sanity run on dummy data
# * `optuna` — run hyperparameter tuning
# * `analyze` — render Optuna visualizations from the journal
#
# Just copy this into `wind_st_all_in_one.py` and run from there.
#
# ```python
#!/usr/bin/env python3
# wind_st_all_in_one.py
# Bundled: dataset + model + trainer + Optuna tuning (GPU-aware) + analysis visuals
# https://chatgpt.com/share/69149ad3-b47c-8008-b09f-dea7b4fd0079
import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional (only used in graph/model parts)
from torch_geometric.nn import GCNConv, Node2Vec, radius_graph, knn_graph


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tensor_maybe(path: str, shape=None):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    t = torch.load(p)
    if shape is not None:
        assert tuple(t.shape) == tuple(shape), f"Expected {shape}, got {tuple(t.shape)}"
    return t


# ============================================================
# Dataset & Collate
# ============================================================

class WindSTDataset(Dataset):
    """
    Builds lagged sequences + future NWP + targets for multi-horizon forecasting.

    obs_feats: [T, N, Fobs]   (past observations; can include y if desired)
    y:         [T, N, 1]      (target)
    nwp:       [T, N, Fnwp]   (aligned by valid time)
    Z:         [N, Dz]        (static node2vec embeddings)
    lags:      list of ints, e.g., [0,1,2,3,4]  (t, t-1, ..., t-4)
    horizons:  H future steps to predict (e.g., 6 for +1..+6)
    """
    def __init__(self, obs_feats, y, nwp, Z, lags, horizons):
        super().__init__()
        self.obs_feats = obs_feats.float()
        self.y         = y.float()
        self.nwp       = nwp.float()
        self.Z         = Z.float()
        self.lags      = list(lags)
        self.H         = int(horizons)

        T = obs_feats.size(0)
        self.t0 = max(self.lags)
        self.t1 = T - self.H - 1
        assert self.t1 >= self.t0, "Not enough history for given lags/horizons."
        self.indices = torch.arange(self.t0, self.t1 + 1)

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, idx):
        t = int(self.indices[idx])
        seq = [self.obs_feats[t - l] for l in self.lags]
        X_seq = torch.stack(seq, dim=0)                        # [W, N, Fobs]
        W, N, _ = X_seq.shape

        Zt = self.Z.unsqueeze(0).expand(W, -1, -1)             # [W, N, Dz]
        X_seq = torch.cat([X_seq, Zt], dim=-1)                 # [W, N, Fobs+Dz]

        X_fut_seq = self.nwp[t+1:t+1+self.H]                   # [H, N, Fnwp]
        Y_seq     = self.y[t+1:t+1+self.H]                     # [H, N, 1]
        return {"X_seq": X_seq, "X_fut_seq": X_fut_seq, "Y_seq": Y_seq}


def wind_collate(batch):
    X_seq     = torch.stack([b["X_seq"] for b in batch], dim=0)      # [B, W, N, F]
    X_fut_seq = torch.stack([b["X_fut_seq"] for b in batch], dim=0)  # [B, H, N, Fnwp]
    Y_seq     = torch.stack([b["Y_seq"] for b in batch], dim=0)      # [B, H, N, 1]
    return X_seq, X_fut_seq, Y_seq


# ============================================================
# Batched GCN → GRU (Multi-Horizon) Model
# ============================================================

def expand_edge_index(edge_index, num_graphs, nodes_per_graph, device=None):
    device = device or edge_index.device
    E = edge_index.size(1)
    offset = (torch.arange(num_graphs, device=device) * nodes_per_graph).view(num_graphs, 1, 1)
    return (edge_index.unsqueeze(0) + offset).permute(1, 0, 2).reshape(2, num_graphs * E)


class GraphGRUForecasterMH(nn.Module):
    """
    Lag encoder: GCN→GCN (shared over lags, batched across B×W)
    Temporal encoder: GRU per node over W lags
    Decoder: parallel multi-horizon head conditioned on horizon-specific NWP
    """
    def __init__(self, f_in, f_nwp, horizons, hidden_gcn=64, hidden_rnn=128, hidden_dec=128):
        super().__init__()
        self.horizons = horizons
        self.gcn1 = GCNConv(f_in, hidden_gcn)
        self.gcn2 = GCNConv(hidden_gcn, hidden_gcn)
        self.gru  = nn.GRU(input_size=hidden_gcn, hidden_size=hidden_rnn, batch_first=True)
        self.decod1 = nn.Linear(hidden_rnn + f_nwp, hidden_dec)
        self.decod2 = nn.Linear(hidden_dec, 1)
        self._cached = {"key": None, "edge_index_big": None}

    def _batched_gconv(self, X_seq, edge_index):
        B, W, N, F = X_seq.shape
        G = B * W
        key = (B, W, N, edge_index.device, edge_index.numel())
        if self._cached["key"] != key:
            self._cached["edge_index_big"] = expand_edge_index(edge_index, G, N, device=edge_index.device)
            self._cached["key"] = key
        edge_big = self._cached["edge_index_big"]

        X_flat = X_seq.reshape(G * N, F)
        H = self.gcn1(X_flat, edge_big); H = F.relu(H)
        H = self.gcn2(H, edge_big)
        return H.view(B, W, N, -1)

    def forward(self, X_seq, X_fut_seq, edge_index):
        """
        X_seq:     [B, W, N, F_in]
        X_fut_seq: [B, H, N, F_nwp]
        edge_index: [2, E]
        """
        B, W, N, _ = X_seq.shape
        H = self.horizons

        H_seq = self._batched_gconv(X_seq, edge_index)          # [B, W, N, H_gcn]
        H_seq = H_seq.permute(0, 2, 1, 3).contiguous()          # [B, N, W, H_gcn]
        H_seq = H_seq.view(B * N, W, -1)                        # [B*N, W, H_gcn]
        _, hT = self.gru(H_seq)                                 # [1, B*N, H_rnn]
        hT = hT.squeeze(0).view(B, N, -1)                       # [B, N, H_rnn]

        hT_H = hT.unsqueeze(1).expand(B, H, N, hT.size(-1))     # [B, H, N, H_rnn]
        dec_in = torch.cat([hT_H, X_fut_seq], dim=-1)           # [B, H, N, H_rnn+F_nwp]
        z = F.relu(self.decod1(dec_in))
        yhat = self.decod2(z)                                   # [B, H, N, 1]
        return yhat


# ============================================================
# Trainer
# ============================================================

class Trainer:
    def __init__(self, model, edge_index, device="cuda", lr=1e-3, wd=1e-5, gamma=0.95):
        self.model = model.to(device)
        self.edge_index = edge_index.to(device)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.device = device
        self.gamma = gamma

    def weighted_mae(self, yhat, y):
        H = yhat.size(1)
        w = torch.tensor([self.gamma**h for h in range(H)], device=yhat.device).view(1, H, 1, 1)
        return (w * (yhat - y).abs()).mean()

    def run_epoch(self, loader, train=True):
        self.model.train(mode=train)
        total, n = 0.0, 0
        for X_seq, X_fut, Y in loader:
            X_seq = X_seq.to(self.device)
            X_fut = X_fut.to(self.device)
            Y     = Y.to(self.device)

            if train: self.opt.zero_grad()
            yhat = self.model(X_seq, X_fut, self.edge_index)
            loss = self.weighted_mae(yhat, Y)
            if train:
                loss.backward()
                self.opt.step()
            total += loss.item() * X_seq.size(0)
            n += X_seq.size(0)
        return total / max(n, 1)


# ============================================================
# GPU Manager (file-lock based)
# ============================================================

import time
import errno
import fcntl  # Linux/macOS; on Windows, use 'portalocker'

class GPUManager:
    """
    Cross-process GPU allocator using file locks.
    """
    def __init__(self, gpu_ids, lock_dir=".gpu_locks", strict=True, try_interval=1.0, timeout=None):
        self.gpu_ids = list(gpu_ids)
        self.lock_dir = lock_dir
        self.strict = strict
        self.try_interval = try_interval
        self.timeout = timeout
        os.makedirs(lock_dir, exist_ok=True)

    def _lock_path(self, gid):
        return os.path.join(self.lock_dir, f"gpu_{gid}.lock")

    def _try_lock(self, gid):
        path = self._lock_path(gid)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode())
            return fd
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return None
            raise

    def acquire(self):
        if not self.gpu_ids:
            return "cpu", (lambda: None)
        start = time.time()
        while True:
            for gid in self.gpu_ids:
                fd = self._try_lock(gid)
                if fd is not None:
                    device = f"cuda:{gid}"
                    def _release(fd_=fd):
                        try:
                            fcntl.flock(fd_, fcntl.LOCK_UN)
                        finally:
                            os.close(fd_)
                    return device, _release
            if not self.strict:
                return "cpu", (lambda: None)
            if self.timeout is not None and (time.time() - start) > self.timeout:
                return "cpu", (lambda: None)
            time.sleep(self.try_interval)


# ============================================================
# Optuna: search space, objective, study runner
# ============================================================

def suggest_hyperparams(trial):
    params = {}
    params["graph_type"] = trial.suggest_categorical("graph_type", ["radius", "knn"])
    if params["graph_type"] == "radius":
        params["radius_r"] = trial.suggest_float("radius_r", 0.15, 0.6, step=0.05)
    else:
        params["knn_k"] = trial.suggest_int("knn_k", 4, 16, step=2)

    params["embedding_dim"]   = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    params["walk_length"]     = trial.suggest_int("walk_length", 10, 30, step=5)
    params["context_size"]    = trial.suggest_int("context_size", 5, 15, step=2)
    params["walks_per_node"]  = trial.suggest_int("walks_per_node", 5, 15, step=2)
    params["p"]               = trial.suggest_float("p", 0.25, 4.0, log=True)
    params["q"]               = trial.suggest_float("q", 0.25, 4.0, log=True)

    params["hidden_gcn"]      = trial.suggest_categorical("hidden_gcn", [32, 64, 96, 128])
    params["hidden_rnn"]      = trial.suggest_categorical("hidden_rnn", [64, 128, 192, 256])
    params["hidden_dec"]      = trial.suggest_categorical("hidden_dec", [64, 128, 192])
    params["num_lags"]        = trial.suggest_int("num_lags", 3, 6)
    params["num_horizons"]    = trial.suggest_int("num_horizons", 3, 12)

    params["lr"]              = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    params["weight_decay"]    = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    params["gamma"]           = trial.suggest_float("gamma", 0.90, 0.99)
    params["batch_size"]      = trial.suggest_categorical("batch_size", [8, 16, 32])

    params["dropout_gcn"]     = trial.suggest_float("dropout_gcn", 0.0, 0.4, step=0.1)
    params["dropout_rnn"]     = trial.suggest_float("dropout_rnn", 0.0, 0.4, step=0.1)
    return params


def make_objective(args, default_device, coords, obs_feats, y, nwp, gpu_manager):
    import optuna  # local import to avoid hard dependency in other subcommands

    def objective(trial: "optuna.trial.Trial"):
        set_seed(args.seed + trial.number)
        device, release_gpu = gpu_manager.acquire()
        try:
            hp = suggest_hyperparams(trial)

            # Graph build on device
            if hp["graph_type"] == "radius":
                edge_index = radius_graph(coords.to(device), r=hp["radius_r"], loop=False)
            else:
                edge_index = knn_graph(coords.to(device), k=hp["knn_k"], loop=False)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

            # Node2Vec pretrain (brief)
            n2v = Node2Vec(
                edge_index,
                embedding_dim=hp["embedding_dim"],
                walk_length=hp["walk_length"],
                context_size=hp["context_size"],
                walks_per_node=hp["walks_per_node"],
                p=hp["p"], q=hp["q"],
                sparse=True,
            ).to(device)

            opt_n2v = torch.optim.SparseAdam(list(n2v.parameters()), lr=0.01)
            loader = n2v.loader(batch_size=128, shuffle=True)
            for _ in range(args.n2v_epochs):
                for pos_rw, neg_rw in loader:
                    opt_n2v.zero_grad()
                    loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    opt_n2v.step()

            with torch.no_grad():
                Z = n2v.embedding.weight.detach().cpu()

            # Dataset / split
            lags = list(range(hp["num_lags"]))
            ds = WindSTDataset(obs_feats, y, nwp, Z, lags, horizons=hp["num_horizons"])
            n_total = len(ds)
            n_train = int(args.train_frac * n_total)
            n_val   = n_total - n_train
            train_ds, val_ds = torch.utils.data.random_split(
                ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(args.seed)
            )
            train_loader = DataLoader(
                train_ds, batch_size=hp["batch_size"], shuffle=True,
                num_workers=args.num_workers, pin_memory=True, collate_fn=wind_collate
            )
            val_loader = DataLoader(
                val_ds, batch_size=max(16, hp["batch_size"]), shuffle=False,
                num_workers=args.num_workers, pin_memory=True, collate_fn=wind_collate
            )

            # Model & Trainer
            F_in = obs_feats.size(-1) + hp["embedding_dim"]
            model = GraphGRUForecasterMH(
                f_in=F_in, f_nwp=nwp.size(-1), horizons=hp["num_horizons"],
                hidden_gcn=hp["hidden_gcn"], hidden_rnn=hp["hidden_rnn"], hidden_dec=hp["hidden_dec"],
            )
            trainer = Trainer(model, edge_index, device=device, lr=hp["lr"],
                              wd=hp["weight_decay"], gamma=hp["gamma"])

            # Prunable training loop
            best_val = float("inf")
            for ep in range(1, args.epochs + 1):
                trainer.run_epoch(train_loader, train=True)
                val_mae = trainer.run_epoch(val_loader, train=False)
                trial.report(val_mae, step=ep)
                if val_mae < best_val:
                    best_val = val_mae
                if trial.should_prune():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return best_val

        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            release_gpu()

    return objective


def run_optuna(args):
    import optuna
    from optuna.storages import JournalStorage, JournalFileStorage

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load or dummy tensors
    obs_feats = load_tensor_maybe(args.obs_feats)
    y         = load_tensor_maybe(args.y)
    nwp       = load_tensor_maybe(args.nwp)
    coords    = load_tensor_maybe(args.coords)

    if any(t is None for t in [obs_feats, y, nwp, coords]):
        T, N, Fobs, Fnwp, D = args.T, args.N, args.Fobs, args.Fnwp, args.Dcoord
        obs_feats = torch.randn(T, N, Fobs)
        y         = torch.randn(T, N, 1)
        nwp       = torch.randn(T, N, Fnwp)
        coords    = torch.randn(N, D)

    # GPU manager
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    gpu_manager = GPUManager(
        gpu_ids=gpu_ids,
        lock_dir=args.gpu_lock_dir,
        strict=bool(args.gpu_strict),
        try_interval=1.0,
        timeout=args.gpu_timeout,
    )

    storage = JournalStorage(JournalFileStorage(args.journal))
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    elif args.pruner == "sha":
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    else:
        pruner = optuna.pruners.HyperbandPruner(min_resource=5, reduction_factor=3)

    study = optuna.create_study(
        study_name=args.study,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
    )

    objective = make_objective(args, device, coords, obs_feats, y, nwp, gpu_manager)
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)

    print("\nBest trial:")
    best = study.best_trial
    print(f"  value (val MAE): {best.value:.6f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\nTrials completed: {len(study.trials)}")
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  pruned: {pruned} | complete: {complete}")


# ============================================================
# Visualization (Optuna plots)
# ============================================================

def run_analyze(args):
    import optuna
    from optuna.storages import JournalStorage, JournalFileStorage
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
        plot_edf,
    )

    try:
        import plotly.io as pio
        HAVE_PNG = True
    except Exception:
        HAVE_PNG = False

    def save_fig(fig, out_html: Path, out_png: Path | None):
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_html), include_plotlyjs="cdn")
        if out_png and HAVE_PNG:
            try:
                pio.write_image(fig, str(out_png), format="png", scale=2)
            except Exception as e:
                print(f"[warn] PNG export failed for {out_png.name}: {e}")

    storage = JournalStorage(JournalFileStorage(args.journal))
    study = optuna.load_study(study_name=args.study, storage=storage)

    outdir = Path(args.outdir)
    export_png = bool(args.png)

    fig_hist = plot_optimization_history(study)
    save_fig(fig_hist, outdir / "optimization_history.html",
             (outdir / "optimization_history.png") if export_png else None)

    fig_pc = plot_parallel_coordinate(study)
    save_fig(fig_pc, outdir / "parallel_coordinate.html",
             (outdir / "parallel_coordinate.png") if export_png else None)

    if args.importance_evaluator == "fanova":
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=42)
    elif args.importance_evaluator == "mean":
        evaluator = optuna.importance.MeanDecreaseImpurityImportanceEvaluator()
    else:
        evaluator = optuna.importance.BoosterImportanceEvaluator()
    fig_imp = plot_param_importances(study, evaluator=evaluator)
    save_fig(fig_imp, outdir / "param_importances.html",
             (outdir / "param_importances.png") if export_png else None)

    fig_slice = plot_slice(study)
    save_fig(fig_slice, outdir / "slice.html",
             (outdir / "slice.png") if export_png else None)

    fig_edf = plot_edf(study)
    save_fig(fig_edf, outdir / "edf.html",
             (outdir / "edf.png") if export_png else None)

    print(f"[done] Reports written to: {outdir.resolve()}")
    if export_png and not HAVE_PNG:
        print("[note] PNG export requested but kaleido is not available. pip install -U kaleido")


# ============================================================
# Demo runner (quick sanity on dummy data)
# ============================================================

def run_demo(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy tensors
    T, N, Fobs, Fnwp, Dz, D = 800, 64, 6, 8, 32, 2
    obs_feats = torch.randn(T, N, Fobs)
    y         = torch.randn(T, N, 1)
    nwp       = torch.randn(T, N, Fnwp)
    coords    = torch.randn(N, D)

    # Graph (undirected)
    edge_index = radius_graph(coords, r=0.35, loop=False)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Tiny node2vec
    n2v = Node2Vec(edge_index, embedding_dim=Dz, walk_length=15, context_size=7,
                   walks_per_node=8, p=1.0, q=1.0, sparse=True).to(device)
    opt_n2v = torch.optim.SparseAdam(n2v.parameters(), lr=0.01)
    loader = n2v.loader(batch_size=128, shuffle=True)
    for _ in range(2):
        for pos_rw, neg_rw in loader:
            opt_n2v.zero_grad()
            loss = n2v.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward(); opt_n2v.step()
    with torch.no_grad():
        Z = n2v.embedding.weight.detach().cpu()

    # Dataset
    lags = [0,1,2,3,4]
    H    = 6
    ds   = WindSTDataset(obs_feats, y, nwp, Z, lags, horizons=H)
    n_total = len(ds)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=wind_collate)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, collate_fn=wind_collate)

    model = GraphGRUForecasterMH(f_in=Fobs+Dz, f_nwp=Fnwp, horizons=H,
                                 hidden_gcn=64, hidden_rnn=64, hidden_dec=64)
    trainer = Trainer(model, edge_index, device=device, lr=1e-3, wd=1e-5, gamma=0.95)
    for ep in range(5):
        tr = trainer.run_epoch(train_loader, train=True)
        va = trainer.run_epoch(val_loader, train=False)
        print(f"epoch {ep+1:02d} | train {tr:.4f} | val {va:.4f}")


# ============================================================
# CLI
# ============================================================

class BuildParser:
    def __init__(self, cmd):
        self.cmd = cmd
        self.journal = "optuna_wind_journal.log"
        self.study = "wind_gnn_gru"
        self.trials = 10
        self.jobs = 3
        self.n2v_epochs = 3
        self.seed = 42
        self.train_frac = 0.8
        self.num_workers = 10
        self.obs_feats = 'Wind.pt'
        self.y = 'MW.pt'
        self.nwp = 'nwp.pt'
        self.coords = 'coords.pt'
        self.T = 4000
        self.N = 200
        self.Fobs = 6
        self.Fnwp = 8
        self.Dcoord = 2
        self.pruner = "median"
        self.gpu_ids = '0'
        self.gpu_lock_dir = ".gpu_locks"
        self.gpu_strict = 1
        self.gpu_timeout = None
        self.outdir = './optuna_reports'
        self.include_pruned = 1
        self.importance_evaluator = 'shap'
        self.png = 1


def build_parser():
    p = argparse.ArgumentParser(description="Wind ST-GNN pipeline: demo | optuna | analyze")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- demo ---
    pd = sub.add_parser("demo", help="Quick sanity run on dummy data.")
    pd.add_argument("--seed", type=int, default=42)

    # --- optuna ---
    po = sub.add_parser("optuna", help="Run Optuna tuning with GPU-aware scheduling.")
    po.add_argument("--journal", type=str, default="optuna_wind_journal.log",
                    help="Path to Optuna journal file.")
    po.add_argument("--study", type=str, default="wind_gnn_gru",
                    help="Study name.")
    po.add_argument("--trials", type=int, default=30, help="Number of trials.")
    po.add_argument("--jobs", type=int, default=1, help="Parallel workers (threads).")
    po.add_argument("--epochs", type=int, default=20, help="Epochs per trial.")
    po.add_argument("--n2v_epochs", type=int, default=3, help="Node2Vec pretrain epochs per trial.")
    po.add_argument("--seed", type=int, default=42, help="Random seed.")
    po.add_argument("--train_frac", type=float, default=0.8, help="Train split fraction.")
    po.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")
    po.add_argument("--obs_feats", type=str, default=None, help="Path to obs_feats .pt (T,N,Fobs)")
    po.add_argument("--y", type=str, default=None, help="Path to y .pt (T,N,1)")
    po.add_argument("--nwp", type=str, default=None, help="Path to nwp .pt (T,N,Fnwp)")
    po.add_argument("--coords", type=str, default=None, help="Path to coords .pt (N,2) or (N,D)")
    po.add_argument("--T", type=int, default=4000, help="Dummy T if tensors not provided.")
    po.add_argument("--N", type=int, default=200, help="Dummy N if tensors not provided.")
    po.add_argument("--Fobs", type=int, default=6, help="Dummy Fobs.")
    po.add_argument("--Fnwp", type=int, default=8, help="Dummy Fnwp.")
    po.add_argument("--Dcoord", type=int, default=2, help="Dummy coord dim.")
    po.add_argument("--pruner", type=str, default="median", choices=["median", "sha", "hyperband"],
                    help="Pruner type.")
    # GPU manager
    po.add_argument("--gpu-ids", type=str, default=None,
                    help="Comma-separated GPU ids to use (e.g., '0,1'). Default: detect all.")
    po.add_argument("--gpu-lock-dir", type=str, default=".gpu_locks",
                    help="Directory for GPU lock files.")
    po.add_argument("--gpu-strict", type=int, default=1,
                    help="1: wait for free GPU; 0: fall back to CPU if busy.")
    po.add_argument("--gpu-timeout", type=int, default=None,
                    help="Seconds to wait for a free GPU (None=infinite).")

    # --- analyze ---
    pa = sub.add_parser("analyze", help="Render Optuna visualizations from journal.")
    pa.add_argument("--journal", required=True, help="Path to Optuna journal file.")
    pa.add_argument("--study", required=True, help="Study name.")
    pa.add_argument("--outdir", default="optuna_reports", help="Output directory for HTML/PNGs.")
    pa.add_argument("--include-pruned", type=int, default=1, help="Include pruned trials flag.")
    pa.add_argument("--importance-evaluator", choices=["fanova", "mean", "shap"], default="fanova",
                    help="Param importance evaluator.")
    pa.add_argument("--png", type=int, default=1, help="Export PNGs if kaleido available (1=yes).")

    return p


def main():
    # parser = build_parser()
    # args = parser.parse_args()
    cmd = 'optuna'
    args = BuildParser(cmd)
    if args.cmd == "demo":
        run_demo(args)
    elif args.cmd == "optuna":
        run_optuna(args)
    elif args.cmd == "analyze":
        run_analyze(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
# ```
#
# ### Run examples
#
# Quick demo on dummy data:
#
# ```bash
# python wind_st_all_in_one.py demo
# ```
#
# Optuna tuning with GPU-aware scheduling (two processes, shared journal):
#
# ```bash
# # Terminal 1
# python wind_st_all_in_one.py optuna --trials 60 --jobs 1 --gpu-ids 0,1 --journal optuna_wind_journal.log
# # Terminal 2
# python wind_st_all_in_one.py optuna --trials 60 --jobs 1 --gpu-ids 0,1 --journal optuna_wind_journal.log
# ```
#
# Visualization:
#
# ```bash
# python wind_st_all_in_one.py analyze --journal optuna_wind_journal.log --study wind_gnn_gru --outdir optuna_reports
# ```
#
# If you want this split back into three files again (`wind_st_template.py`, `run_optuna.py`, `analyze_study.py`),
# say the word and I’ll output them neatly.
