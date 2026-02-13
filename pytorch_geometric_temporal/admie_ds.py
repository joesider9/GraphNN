import os
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from einops import rearrange, repeat
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ADMIEDataset(Dataset):
    def __init__(
            self,
            path_data,
            n_lags,
            n_pred_lags,
            n_per_hour,
            rated,
            task='train',
            data_struct='lstm',
            return_graph=False,
            graph_topk=10,
            graph_self_loops=False,
            return_edge_tau=True,
    ):
        self.path_data = path_data
        data = joblib.load(os.path.join(path_data, 'data_parks_imputed.pickle'))
        self.park_names = data['park_names']
        self.dates = data['dates']
        self.X = dict()
        norm_x = np.array([30, 360])[np.newaxis, np.newaxis, :]
        self.X = torch.from_numpy(np.clip(data['X'] / norm_x, 0, 1)).float()
        norm_y = np.array(rated)[np.newaxis, :, np.newaxis]
        self.y = torch.from_numpy(np.clip(data['y'] / norm_y, 0, 1)).float()
        self.spatial = data['spatial']
        self.indices = data['indices']
        self.pred_lags = list(range(n_pred_lags))
        self.lags = list(range(n_lags))
        self.n_lags = n_lags
        self.n_per_hour = n_per_hour
        self.n_hour = n_lags // n_per_hour
        self.data_struct = data_struct
        self.return_graph = return_graph
        self.graph_topk = graph_topk
        self.graph_self_loops = graph_self_loops
        self.return_edge_tau = return_edge_tau
        if task == 'train':
            self.indices = self.indices[12:int(len(self.indices) * 0.7)]
        elif task == 'val':
            self.indices = self.indices[int(len(self.indices) * 0.7):int(len(self.indices) * 0.85)]
        elif task == 'test':
            self.indices = self.indices[int(len(self.indices) * 0.85):]

    def __len__(self):
        return len(self.indices)

    def corr_windows_at_idx(self, X, idx: int, w: int = 12, H: int = 37, eps: float = 1e-6):
        """
        X: [T, N, 1] (or [T, N]) tensor
        idx: int, must satisfy idx - w - (H-1) >= 0
        w: window length (12)
        H: number of lags (37 for h=0..36)

        returns corr: [H, N, N] where
          corr[h, i, j] = corr( X[idx-w:idx, i], X[idx-h-w:idx-h, j] )
        """
        if X.dim() == 3:
            X2 = X[..., 0]  # [T, N]
        else:
            X2 = X  # [T, N]

        # A: base window [w, N]
        A = X2[idx - w: idx]  # [w, N]

        # B: stacked lagged windows [H, w, N]
        hs = torch.arange(H, device=X2.device)  # [H]
        starts = (idx - w) - hs  # [H]
        t = starts[:, None] + torch.arange(w, device=X2.device)[None]  # [H, w]
        B = X2[t]  # [H, w, N]

        # z-score A over time dimension
        A0 = A - A.mean(dim=0, keepdim=True)
        A0 = A0 / (A0.std(dim=0, keepdim=True, unbiased=False) + eps)  # [w, N]

        # z-score each B[h] over its time dimension
        B0 = B - B.mean(dim=1, keepdim=True)
        B0 = B0 / (B0.std(dim=1, keepdim=True, unbiased=False) + eps)  # [H, w, N]

        # corr[h, i, j] = mean_t A0[t,i] * B0[h,t,j]
        corr = torch.einsum("ti,htj->hij", A0, B0) / w  # [H, N, N]
        return corr

    def sparse_graph_at_idx(self, idx: int, eps: float = 1e-6):
        X = self.X
        C = self.corr_windows_at_idx(
            X,
            idx,
            w=self.n_per_hour,
            H=self.n_lags - self.n_per_hour,
            eps=eps,
        )  # [H, N, N]
        A, tau = C.max(dim=0)  # A: [N, N], tau: [N, N]

        if not self.graph_self_loops:
            A.fill_diagonal_(0)

        N = A.size(0)
        k = min(int(self.graph_topk), int(N))
        vals, js = A.topk(k, dim=1)  # [N, k]
        targets = torch.arange(N, device=A.device).unsqueeze(1).expand(N, k)
        sources = js
        edge_index = torch.stack([targets.reshape(-1), sources.reshape(-1)], dim=0)
        edge_weight = vals.reshape(-1)

        if self.return_edge_tau:
            edge_tau = tau[targets, sources].reshape(-1).to(torch.long)
        else:
            edge_tau = None

        return edge_index, edge_weight, edge_tau

    def get_correlation(self, idx):
        farms = self.X.shape[1]
        correlation = torch.zeros((self.n_lags - self.n_per_hour, farms, farms))
        for i in range(farms):
            for j in range(i, farms):
                for h in range(self.n_lags - self.n_per_hour):
                    correlation[h, i, j] = torch.corrcoef(torch.stack([self.X[idx - self.n_per_hour:idx, i, 0],
                                                                       self.X[idx - h - self.n_per_hour:idx - h , j, 0]]))[0, 1]
        return correlation.flip(dims=(0,))

    def __getitem__(self, i):
        idx = self.indices[i]

        ind = [idx - l for l in self.lags]
        past = self.X[ind].flip(dims=(0,))
        edge_index, edge_weight, edge_tau = self.sparse_graph_at_idx(idx)
        edge_tau = past[edge_tau, edge_index[1,:], 0]
        ind = [idx + l for l in self.pred_lags]
        target = rearrange(self.y[ind].squeeze(-1), 'l N -> N l')
        if torch.isnan(target).any() or torch.isnan(past).any() or torch.isnan(edge_index).any() or torch.isnan(edge_tau).any():
            return None, None, None, None, None
        return past, target, edge_index, edge_tau


