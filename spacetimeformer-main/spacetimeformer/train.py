from argparse import ArgumentParser
import random
import sys
import warnings
import os
import uuid

import pytorch_lightning as pl
import torch

import spacetimeformer as stf

_MODELS = ["spacetimeformer", "mtgnn", "lstnet"]

_DSETS = [
    "mnist",
    "cifar",
]


def create_parser(params, dset, model):


    if dset == "mnist":
        p = stf.data.image_completion.MNISTDset.add_cli()
    else:
        p = stf.data.image_completion.CIFARDset.add_cli()
    params.update(p)
    params.update(stf.data.DataModule.add_cli())

    if model == "lstnet":
        params.update(stf.lstnet_model.LSTNet_Forecaster.add_cli())
    elif model == "mtgnn":
        params.update(stf.mtgnn_model.MTGNN_Forecaster.add_cli())
    elif model == "heuristic":
        params.update(stf.heuristic_model.Heuristic_Forecaster.add_cli())
    elif model == "spacetimeformer":
        params.update(stf.spacetimeformer_model.Spacetimeformer_Forecaster.add_cli())

    params.update(stf.callbacks.TimeMaskedLossCallback.add_cli())

    return params

class make_config:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)

def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "mnist":
        x_dim = 1
        yc_dim = 28
        yt_dim = 28
    else:
        x_dim = 1
        yc_dim = 3
        yt_dim = 3

    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "heuristic":
        forecaster = stf.heuristic_model.Heuristic_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            loss=config.loss,
            method=config.method,
        )
    elif config.model == "mtgnn":
        forecaster = stf.mtgnn_model.MTGNN_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            gcn_depth=config.gcn_depth,
            dropout_p=config.dropout_p,
            node_dim=config.node_dim,
            dilation_exponential=config.dilation_exponential,
            conv_channels=config.conv_channels,
            subgraph_size=config.subgraph_size,
            skip_channels=config.skip_channels,
            end_channels=config.end_channels,
            residual_channels=config.residual_channels,
            layers=config.layers,
            propalpha=config.propalpha,
            tanhalpha=config.tanhalpha,
            learning_rate=config.learning_rate,
            kernel_size=config.kernel_size,
            l2_coeff=config.l2_coeff,
            time_emb_dim=config.time_emb_dim,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_revin=config.use_revin,
        )
    elif config.model == "lstnet":
        forecaster = stf.lstnet_model.LSTNet_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            hidSkip=config.hidSkip,
            CNN_kernel=config.CNN_kernel,
            skip=config.skip,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
        )
    else:
        if hasattr(config, "context_points") and hasattr(config, "target_points"):
            max_seq_len = config.context_points + config.target_points
        elif hasattr(config, "max_len"):
            max_seq_len = config.max_len
        else:
            raise ValueError("Undefined max_seq_len")
        forecaster = stf.spacetimeformer_model.Spacetimeformer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            d_queries_keys=config.d_qk,
            d_values=config.d_v,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_attn_out=config.dropout_attn_out,
            dropout_attn_matrix=config.dropout_attn_matrix,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            pos_emb_type=config.pos_emb_type,
            use_final_norm=not config.no_final_norm,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            attn_time_windows=config.attn_time_windows,
            use_shifted_time_windows=config.use_shifted_time_windows,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            class_loss_imp=config.class_loss_imp,
            recon_loss_imp=config.recon_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
            pad_value=config.pad_value,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_val=not config.no_val,
            use_time=not config.no_time,
            use_space=not config.no_space,
            use_given=not config.no_given,
            recon_mask_skip_all=config.recon_mask_skip_all,
            recon_mask_max_seq_len=config.recon_mask_max_seq_len,
            recon_mask_drop_seq=config.recon_mask_drop_seq,
            recon_mask_drop_standard=config.recon_mask_drop_standard,
            recon_mask_drop_full=config.recon_mask_drop_full,
        )

    return forecaster


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None

    if config.dset == "mnist":
        config.target_points = 28 - config.context_points
        datasetCls = stf.data.image_completion.MNISTDset
        PLOT_VAR_IDXS = [18, 24]
        PLOT_VAR_NAMES = ["18th row", "24th row"]
    else:
        config.target_points = 32 * 32 - config.context_points
        datasetCls = stf.data.image_completion.CIFARDset
        PLOT_VAR_IDXS = [0]
        PLOT_VAR_NAMES = ["Reds"]
    DATA_MODULE = stf.data.DataModule(
        datasetCls=datasetCls,
        dataset_kwargs={"context_points": config.context_points},
        batch_size=config.batch_size,
        workers=config.workers,
        )


    return (
        DATA_MODULE,
        INV_SCALER,
        SCALER,
        NULL_VAL,
        PLOT_VAR_IDXS,
        PLOT_VAR_NAMES,
        PAD_VAL,
    )


def create_callbacks(config, save_dir):
    filename = f"{config.run_name}_" + str(uuid.uuid1()).split("-")[0]
    model_ckpt_dir = os.path.join(save_dir, filename)
    config.model_ckpt_dir = model_ckpt_dir
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=model_ckpt_dir,
        monitor="val/loss",
        mode="min",
        filename=f"{config.run_name}" + "{epoch:02d}",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    callbacks = [saving]

    callbacks.append(
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val/loss",
            patience=100,
        )
    )


    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                steps=config.teacher_forcing_anneal_steps,
            )
        )
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.time_mask_end,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks


def main(params):

    log_dir = ('/home/smartrue/Dropbox/current_codes/PycharmProjects/Satellites_projects'
               '/image2power/spacetimeformer-main/spacetimeformer/log')

    # Dset
    (
        data_module,
        inv_scaler,
        scaler,
        null_val,
        plot_var_idxs,
        plot_var_names,
        pad_val,
    ) = create_dset(params)

    # Model
    params.null_value = null_val
    params.pad_value = pad_val
    forecaster = create_model(params)
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)

    # Callbacks
    callbacks = create_callbacks(params, save_dir=log_dir)
    test_samples = next(iter(data_module.test_dataloader()))

    # if params.dset in ["mnist", "cifar"]:
    #     callbacks.append(
    #         stf.plot.ImageCompletionCallback(
    #             test_samples,
    #             total_samples=min(16, params.batch_size),
    #             mode="left-right" if params.dset == "mnist" else "flat",
    #         )
    #     )
    #
    # if params.model == "spacetimeformer":
    #
    #     callbacks.append(
    #         stf.plot.AttentionMatrixCallback(
    #             test_samples,
    #             layer=0,
    #             total_samples=min(16, params.batch_size),
    #         )
    #     )



    val_control = {"check_val_every_n_epoch": 100}

    trainer = pl.Trainer(
        callbacks=callbacks,
        strategy='ddp_find_unused_parameters_true',
        logger=None,
        accelerator="gpu",
        max_epochs=10,
        gradient_clip_val=5,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=4,
        sync_batchnorm=True,
        **val_control,
    )

    # Train
    trainer.fit(forecaster, datamodule=data_module)

    # Test
    trainer.test(datamodule=data_module, ckpt_path="best")

    # Predict (only here as a demo and test)
    # forecaster.to("cuda")
    # xc, yc, xt, _ = test_samples
    # yt_pred = forecaster.predict(xc, yc, xt)



if __name__ == "__main__":
    for model in _MODELS:
        for dset in _DSETS:
            parser = dict()
            parser = create_parser(parser, dset, model)
            parser = make_config(parser)
            parser.dset = dset
            parser.model = model
            parser.run_name = f'{model}_{dset}'
            main(parser)
