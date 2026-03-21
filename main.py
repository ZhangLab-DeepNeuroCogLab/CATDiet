import os
from datetime import datetime
from pathlib import Path
from typing import Dict
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    DeviceStatsMonitor,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero
import hydra
from omegaconf import DictConfig, OmegaConf
import linear_eval
from models_collection import (
    dino,
    simclr
)
from transforms_ffcv.dataloaders import get_val_loader
from transforms_ffcv.simclr_transform import (
    ADiet_ffcv_loader as simclr_ADiet,
    CDiet_ffcv_loader as simclr_CDiet,
    TDiet_ffcv_loader as simclr_TDiet,
    CATDiet_ffcv_loader as simclr_CATDiet,
    CombDiet_ffcv_loader as simclr_CombDiet,
)
from transforms_ffcv.dino_transform import (
    ADiet_ffcv_loader as dino_ADiet,
    CDiet_ffcv_loader as dino_CDiet,
    TDiet_ffcv_loader as dino_TDiet,
    CATDiet_ffcv_loader as dino_CATDiet,
    CombDiet_ffcv_loader as dino_CombDiet,
)

METHODS = {
    "simclr": {
        "model": simclr.SimCLR,
        "dataloaders": {
            "Acuity": simclr_ADiet,
            "Color": simclr_CDiet,
            "Temporal": simclr_TDiet,
            "CAT": simclr_CATDiet,
            "Comb": simclr_CombDiet,
        },
    },
    "dino": {
        "model": dino.DINO,
        "dataloaders": {
            "Acuity": dino_ADiet,
            "Color": dino_CDiet,
            "Temporal": dino_TDiet,
            "CAT": dino_CATDiet,
            "Comb": dino_CombDiet,
        },
    },
    
}


@hydra.main(config_path="configs", config_name="pretrain_co3d")
def main(cfg: DictConfig) -> None:
    print_rank_zero(f"Args: {OmegaConf.to_yaml(cfg)}")
    print_rank_zero(f"Working directory : {os.getcwd()}")
    print_rank_zero(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    seed_everything(cfg.seed, workers=True, verbose=True)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    method_names = cfg.methods
    
    
    # logging
    wandb_logger = WandbLogger(
        project="baby-vision",
        name=cfg.run_name,
        group=f"co3d-{method_names}",
    )
    cfg.log_dir = Path(cfg.log_dir)
    

    method_cfg = METHODS[method_names]
    # get dataloaders
    dataloader_fns = method_cfg.get("dataloaders")
    train_dataloader = dataloader_fns.get(cfg.diet_name)(cfg)
    val_dataloader = get_val_loader(cfg.batch_size_per_device,cfg.num_workers,cfg.val_dir)
    print_rank_zero(f"[info] stage duration schedule: {cfg.stage_epochs}")
    
    if cfg.skip_pretrain:
        if cfg.ckpt_path is None:
            raise ValueError("cfg.ckpt_path must be provided when skip_pretrain=True")
        ckpt = torch.load(cfg.ckpt_path)
        model = METHODS[method_names]["model"](
            batch_size_per_device=cfg.batch_size_per_device,
            num_classes=cfg.num_classes,
            stage_epochs=cfg.stage_epochs,
            reload_freq=cfg.reload_freq,
            dataloaders=train_dataloader,
            val_loader=val_dataloader,
            scheme=cfg.scheme_id,
            backbone=cfg.backbone,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = METHODS[method_names]["model"](
            batch_size_per_device=cfg.batch_size_per_device,
            num_classes=cfg.num_classes,
            stage_epochs=cfg.stage_epochs,
            reload_freq=cfg.reload_freq,
            dataloaders=train_dataloader,
            val_loader=val_dataloader,
            scheme=cfg.scheme_id,
            backbone=cfg.backbone,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Train model.
        metric_callback = MetricCallback()

        trainer = Trainer(
            max_epochs=sum(cfg.stage_epochs),
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            callbacks=[
                ModelCheckpoint(
                    dirpath=f"checkpoints/pretrain",
                    filename="epoch-{epoch:03d}-step-{step}",
                    save_last=True,
                    every_n_epochs=cfg.log_ckpt_freq,
                    save_top_k=-1,
                ),
                LearningRateMonitor(),
                DeviceStatsMonitor(),
                metric_callback,
            ],
            logger=wandb_logger,
            precision=cfg.precision,
            strategy=cfg.strategy,
            sync_batchnorm=cfg.accelerator != "cpu",
        )

        trainer.fit(model=model, ckpt_path=cfg.ckpt_path) # resume pretraining
        for metric in ["val_online_cls_top1", "val_online_cls_top5"]:
            print_rank_zero(
                f"max {metric}: {max(metric_callback.val_metrics.get(metric, [-1]))}"
            )

    eval_metrics: Dict[str, Dict[str, float]] = dict()

    if cfg.skip_linear_probe:
        print_rank_zero("Skipping linear probe.")
    else:
        eval_metrics["linear"] = linear_eval.linear_eval(
            model=model,
            train_dir=Path(cfg.linear_train_dir),
            val_dir=Path(cfg.val_dir),
            ood_dir=Path(cfg.ood_dir),
            batch_size_per_device=cfg.batch_size_per_device,
            num_workers=cfg.num_workers,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
            strategy=cfg.strategy,
            num_classes=cfg.num_classes_linear,
            scheme_id=cfg.scheme_id,
            skip_linear_train=cfg.skip_linear_train,
            linear_ckpt_path=cfg.lin_ckpt_path,
        )


    if eval_metrics:
        print_rank_zero(f"Results for {method_names}:")
        print_rank_zero(eval_metrics_to_markdown(eval_metrics))



def eval_metrics_to_markdown(metrics: Dict[str, Dict[str, float]]) -> str:
    EVAL_NAME_COLUMN_NAME = "Eval Name"
    METRIC_COLUMN_NAME = "Metric Name"
    VALUE_COLUMN_NAME = "Value"

    eval_name_max_len = max(
        len(eval_name) for eval_name in list(metrics.keys()) + [EVAL_NAME_COLUMN_NAME]
    )
    metric_name_max_len = max(
        len(metric_name)
        for metric_dict in metrics.values()
        for metric_name in list(metric_dict.keys()) + [METRIC_COLUMN_NAME]
    )
    value_max_len = max(
        len(metric_value)
        for metric_dict in metrics.values()
        for metric_value in list(f"{value:.2f}" for value in metric_dict.values())
        + [VALUE_COLUMN_NAME]
    )

    header = f"| {EVAL_NAME_COLUMN_NAME.ljust(eval_name_max_len)} | {METRIC_COLUMN_NAME.ljust(metric_name_max_len)} | {VALUE_COLUMN_NAME.ljust(value_max_len)} |"
    separator = f"|:{'-' * (eval_name_max_len)}:|:{'-' * (metric_name_max_len)}:|:{'-' * (value_max_len)}:|"

    lines = [header, separator] + [
        f"| {eval_name.ljust(eval_name_max_len)} | {metric_name.ljust(metric_name_max_len)} | {f'{metric_value:.2f}'.ljust(value_max_len)} |"
        for eval_name, metric_dict in metrics.items()
        for metric_name, metric_value in metric_dict.items()
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
