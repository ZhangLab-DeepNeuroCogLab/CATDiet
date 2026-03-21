from pathlib import Path
from typing import Dict
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.nn import Module

from lightly.utils.benchmarking import LinearClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero
from transforms_ffcv.dataloaders import (
    get_train_loader,
    get_val_loader,
    get_eval_loader,
)


def linear_eval(
    model: Module,
    train_dir: Path,
    val_dir: Path,
    ood_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    precision: str,
    strategy: str,
    num_classes: int,
    scheme_id: str,
    skip_linear_train,
    linear_ckpt_path,
) -> Dict[str, float]:
    """Runs a linear evaluation on the given model.


    The most important settings are:
        - Backbone: Frozen
        - Epochs: 90
        - Optimizer: SGD
        - Base Learning Rate: 0.1
        - Momentum: 0.9
        - Weight Decay: 0.0
        - LR Schedule: Cosine without warmup

    """
    print_rank_zero("Running linear evaluation...")

    metrics_dict: Dict[str, float] = dict()
    classifier = LinearClassifier(
            model=model,
            batch_size_per_device=batch_size_per_device,
            feature_dim=model.online_classifier.feature_dim,
            num_classes=num_classes,
        )
    if skip_linear_train:
        if linear_ckpt_path is None:
            raise ValueError("linear_ckpt_path must be provided when skip_linear_train=True")
        ckpt = torch.load(linear_ckpt_path)
        classifier.load_state_dict(ckpt["state_dict"])
    else:
        # Setup training and val data.
        train_dataloader = get_train_loader(
            batch_size=batch_size_per_device,
            num_workers=num_workers,
            root_dir=train_dir,
        )
        val_dataloader = get_val_loader(
            batch_size=batch_size_per_device, num_workers=num_workers, root_dir=val_dir
        )
        classifier = LinearClassifier(
            model=model,
            batch_size_per_device=batch_size_per_device,
            feature_dim=model.online_classifier.feature_dim,
            num_classes=num_classes,
        )
        ## Train linear classifier.
        metric_callback = MetricCallback()
        trainer = Trainer(
            max_epochs=90,
            accelerator=accelerator,
            devices=devices,
            callbacks=[
                ModelCheckpoint(
                    dirpath=f"checkpoints/linprobe",
                    filename="best",  
                    save_top_k=1,
                    monitor="val_offline_cls_top1", 
                    mode="max",  
                    save_last=True,  
                ),
                LearningRateMonitor(),
                DeviceStatsMonitor(),
                metric_callback,
            ],
            logger=(WandbLogger(project="baby-vision", name=f"{scheme_id}")),
            precision=precision,
            strategy=strategy,
            num_sanity_val_steps=0,  
        )
        trainer.fit(
            model=classifier,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=linear_ckpt_path,
        )
        
        
    ood_loaders = get_ood_loaders_ic(
        val_dir=val_dir,
        ood_dir=ood_dir,
        batch_size=batch_size_per_device,
        num_workers=num_workers
    )
    
    ood_metrics = evaluate_on_ood_loaders(classifier, ood_loaders, output_path=f"{scheme_id}.txt")
    metrics_dict.update(ood_metrics)
    return metrics_dict


def evaluate_on_ood_loaders(
    model, ood_loaders_dict, output_path="./ood_results.txt", precision="bf16-mixed"
):
    trainer = Trainer(logger=False, inference_mode=False, precision=precision)
    results = [] # save the details
    results_dict = {} # for later aggregation

    for name, loader in ood_loaders_dict.items():
        print(f"Evaluating on OOD dataset: {name}")
        metrics = trainer.validate(model, dataloaders=loader, verbose=False)
        results_dict[name] = metrics[0] # the top-1 acc
        result_str = f"{name}: {metrics}\n"
        print(result_str.strip())
        results.append(result_str)

    output_path = Path(output_path)
    if output_path.parent != Path("."):  # avoid creating '' dir
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as f:
        f.writelines(results)
    print(f"OOD evaluation results saved to {output_path}")
    summary_metrics = compute_ic_metrics_from_results(results_dict)
    return summary_metrics


def get_ood_loaders_ic(val_dir,ood_dir, batch_size=128, num_workers=0):
    corruptions = [
        "glass_blur",
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    print("Using beton loader")
    dataloaders = {}
    dataloaders["normal"] = get_val_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        root_dir=val_dir,
    )
    ood_dir = Path(ood_dir)
    for cor_name in corruptions:
        for ser in range(1, 6):
            corrupts = f"imagenet_c_{cor_name}_{ser}"
            img_dir = ood_dir / f"{cor_name}_{ser}.beton"
            dataloaders[corrupts] = get_eval_loader(batch_size, num_workers, img_dir)
    return dataloaders


def compute_ic_metrics_from_results(results_dict):
    alexnet_baseline = {
        "gaussian_noise": 0.886,
        "shot_noise": 0.894,
        "impulse_noise": 0.923,
        "defocus_blur": 0.820,
        "glass_blur": 0.826,
        "motion_blur": 0.786,
        "zoom_blur": 0.798,
        "snow": 0.867,
        "frost": 0.827,
        "fog": 0.819,
        "brightness": 0.565,
        "contrast": 0.853,
        "elastic_transform": 0.646,
        "pixelate": 0.718,
        "jpeg_compression": 0.607,
    }

    corruption_order = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]

    # normal top1
    if "normal" not in results_dict:
        raise ValueError("'normal' not found in OOD evaluation results.")

    normal_top1 = results_dict["normal"]["val_offline_cls_top1"] * 100

    # collect corruption accuracies across severities
    corruption_to_accs = {c: [] for c in corruption_order}

    for name, metrics in results_dict.items():
        if not name.startswith("imagenet_c_"):
            continue

        # e.g. imagenet_c_glass_blur_1
        parts = name.split("_")
        corruption = "_".join(parts[2:-1])
        severity = int(parts[-1])  # not strictly needed, but useful for sanity check

        if corruption not in corruption_to_accs:
            continue

        corruption_to_accs[corruption].append(metrics["val_offline_cls_top1"])

    # mean accuracy over severities for each corruption
    mean_ce_terms = []
    for corruption in corruption_order:
        accs = corruption_to_accs[corruption]
        if len(accs) == 0:
            raise ValueError(f"No results found for corruption: {corruption}")

        mean_acc = sum(accs) / len(accs)
        ce = (1.0 - mean_acc) / alexnet_baseline[corruption]
        mean_ce_terms.append(ce)

    ic_meanCE = sum(mean_ce_terms) / len(mean_ce_terms) * 100.0

    return {
        "clean_Acc": normal_top1,
        "corrupted_mCE": ic_meanCE,
    }