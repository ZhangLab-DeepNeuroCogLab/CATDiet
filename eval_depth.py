from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

import torch

from lightly.utils.dist import print_rank_zero
from lightly.utils.benchmarking import LinearClassifier, MetricCallback
from torchvision import transforms
from models_collection import dino, simclr

METHODS = {
    "simclr": simclr.SimCLR,
    "dino": dino.DINO,
}
class DummyLoader:
    def __len__(self) -> int:
        return 1
    

class ParquetTransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if image.mode == "RGBA":
            image = image.convert("RGB")
        label = int(item["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_depth_dataloaders(cfg: DictConfig):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset_dict = load_dataset(
        "parquet",
        data_files={
            "train": cfg.train_parquet,
            "test": cfg.test_parquet,
        },
    )

    train_dataset = ParquetTransformDataset(dataset_dict["train"], transform=transform)
    test_dataset = ParquetTransformDataset(dataset_dict["test"], transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_per_device,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size_per_device,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, test_loader


def load_encoder(cfg: DictConfig):
    model_cls = METHODS[cfg.methods]
    dummy_loader = DummyLoader()
    ckpt = torch.load(cfg.ckpt_path)
    model = model_cls(
        batch_size_per_device=cfg.batch_size_per_device,
        num_classes=cfg.num_classes,
        scheme=cfg.scheme_id,
        stage_epochs=cfg.stage_epochs,
        reload_freq=cfg.reload_freq,
        dataloaders=[dummy_loader],
        val_loader=dummy_loader,
        backbone=cfg.backbone
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@hydra.main(config_path="configs", config_name="eval_depth", version_base=None)
def main(cfg: DictConfig) -> None:
    print_rank_zero(f"Args:\n{OmegaConf.to_yaml(cfg)}")
    seed_everything(cfg.seed, workers=True, verbose=True)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    train_loader, test_loader = build_depth_dataloaders(cfg)
    model = load_encoder(cfg)

    if cfg.skip_depth_train:
        ckpt = torch.load(cfg.lin_ckpt_path)
        classifier = LinearClassifier(
            model=model,
            batch_size_per_device=cfg.batch_size_per_device,
            feature_dim=model.online_classifier.feature_dim,
            num_classes=2,
            topk=(1,),
        )
        classifier.load_state_dict(ckpt["state_dict"])
        trainer = Trainer(logger=False, inference_mode=False, precision=cfg.precision)
    else:
        metric_callback = MetricCallback()
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/linprobe-depth",
            filename="best",
            save_top_k=1,
            monitor="val_offline_cls_top1",
            mode="max",
            save_last=True,
        )
        trainer = Trainer(
            max_epochs=cfg.epochs,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            callbacks=[checkpoint_callback, metric_callback],
            logger=WandbLogger(project="baby-vision", name=f"depth-linpb-{cfg.scheme_id}"),
            precision=cfg.precision,
            strategy=cfg.strategy,
            num_sanity_val_steps=0,
        )

        classifier = LinearClassifier(
            model=model,
            batch_size_per_device=cfg.batch_size_per_device,
            feature_dim=model.online_classifier.feature_dim,
            num_classes=2,
            topk=(1,),
        )

        trainer.fit(
            model=classifier,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
        )

    results = []
    metrics = trainer.test(model=classifier, dataloaders=test_loader, verbose=False)
    print(f"test: {metrics}\n")
    results.append(f"test: {metrics}\n")
    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "linpb-depth.txt"
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as f:
        f.writelines(results)



if __name__ == "__main__":
    main()
