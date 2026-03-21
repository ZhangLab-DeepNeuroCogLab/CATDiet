import json
import os
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torchvision import transforms
from torchvision.datasets import ImageFolder

from lightly.utils.benchmarking import LinearClassifier
from lightly.utils.dist import print_rank_zero

from models_collection import dino, simclr

METHODS = {
    "simclr": simclr.SimCLR,
    "dino": dino.DINO,
}

class DummyLoader:
    def __len__(self) -> int:
        return 1
def get_eval_setup(map_path):
    
    
    model_classes = ["cat", "chair", "car"]
    dataset_to_model = {
        "cat": "cat",
        "chair": "chair",
        "car": "car",
    }
    
    with open(map_path, "r") as f:
        class_map = json.load(f)

    idx_to_name = {v: k for k, v in class_map.items()}
    considered_indices = [class_map[c] for c in model_classes]

    return class_map, idx_to_name, dataset_to_model, considered_indices

def load_classifier(cfg: DictConfig, device: str):
    if cfg.methods not in METHODS:
        raise ValueError(f"Unknown method: {cfg.methods}")
    dummy_loader = DummyLoader()
    model = METHODS[cfg.methods](
        batch_size_per_device=cfg.batch_size_per_device,
        num_classes=cfg.num_classes,
        scheme=cfg.scheme_id,
        stage_epochs=cfg.stage_epochs,
        reload_freq=cfg.reload_freq,
        dataloaders=[dummy_loader],
        val_loader=dummy_loader,
        backbone=cfg.backbone
    )
    
    model_ckpt = torch.load(cfg.ckpt_path)
    model.load_state_dict(model_ckpt["state_dict"])
    model.eval()
    lin_ckpt = torch.load(cfg.lin_ckpt_path)
    classifier = LinearClassifier(
        model=model,
        batch_size_per_device=cfg.batch_size_per_device,
        feature_dim=model.online_classifier.feature_dim,
        num_classes=cfg.num_classes_linear,
    )
    classifier.load_state_dict(lin_ckpt["state_dict"])
    classifier.eval()
    classifier.to(device)
    return classifier

def build_eval_dataset(dataset_root: str):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return ImageFolder(root=dataset_root, transform=transform)

def evaluate_silhouette_only_dataset(
    dataset: ImageFolder,
    classifier,
    dataset_to_model: Dict[str, str],
    considered_indices: List[int],
    idx_to_name: Dict[int, str],
    device: str,
):
    class_names = dataset.classes

    correct_shape = 0
    incorrect = 0
    total = 0

    for image, label in dataset:
        shape_class = class_names[label]
        if shape_class not in dataset_to_model:
            continue

        shape_model = dataset_to_model[shape_class]

        with torch.no_grad():
            output = classifier(image.unsqueeze(0).to(device))

        considered_logits = output[0, considered_indices]
        pred_idx = considered_indices[considered_logits.argmax().item()]
        pred_model = idx_to_name[pred_idx]

        total += 1
        if pred_model == shape_model:
            correct_shape += 1
        else:
            incorrect += 1

    acc = correct_shape / total if total > 0 else 0.0

    return {
        "acc": acc,
        "correct": correct_shape,
        "incorrect": incorrect,
        "total": total,
    }
    
def save_results(output_path: str, metrics: Dict[str, float]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as f:
        f.write(f"acc:{metrics['acc']:.6f}\n")
        f.write("=======================\n")
        
@hydra.main(config_path="configs", config_name="eval_silhouette", version_base=None)
def main(cfg: DictConfig) -> None:
    print_rank_zero(f"Args:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.seed, workers=True, verbose=True)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    
    classifier = load_classifier(cfg, device="cuda:0")

    _, idx_to_name, dataset_to_model, considered_indices = get_eval_setup(
        cfg.map_path
    )

    dataset = build_eval_dataset(cfg.dataset_root)

    metrics = evaluate_silhouette_only_dataset(
        dataset=dataset,
        classifier=classifier,
        dataset_to_model=dataset_to_model,
        considered_indices=considered_indices,
        idx_to_name=idx_to_name,
        device="cuda:0",
    )

    print_rank_zero(f"acc: {metrics['acc']:.6f}")
    print_rank_zero(f"incorrect: {metrics['incorrect']} / {metrics['total']}")

    save_results(
        output_path=f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/silhouette.txt",
        metrics=metrics,
    )

if __name__ == "__main__":
    main()