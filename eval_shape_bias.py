import json
import os
from typing import Dict, List, Tuple
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torchvision import transforms
from torchvision.datasets import ImageFolder
from lightly.utils.benchmarking import LinearClassifier
from lightly.utils.dist import print_rank_zero
from models_collection import (
    dino, 
    simclr,
)
from lightly.utils.dist import print_rank_zero


METHODS = {
    "simclr": simclr.SimCLR,
    "dino": dino.DINO,
}


class DummyLoader:
    """Minimal loader to satisfy model __init__ when only __len__ is needed."""

    def __len__(self) -> int:
        return 1


def extract_texture_class(filename: str) -> str:
    """
    Extract texture class from a filename like:
        airplane1-bicycle2.png -> bicycle
    """
    parts = filename.split("-")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {filename}")

    texture_part = os.path.splitext(parts[1])[0]
    for i, char in enumerate(texture_part):
        if char.isdigit():
            return texture_part[:i]
    return texture_part


def get_class_setup(map_path):
    """
    Return label indices of SAY classes that overlap with the cue-conflict dataset
    """
    
    with open(map_path, "r") as f:
        class_map = json.load(f)
    
    # labels of SAY class to use
    model_classes = ["cat", "chair", "car"] 
    
    # mapping of labels from cue-conflict dataset to SAY
    dataset_to_model = {
        "cat": "cat",
        "chair": "chair",
        "car": "car",
    }


    idx_to_name = {v: k for k, v in class_map.items()}
    considered_idx = [class_map[c] for c in model_classes]

    return class_map, idx_to_name, model_classes, dataset_to_model, considered_idx


def build_cue_conflict_dataset(dataset_root: str) -> ImageFolder:
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


def load_classifier_from_cfg(
    cfg: DictConfig,
    model_cfg: DictConfig,
    device: str,
) -> LinearClassifier:
    method_name = model_cfg.method
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}")
    dummy_loader = DummyLoader()
    print_rank_zero(f"Loading backbone {cfg.backbone} for linear classifier...")
    model = METHODS[method_name](
        batch_size_per_device=cfg.batch_size_per_device,
        num_classes=cfg.num_classes,
        scheme=cfg.scheme_id,
        stage_epochs=cfg.stage_epochs,
        reload_freq=cfg.reload_freq,
        dataloaders=[dummy_loader],
        val_loader=dummy_loader,
        backbone=cfg.backbone
    )
    
    print_rank_zero(f"loading {model_cfg.ckpt_path}")
    model_ckpt = torch.load(model_cfg.ckpt_path)
    model.load_state_dict(model_ckpt["state_dict"])
    model.eval()

    lin_ckpt = torch.load(model_cfg.lin_ckpt_path)
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


def load_all_classifiers(cfg: DictConfig, device: str) -> List[LinearClassifier]:
    classifiers: List[LinearClassifier] = []

    for model_cfg in cfg.models:
        classifier = load_classifier_from_cfg(
            cfg=cfg,
            model_cfg=model_cfg,
            device=device,
        )
        classifiers.append(classifier)

    return classifiers


def collect_predictions(
    dataset: ImageFolder,
    classifiers: List[LinearClassifier],
    dataset_to_model: Dict[str, str],
    considered_idx: List[int],
    idx_to_name: Dict[int, str],
    device: str,
) -> Tuple[List[Tuple[str, str]], List[List[str]]]:
    """
    Returns:
        labels: list of (shape_model, texture_model)
        preds_all: list over models, each is a list of predicted model-class strings
    """
    class_names = dataset.classes
    preds_all: List[List[str]] = [[] for _ in range(len(classifiers))]
    labels: List[Tuple[str, str]] = []

    for i, (image, label) in enumerate(dataset):
        shape_class = class_names[label]
        if shape_class not in dataset_to_model:
            continue
        shape_model = dataset_to_model[shape_class]

        image_path = dataset.samples[i][0]
        texture_class = extract_texture_class(os.path.basename(image_path))
        if texture_class not in dataset_to_model:
            continue
        texture_model = dataset_to_model[texture_class]

        # skip non cue-conflict samples
        if shape_model == texture_model:
            continue

        image = image.unsqueeze(0).to(device)
        labels.append((shape_model, texture_model))

        for m_i, clf in enumerate(classifiers):
            with torch.no_grad():
                logits = clf(image)
                logits = logits[0, considered_idx]
                pred_idx = considered_idx[logits.argmax().item()]
                pred_model = idx_to_name[pred_idx]
                preds_all[m_i].append(pred_model)

    return labels, preds_all


def is_correct(pred: str, shape_label: str, texture_label: str) -> bool:
    return pred == shape_label or pred == texture_label


def get_collectively_correct_indices(
    labels: List[Tuple[str, str]],
    preds_all: List[List[str]],
) -> List[int]:
    return [
        i
        for i, (shape_label, texture_label) in enumerate(labels)
        if all(
            is_correct(preds_all[m_i][i], shape_label, texture_label)
            for m_i in range(len(preds_all))
        )
    ]


def compute_shape_bias_for_model(
    model_idx: int,
    labels: List[Tuple[str, str]],
    preds_all: List[List[str]],
    collectively_correct_idx: List[int],
) -> Dict[str, float]:
    correct_shape = sum(
        1 for i in collectively_correct_idx if preds_all[model_idx][i] == labels[i][0]
    )
    correct_texture = sum(
        1 for i in collectively_correct_idx if preds_all[model_idx][i] == labels[i][1]
    )
    total = correct_shape + correct_texture
    shape_bias = correct_shape / total if total > 0 else 0.0

    return {
        "shape_bias": shape_bias,
        "correct_shape": float(correct_shape),
        "correct_texture": float(correct_texture),
        "total": float(total),
    }


@hydra.main(config_path="configs", config_name="eval_shape_bias", version_base=None)
def main(cfg: DictConfig) -> None:
    print_rank_zero(f"Args:\n{OmegaConf.to_yaml(cfg)}")
    print_rank_zero(f"Working directory: {os.getcwd()}")
    print_rank_zero(
        f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    seed_everything(cfg.seed, workers=True, verbose=True)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    classifiers = load_all_classifiers(cfg, device= "cuda:0")
    
    _, idx_to_name, _, dataset_to_model, considered_idx = get_class_setup(
        cfg.map_path
    )
    dataset = build_cue_conflict_dataset(cfg.cue_conflict_root)

    labels, preds_all = collect_predictions(
        dataset=dataset,
        classifiers=classifiers,
        dataset_to_model=dataset_to_model,
        considered_idx=considered_idx,
        idx_to_name=idx_to_name,
        device= "cuda:0",
    )

    collectively_correct_idx = get_collectively_correct_indices(labels, preds_all)

    print_rank_zero(f"Total cue-conflict images: {len(labels)}")
    print_rank_zero(f"Collectively correct: {len(collectively_correct_idx)}")

    all_results = {}
    for m_i in range(len(classifiers)):
        results = compute_shape_bias_for_model(
            model_idx=m_i,
            labels=labels,
            preds_all=preds_all,
            collectively_correct_idx=collectively_correct_idx,
        )
        all_results[f"model_{m_i+1}"] = results
        print_rank_zero(
            f"[Model {m_i+1}] Shape bias: {results['shape_bias']:.4f} "
            f"({int(results['correct_shape'])}/{int(results['total'])})"
        )


if __name__ == "__main__":
    main()