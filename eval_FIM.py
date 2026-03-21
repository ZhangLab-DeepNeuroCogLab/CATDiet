
import os
import re
import json
import matplotlib.pyplot as plt
from models_collection import (
    dino,
    simclr, 
)
import torch
from lightly.utils.dist import print_rank_zero
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
from transforms_ffcv.dataloaders import ( 
    get_val_loader,
)
import hydra
from omegaconf import DictConfig
from collections import defaultdict
import itertools
import bisect

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



def fim_empirical_batch(model, batch, device="cuda"):
    """
    Compute empirical Fisher trace for one batch.
    Args:
        model: SimCLR LightningModule
        batch: raw batch from dataloader
        tau: temperature for InfoNCE
    """
    model = model.float()
    b = model._unify_ffcv_batch(batch)  
    views, keys = b["views"], b["keys"]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        # Forward
        x = torch.cat(views).to(device)  # (num_views*B, C, H, W)
        features = model.forward(x).flatten(start_dim=1)
        z = model.projection_head(features)
        loss = model.criterion(
            z, keys, temperature=model.temperature[model.current_dataloader_index]
        )

        grads = torch.autograd.grad(
            loss,
            model.backbone.parameters(),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Map param names to grads
        name_to_grad = {
            name: g for (name, p), g in zip(model.backbone.named_parameters(), grads)
        }

        results = {}
        for name, g in name_to_grad.items():
            if (
                g is not None and  name == "conv1.weight"
            ):  # only first conv of each block
                results[name] = g.flatten().pow(2).sum().item()
    return results


def fim_empirical_ckpt(
    cfg,
    model_class,
    train_dataloader,
    ckpt_path,
    device="cuda:0",
    max_batches=5,
):
    """
    Compute average empirical Fisher trace for a checkpoint.
    """
    ckpt = torch.load(ckpt_path)
    dummy_loader = DummyLoader()
    model = model_class(
        batch_size_per_device=cfg.batch_size_per_device,
        num_classes=cfg.num_classes,
        scheme=cfg.scheme_id,
        stage_epochs=cfg.stage_epochs,
        reload_freq=cfg.reload_freq,
        dataloaders=[dummy_loader],
        val_loader=dummy_loader,
        backbone=cfg.backbone,
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    agg = defaultdict(list)
    
    for i, batch in enumerate(train_dataloader):
        if i >= max_batches:
            break
        val_dict = fim_empirical_batch(model, batch, device=device)  # returns dict
        for k, v in val_dict.items():
            agg[k].append(v)

    # average per layer
    avg_dict = {k: sum(vs) / len(vs) for k, vs in agg.items()}
    return avg_dict



def fim_multi_ckpt(cfg, ckpt_paths, device="cuda:0"):
    # Collect results across checkpoints
    all_results = {}  # key -> list of values
    steps = []
    method_cfg = METHODS[cfg.methods]
    dataloader_fns = method_cfg.get("dataloaders")
    train_dataloader = dataloader_fns.get(cfg.diet_name)(cfg)
    
    model_class = method_cfg.get("model")
    for path in ckpt_paths:
        step = get_step_num(path)
        dataloader_idx = bisect.bisect_right(list(itertools.accumulate(cfg.stage_epochs)), get_epoch_num(path))
        fim_dict = fim_empirical_ckpt(
            cfg,
            model_class,
            train_dataloader[dataloader_idx],
            path,
            device,
            max_batches=5,
        )
        steps.append(step)

        # store per-layer values
        for k, v in fim_dict.items():
            all_results.setdefault(k, []).append(v)

        print(f"Step {step}:")
        for k, v in fim_dict.items():
            print(f"  {k}: {v:.4f}")
    safe_results = {k: [float(x) for x in v] for k, v in all_results.items()}
    with open("fim_epoch.json", "w") as f:
        json.dump(safe_results, f, indent=4)

class DummyLoader:
    def __len__(self) -> int:
        return 1

def get_step_num(fname):
    m = re.search(r"step=(\d+)", fname)
    return int(m.group(1)) if m else -1


def get_epoch_num(fname):
    m = re.search(r"epoch=(\d+)", fname)
    return int(m.group(1)) if m else -1


@hydra.main(config_path="configs", config_name="eval_fim")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    print_rank_zero(f"Working directory : {os.getcwd()}")
    ckpt_paths = [os.path.join(cfg.ckpt_dir, f) for f in os.listdir(cfg.ckpt_dir) if f != "last.ckpt"]
    
    # sort by step number
    ckpt_paths = sorted(ckpt_paths, key=get_step_num)
    fim_multi_ckpt(cfg, ckpt_paths)


if __name__ == "__main__":
    
    main()
