from typing import List, Tuple

import torch
import gc
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50
from timm.models.vision_transformer import vit_small_patch16_224
from lightly.loss.ntx_ent_loss import InfoNCELoss
from lightly.models.modules import (
    SimCLRProjectionHead,
    MaskedVisionTransformerTIMM,
)
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler
import wandb
from torchvision.utils import make_grid
import itertools
from lightly.utils.dist import print_rank_zero
from transforms_ffcv.dataloaders import get_train_loader


class SimCLR(LightningModule):
    def __init__(
        self,
        batch_size_per_device: int,
        num_classes: int,
        stage_epochs,
        reload_freq,
        dataloaders,
        val_loader,
        scheme,
        backbone: str = "resnet50",
        is_say = False,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.0001
    ) -> None:
        super().__init__()
       
        self.batch_size_per_device = batch_size_per_device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # backbone
        if backbone == "resnet50":
            resnet = resnet50()
            resnet.fc = Identity()  
            self.backbone = resnet
            self.projection_head = SimCLRProjectionHead()
            self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)
        elif backbone == "vit":
            vit = vit_small_patch16_224(dynamic_img_size=True)
            self.backbone = MaskedVisionTransformerTIMM(
                vit=vit
            )  
            self.projection_head = SimCLRProjectionHead(input_dim=384)
            self.online_classifier = OnlineLinearClassifier(
                feature_dim=384, num_classes=num_classes
            )
        self.scheme = scheme
        #print_rank_zero("Using scheme: ", self.scheme)
        #print_rank_zero(stage_epochs)
        self.criterion = InfoNCELoss(gather_distributed=True)
        self.dataloaders = dataloaders  # a list
        self.current_dataloader_index = 0
        self.val_loader = val_loader
        self.clear_cache = False
        self.stage_epochs = stage_epochs
        self.tr_epochs = sum(self.stage_epochs)
        acc_tpt = list(itertools.accumulate(self.stage_epochs)) # stage boundaries
        
        self.vis_stages = [0, 1] + acc_tpt[:-1] + [acc_tpt[-1] - 1] # for visualization
        self.reload = reload_freq 
        self.TEMP_SCHEDULES = {
            5: [0.5, 0.4, 0.3, 0.2, 0.1],
            8: [0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.15, 0.1],
            9: [0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.15, 0.1, 0.1],
        }
        self.temperature = self.TEMP_SCHEDULES.get(len(self.dataloaders), [0.1] * len(self.dataloaders))
        self.steps_per_epoch = len(self.dataloaders[0])
        self.is_say = is_say # for initializing the online train dataloader

    def train_dataloader(self):
        # called before on_train_epoch_start
        print_rank_zero("Using dataloader index: ", self.current_dataloader_index)
        print_rank_zero(
            "Using temperature: ", self.temperature[self.current_dataloader_index]
        )
        return self.dataloaders[self.current_dataloader_index]

    def val_dataloader(self):
        return self.val_loader

    def on_train_epoch_start(self):
        # detect stage boundary and set flag to reload dataloader in the end of the epoch
        if (self.current_epoch + 1) % self.reload == 0:
            stage_end = sum(self.stage_epochs[: (self.current_dataloader_index + 1)])
            if (self.current_epoch + 1) % stage_end == 0:
                self.clear_cache = True

    def on_train_epoch_end(self):

        if self.clear_cache:
            print(f"Epoch {self.current_epoch} finished. Cleaning up CUDA memory...")
            torch.cuda.empty_cache()
            gc.collect()
            self.clear_cache = False
            if self.current_epoch != (self.tr_epochs - 1):
                self.current_dataloader_index += 1
                # call train_dataloader()
                self.trainer.fit_loop._combined_loader = None
                self.trainer.fit_loop.setup_data() 

    def on_save_checkpoint(self, checkpoint):
        checkpoint["current_dataloader_index"] = self.current_dataloader_index

    def on_train_start(self):
        # Create iterator from dataloader for the classifier
        if self.is_say:
            self.clf_loader = get_train_loader(
                64, 8, "/data/say_train_linear.beton"
            )

            self.clf_iter = iter(self.clf_loader)

    #def on_load_checkpoint(self, checkpoint):
    #    self.current_dataloader_index = checkpoint.get("current_dataloader_index", 0)
    #    print_rank_zero(
    #        f"[Resume] Restored dataloader index = {self.current_dataloader_index}"
    #    )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def _unify_ffcv_batch(self, batch):
        """Turn ffcv tuple into a unified dict for the model."""
        FFCV_SCHEMA_PAIR = [
            "imgs0",
            "targets0",
            "instance_ids0",
            "frame_ids0",
            "imgs1",
            "targets1",
            "instance_ids1",
            "frame_ids1",
            "imgs0_aug",
            "imgs1_aug",
        ]

        m = {k: v for k, v in zip(FFCV_SCHEMA_PAIR, batch)}
        views = [m["imgs0"], m["imgs0_aug"], m["imgs1"], m["imgs1_aug"]]
        targets = m["targets0"]

        if self.scheme in ["TDiet","CATDiet","CombDiet"]:
            keys = torch.cat(
                [m["instance_ids0"].repeat(2), m["instance_ids1"].repeat(2)]
            )
        else:
            keys = torch.cat([m["frame_ids0"].repeat(2), m["frame_ids1"].repeat(2)])

        viz = (m["imgs0"], m["imgs1"])
        return {"views": views, "targets": targets, "keys": keys, "viz": viz}

        

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        
        b = self._unify_ffcv_batch(batch)
        views, targets, keys = (
            b["views"],
            b["targets"],
            b["keys"],
        )  # (4B,C,H,W)

        features = self.forward(torch.cat(views)).flatten(start_dim=1)
        z = self.projection_head(features)
        loss = self.criterion(
            z, keys, temperature=self.temperature[self.current_dataloader_index]
        )

        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=len(targets)
        )

        

        # Online Cls:
        if self.is_say:
            try:
                cls_imgs, cls_labels = next(self.clf_iter)
            except StopIteration:
                # restart when the classifier loader is exhausted
                self.clf_iter = iter(self.clf_loader)
                cls_imgs, cls_labels = next(self.clf_iter)
            cls_features = self.forward(cls_imgs).flatten(start_dim=1)
            cls_loss, cls_log = self.online_classifier.training_step(
                (cls_features.detach(), cls_labels), batch_idx
            )
        else:
            cls_loss, cls_log = self.online_classifier.training_step(
                (features.detach(), targets.repeat(len(views))), batch_idx
            )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))
        
        # Visualize some images
        if self.current_epoch in self.vis_stages and batch_idx == 0:
            imgs0, imgs1 = b["viz"]
            show_imgs0 = denormalize(imgs0[:8, :, :, :])
            show_imgs1 = denormalize(imgs1[:8, :, :, :])
            B, C, H, W = show_imgs0.shape
            show_imgs = torch.stack(
                [show_imgs0, show_imgs1], dim=1
            )  # shape: (B, 2, C, H, W)
            show_imgs = show_imgs.reshape(2 * B, C, H, W)
            self.logger.experiment.log(
                {
                    "train_loader_images": [
                        wandb.Image(
                            make_grid(show_imgs.clamp(0, 1), nrow=4),
                            caption="Epoch {}".format(self.current_epoch),
                        )
                    ],
                }
            )

        return loss + cls_loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        features = self.forward(images).flatten(start_dim=1)

        cls_loss, cls_log = self.online_classifier.validation_step(
            (features.detach(), targets), batch_idx
        )
        self.log_dict(cls_log, prog_bar=True, sync_dist=True, batch_size=len(targets))
        return cls_loss


    def configure_optimizers(self):
        total_steps = self.tr_epochs * self.steps_per_epoch
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        param_groups = [
            {"name": "simclr", "params": params},
            {
                "name": "simclr_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            },
            {
                "name": "online_classifier",
                "params": self.online_classifier.parameters(),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=int(
                    self.steps_per_epoch * 10
                ),  # first 10 epochs
                max_epochs=int(total_steps),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


def format_list(tensor):
    return f"[{', '.join(f'{i:.2f}' for i in tensor)}]"


def denormalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # imgs: Tensor of shape (B, C, H, W)
    mean = torch.tensor(mean, device=imgs.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=imgs.device).view(1, -1, 1, 1)
    return imgs * std + mean
