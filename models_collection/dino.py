import copy
from typing import List, Tuple

import torch
import gc
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Identity
from torchvision.models import resnet50
from timm.models.vision_transformer import vit_small_patch16_224
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import (
    activate_requires_grad,
    deactivate_requires_grad,
    get_weight_decay_parameters,
    update_momentum,
)
from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
import wandb
from torchvision.utils import make_grid
import itertools
from lightly.utils.dist import print_rank_zero
from transforms_ffcv.dataloaders import get_train_loader


class DINO(LightningModule):
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
        is_say=False,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.0001
    ) -> None:
        super().__init__()
        
        self.batch_size_per_device = batch_size_per_device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if backbone == "resnet50":
            resnet = resnet50()
            resnet.fc = Identity()  # Ignore classification head
            self.backbone = resnet
            self.projection_head = DINOProjectionHead()
            self.student_backbone = copy.deepcopy(self.backbone)
            self.student_projection_head = DINOProjectionHead(freeze_last_layer=1)
            self.online_classifier = OnlineLinearClassifier(num_classes=num_classes)
        elif backbone == "vit":
            vit = vit_small_patch16_224(dynamic_img_size=True)
            self.backbone = MaskedVisionTransformerTIMM(vit=vit)
            self.projection_head = DINOProjectionHead(
                input_dim=384, norm_last_layer=False
            )

            vit_student = vit_small_patch16_224(
                dynamic_img_size=True, drop_path_rate=0.1
            )
            self.student_backbone = MaskedVisionTransformerTIMM(vit=vit_student)
            self.student_projection_head = DINOProjectionHead(
                input_dim=384, freeze_last_layer=1, norm_last_layer=False
            )

            self.online_classifier = OnlineLinearClassifier(
                feature_dim=384, num_classes=num_classes
            )
       
        self.scheme = scheme
        self.criterion = DINOLoss()
        self.dataloaders = dataloaders
        self.current_dataloader_index = 0
        self.val_loader = val_loader
        self.clear_cache = False
        self.stage_epochs = stage_epochs
        self.tr_epochs = sum(self.stage_epochs)
        
        acc_tpt = list(itertools.accumulate(self.stage_epochs))
        self.cum_stages = [0, 1] + acc_tpt[:-1] + [acc_tpt[-1] - 1] # for visualization
        self.reload = reload_freq  

        self.teacher_t = [0.04] * len(self.dataloaders)#[0.10, 0.10, 0.08, 0.06, 0.06, 0.05, 0.045, 0.04, 0.04]
        self.student_t = [0.1] * len(self.dataloaders)#[0.20, 0.20, 0.15, 0.12, 0.12, 0.10, 0.10, 0.10, 0.10]
        
        self.milestone_epoch = acc_tpt[-2] # Phase 1 duration
        self.steps_per_epoch = len(self.dataloaders[0])
        self.total_steps_epoch_phase1 = int(self.milestone_epoch * self.steps_per_epoch)
        self.total_steps_epoch_phase2 = int(
            (self.tr_epochs - self.milestone_epoch) * self.steps_per_epoch
        )
        self.is_say=is_say
        print_rank_zero(self.stage_epochs)

    def train_dataloader(self):
        # called before on_train_epoch_start
        print_rank_zero("Using dataloader index: ", self.current_dataloader_index)
        print_rank_zero(
            f"Using teacher_t: {self.teacher_t[self.current_dataloader_index]}, student_t: {self.student_t[self.current_dataloader_index]}"
        )
        return self.dataloaders[self.current_dataloader_index]

    def val_dataloader(self):

        return self.val_loader

    def on_train_epoch_start(self):
        # check stage boundary and set flag to reload dataloader in the end of the epoch
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

    def on_load_checkpoint(self, checkpoint):
        self.current_dataloader_index = checkpoint.get("current_dataloader_index", 0)
        print_rank_zero(
            f"[Resume] Restored dataloader index = {self.current_dataloader_index}"
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def forward_student(self, x: Tensor) -> Tensor:
        features = self.student_backbone(x).flatten(start_dim=1)
        projections = self.student_projection_head(features)
        return projections

    def on_train_start(self) -> None:
        deactivate_requires_grad(self.backbone)
        deactivate_requires_grad(self.projection_head)
        if self.is_say:
            self.clf_loader = get_train_loader(
                64, 8, "/data/say_train_linear.beton"
            )

            self.clf_iter = iter(self.clf_loader)

    def on_train_end(self) -> None:
        activate_requires_grad(self.backbone)
        activate_requires_grad(self.projection_head)

    def _unify_ffcv_batch(self, batch):
        """Turn ffcv tuple into a unified dict for the model."""
        FFCV_SCHEMA_PAIR = [
            "image0",
            "label0",
            "instance_id0",
            "frame_id0",
            "image1",
            "label1",
            "instance_id1",
            "frame_id1",
            "image0_g2",
            "image0_l1",
            "image0_l2",
            "image0_l3",
            "image0_l4",
            "image0_l5",
            "image0_l6",
            "image1_g2",
            "image1_l1",
            "image1_l2",
            "image1_l3",
            "image1_l4",
            "image1_l5",
            "image1_l6",
        ]
        
        

        m = {k: v for k, v in zip(FFCV_SCHEMA_PAIR, batch)}
        global_views = torch.cat(
            [m["image0"], m["image1"], m["image0_g2"], m["image1_g2"]]
        )
        local_views = torch.cat(
            [
                m["image0_l1"],
                m["image1_l1"],
                m["image0_l2"],
                m["image1_l2"],
                m["image0_l3"],
                m["image1_l3"],
                m["image0_l4"],
                m["image1_l4"],
                m["image0_l5"],
                m["image1_l5"],
                m["image0_l6"],
                m["image1_l6"],
            ]
        )
        if self.scheme in ["TDiet","CATDiet","CombDiet"]:
            chunk_teacher = 4
            chunk_student = 16
            return {
                "global_views": global_views,
                "local_views": local_views,
                "targets": torch.cat([m["label0"], m["label1"]]),
                "chunk_teacher": chunk_teacher,
                "chunk_student": chunk_student,
            }
        else:
            chunk_teacher = 2
            chunk_student = 8
            return {
                "global_views": global_views,
                "local_views": local_views,
                "targets": torch.cat([m["label0"], m["label1"]]),
                "chunk_teacher": chunk_teacher,
                "chunk_student": chunk_student,
            }

        

    def training_step(
        self, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ) -> Tensor:
        # Momentum update teacher.
        if self.trainer.global_step < self.total_steps_epoch_phase1:
            momentum = cosine_schedule(
                step=self.trainer.global_step,
                max_steps=self.total_steps_epoch_phase1,
                start_value=0.996,
                end_value=1.0,
            )
        else:
            momentum = cosine_schedule(
                step=(self.trainer.global_step - self.total_steps_epoch_phase1),
                max_steps=self.total_steps_epoch_phase2, 
                start_value=0.996,
                end_value=1.0,
            )
        update_momentum(self.student_backbone, self.backbone, m=momentum)
        update_momentum(self.student_projection_head, self.projection_head, m=momentum)

        b = self._unify_ffcv_batch(batch)
        global_views, local_views, targets, chunk_teacher, chunk_student = (
            b["global_views"],
            b["local_views"],
            b["targets"],
            b["chunk_teacher"],
            b["chunk_student"],
        )

        teacher_features = self.forward(global_views).flatten(start_dim=1)
        teacher_projections = self.projection_head(teacher_features)
        student_projections = torch.cat(
            [self.forward_student(global_views), self.forward_student(local_views)]
        )

        loss = self.criterion(
            teacher_out=teacher_projections.chunk(chunk_teacher),
            student_out=student_projections.chunk(chunk_student),
            teacher_temp=self.teacher_t[self.current_dataloader_index],
            student_temp=self.student_t[self.current_dataloader_index],
        )
        self.log_dict(
            {"train_loss": loss, "ema_momentum": momentum},
            prog_bar=True,
            sync_dist=True,
            batch_size=len(targets),
        )

        # Online classification.
        if self.is_say:
            try:
                clf_batch = next(self.clf_iter)
            except StopIteration:
                # restart when the classifier loader is exhausted
                self.clf_iter = iter(self.clf_loader)
                clf_batch = next(self.clf_iter)
            cls_imgs, cls_labels = clf_batch
            cls_features = self.forward(cls_imgs).flatten(start_dim=1)
            cls_loss, cls_log = self.online_classifier.training_step(
                (cls_features.detach(), cls_labels), batch_idx
            )
        else:
            cls_loss, cls_log = self.online_classifier.training_step(
                (teacher_features.chunk(2)[0].detach(), targets), batch_idx
            )
        self.log_dict(cls_log, sync_dist=True, batch_size=len(targets))

        if self.current_epoch in self.cum_stages and batch_idx == 0:
            n_show = 8
            globals_ = [
                denormalize(v[:n_show]) for v in global_views.chunk(chunk_teacher)[:2]
            ]
            # locals_ = [denormalize(v[:n_show]) for v in views[2:]]

            g = torch.stack(globals_, dim=1).clamp(0, 1).cpu()  # (n_show,2, C, 224,224)
            # l = torch.stack(locals_, dim=1).clamp(0, 1).cpu()  # (n_show,6, C, 96,96)
            g = g.reshape(2 * n_show, 3, 224, 224)
            # l = l.reshape(6 * n_show, 3, 96, 96)

            grid_g = make_grid(g, nrow=4)
            # grid_l = make_grid(l, nrow=6)

            self.logger.experiment.log(
                {
                    "train_loader_globals": [
                        wandb.Image(
                            grid_g, caption=f"Epoch {self.current_epoch} globals"
                        )
                    ],
                    # "train_loader_locals": [
                    #    wandb.Image(
                    #        grid_l, caption=f"Epoch {self.current_epoch} locals"
                    #    )
                    # ],
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
            [self.student_backbone, self.student_projection_head]
        )

        param_groups = [
            {"name": "dino", "params": params},
            {
                "name": "dino_no_weight_decay",
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
                warmup_epochs=int(self.steps_per_epoch * 10 ),
                max_epochs=int(total_steps),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    

def denormalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # imgs: Tensor of shape (B, C, H, W)
    mean = torch.tensor(mean, device=imgs.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=imgs.device).view(1, -1, 1, 1)
    return imgs * std + mean
