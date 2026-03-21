from typing import Dict, Tuple

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear
import torch
from lightly.utils.dist import print_rank_zero
from lightly.utils.benchmarking.topk import mean_topk_accuracy


class OnlineLinearClassifier(LightningModule):
    def __init__(
        self,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.classification_head(x.detach().flatten(start_dim=1))

    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        features, targets = batch[0], batch[1]
        # debug:
        # if batch_idx == 0:
        #    f0_norm = features.norm(dim=1)
        #    f0_std = f0_norm.std()
        #    print_rank_zero(f"[debug]: f0_norm is {f0_norm[[0,1,2,-3,-2,-1]]}")
        #    print_rank_zero(f"[debug]: f0 std is {f0_std}")

        predictions = self.forward(features)

        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            if torch.isnan(predictions).any():
                print_rank_zero("⚠️ Logits contain NaN!")
            if torch.isinf(predictions).any():
                print_rank_zero("⚠️ Logits contain Inf!")
            # print_rank_zero("⚠️ Logits contain NaN or Inf!")
            # exit(1)
        loss = self.criterion(predictions, targets)
        _, predicted_classes = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_classes, targets, k=self.topk)
        return loss, topk

    def training_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"train_online_cls_loss": loss}
        log_dict.update({f"train_online_cls_top{k}": acc for k, acc in topk.items()})
        return loss, log_dict

    def validation_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        log_dict = {"val_online_cls_loss": loss}
        log_dict.update({f"val_online_cls_top{k}": acc for k, acc in topk.items()})
        return loss, log_dict
