from pytorch_lightning.metrics.functional import accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_system import concat_all_ddp, aggregate_dict


class LinearMixin():
    def _forward_batch(self, x):
        return self.forward(x)

    def _linear_validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        mlp_preds, *_ = self._forward_batch(x)
        mlp_loss = torch.nn.functional.cross_entropy(mlp_preds, y)
        self.log("linear_loss", mlp_loss, on_epoch=True)
        # result.loss = mlp_loss
        prob = mlp_preds.softmax(dim=-1)
        gt = y
        return {"prob": prob, "gt": gt}

    def _linear_validation_epoch_end(self, outputs, *args, **kwargs):
        outputs = aggregate_dict(outputs)
        epoch_probs = concat_all_ddp(outputs["prob"])
        epoch_gt = concat_all_ddp(outputs["gt"])
        epoch_preds = torch.argmax(epoch_probs, dim=-1)

        mean_accuracy = accuracy(epoch_preds, epoch_gt)

        self.log("acc_mean", mean_accuracy)
