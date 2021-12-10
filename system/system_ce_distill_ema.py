import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
from system import system_distill_abstract
import copy
from typing import Any
import torch.distributed as dist


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_distill_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)

        if self.hparams.apply_center:
            self.register_buffer("center", torch.zeros(1, self.num_classes))

    def create_teacher(self):
        teacher_modules = []
        teacher_modules.append(copy.deepcopy(self.feature))
        teacher_modules.append(copy.deepcopy((self.get_header())))
        self.teacher = nn.Sequential(*teacher_modules)
        self.teacher.requires_grad_(False)

    # --------------------------------- training --------------------------------- #
    def _forward_loss(self, batch, batch_u):
        x, y = batch

        (x_u_w, x_u_s), *_ = batch_u

        scores, scores_u = self.set_forward(x, x_u_s)
        # ce loss
        loss_ce = self.ce_loss(scores, y)
        top1 = self.train_acc(scores.argmax(dim=-1), y)

        # pseudo loss
        if self.hparams.unlabel_params.no_stop_grad is False:
            torch.set_grad_enabled(False)

        logit_pseudo = self.teacher(x_u_w)
        if self.hparams.apply_center:
            logit_pseudo = logit_pseudo - self.center
            self.update_center(logit_pseudo.clone().detach())
        logit_pseudo = logit_pseudo.detach()

        if self.hparams.unlabel_params.no_stop_grad is False:
            torch.set_grad_enabled(True)

        loss_pseudo = self.distill_loss(scores_u, logit_pseudo)

        return loss_ce, loss_pseudo, top1

    def on_train_batch_start(self, batch: Any, batch_idx: int,
                             dataloader_idx: int) -> None:
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)

        # EMA update for the teacher
        with torch.no_grad():
            m = self.hparams.mometum_update  # momentum parameter
            for param_q, param_k in zip(self.student.parameters(),
                                        self.teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def on_epoch_start(self) -> None:
        super().on_epoch_start()
        self.teacher.requires_grad_(False)

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))
        # ema update
        self.center = self.center * self.hparams.center_momentum + batch_center * (
            1 - self.hparams.center_momentum)

    def get_feature_extractor(self):
        if self.hparams.extractor == "teacher":
            return self.teacher[0]
        else:
            return self.feature