import torch
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F

from system import system_abstract
from typing import MutableSequence


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)

        if isinstance(self.hparams.data.dataset, MutableSequence) and len(
                self.hparams.data.dataset) == 1:
            self.hparams.data.dataset = self.hparams.data.dataset[0]
        self._create_model(self.num_classes)

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores, out

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, MutableSequence) or isinstance(x, tuple):
            # MoCo augmentation
            x = torch.cat(x, dim=0)
            y = torch.cat((y, y), dim=0)

        scores, _ = self(x)
        _, predicted = torch.max(scores.data, 1)
        loss = torch.nn.functional.cross_entropy(scores, y)

        with torch.no_grad():
            accur = accuracy(predicted.detach(), y)

        tqdm_dict = {"loss_train": loss, "top1": accur}
        self.log_dict(tqdm_dict,
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True,
                      logger=True)
        return loss

    def get_feature_extractor(self):
        """ return feature extractor """
        return self.feature
