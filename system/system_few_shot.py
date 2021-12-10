from system import system_abstract
import torch
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

    def load_base(self, ckpt_path=None):
        if ckpt_path is not None:
            ckpt = torch.load(
                ckpt_path,
                map_location=lambda storage, loc: storage)['state_dict']
            new_state = {}
            for k, v in ckpt.items():
                if 'feature.' in k:
                    new_state[k.replace('feature.', '')] = v
            self.feature.load_state_dict(new_state,
                                         strict=not self.hparams.load_flexible)

    def forward(self, x):
        out = self.feature.forward(x)
        return out

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_feature_extractor(self):
        """ return feature extractor """
        return self.feature

    def train_dataloader(self):
        return None
