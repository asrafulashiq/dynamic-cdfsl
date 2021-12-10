import torch
import torch
import torch.nn as nn
import abc
import hydra
import utils.utils_system as utils_system
# import backbone
from system.few_shot_mixin import FewShotMixin
from system.linear_mixin import LinearMixin
from system.data_mixin import LightningDataMixin


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(LightningDataMixin, FewShotMixin, LinearMixin):
    """ Abstract class """
    def __init__(self, hparams, datamodule=None):
        super().__init__()
        self.hparams = hparams
        self.dm = datamodule

        if self.hparams.data.num_classes is None and self.dm is not None:
            self.num_classes = self._cal_num_classes(self.hparams.data.dataset)
        else:
            self.num_classes = self.hparams.data.num_classes

    def setup(self, stage):
        if stage == 'fit' and self.trainer.use_ddp:
            num_proc = self.trainer.num_nodes * self.trainer.num_processes
            self.hparams.optimizer.lr *= num_proc

    @abc.abstractmethod
    def forward(self, x):
        return

    def load_base(self, ckpt_path=None, prefix='feature'):
        if ckpt_path is not None:
            ckpt = torch.load(
                ckpt_path,
                map_location=lambda storage, loc: storage)['state_dict']
            new_state = {}
            for k, v in ckpt.items():
                if f'{prefix}.' in k:
                    new_state[k.replace(f'{prefix}.', '')] = v
            self.feature.load_state_dict(new_state,
                                         strict=not self.hparams.load_flexible)

    # --------------------------------- training --------------------------------- #
    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        """ implement your own training step """

    # --------------------------------  validation -------------------------------- #

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if self.hparams.eval_mode == 'linear':
            return self._linear_validation_step(batch, batch_idx, dataset_idx)
        elif self.hparams.eval_mode == 'few_shot':
            return self.few_shot_validation_step(batch, batch_idx, dataset_idx)
        else:
            raise NotImplementedError

    def validation_epoch_end(self, outputs):
        if self.hparams.eval_mode == 'linear':
            return self._linear_validation_epoch_end(outputs)
        elif self.hparams.eval_mode == 'few_shot':
            return self.few_shot_val_end(outputs)
        else:
            raise NotImplementedError

    # ----------------------------------- test ----------------------------------- #
    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args, **kwargs)

    # ---------------------------------- config ---------------------------------- #
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            filter(lambda p: p.requires_grad, self.parameters()))

        if 'scheduler' in self.hparams and self.hparams.scheduler:
            scheduler = hydra.utils.instantiate(self.hparams.scheduler,
                                                optimizer=optimizer)
            return [optimizer], [scheduler]

        return optimizer

    # ----------------------------------- model ---------------------------------- #
    def _create_model(self, num_class):
        """ create encoder and classifier head """
        self.feature = utils_system.build_base_encoder(self.hparams.backbone,
                                                       self.hparams.pretrained,
                                                       self.hparams.model_args)
        if num_class is None:
            self.classifier = None
            return

        # classifier head
        if not self.hparams.linear_bn:
            self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        else:
            # add a batchnorm before classifier
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(self.feature.final_feat_dim,
                               affine=self.hparams.linear_bn_affine),
                nn.Linear(self.feature.final_feat_dim, num_class))

    def get_feature_extractor(self):
        return self.feature
