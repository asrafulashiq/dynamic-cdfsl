from typing import Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import os
import shutil
import math
from pathlib import Path


class CheckpointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self,
                 dirpath: Optional[Union[str, Path]] = None,
                 filename: Optional[str] = None,
                 monitor: Optional[str] = None,
                 verbose: bool = False,
                 save_last: Optional[bool] = None,
                 save_top_k: Optional[int] = None,
                 save_weights_only: bool = False,
                 mode: str = "auto",
                 period: int = 1,
                 prefix: str = "",
                 **kwargs):
        super().__init__(dirpath=dirpath,
                         filename=filename,
                         monitor=monitor,
                         verbose=verbose,
                         save_last=save_last,
                         save_top_k=save_top_k,
                         save_weights_only=save_weights_only,
                         mode=mode,
                         period=period,
                         prefix=prefix)

        self.save_every_n_epoch = kwargs['save_every_n_epoch']

    def on_validation_end(self, trainer, pl_module):
        try:
            self.save_checkpoint(trainer, pl_module)
        except pl.utilities.exceptions.MisconfigurationException:
            return

    @rank_zero_only
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch % self.period == 0 and pl_module.hparams.test is False:
            last_filepath = os.path.join(self.dirpath, "last.ckpt")
            self._save_model(last_filepath, trainer, pl_module)

        if (self.save_every_n_epoch is not None
                and trainer.current_epoch % self.save_every_n_epoch == 0):
            _filepath = os.path.join(self.dirpath,
                                     f"epoch_{trainer.current_epoch}.ckpt")
            self._save_model(_filepath, trainer, pl_module)

    @rank_zero_only
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # save best checkpoint as 'best.ckpt'
        if hasattr(self, 'best_model_path') and os.path.exists(
                self.best_model_path):
            best_path = os.path.join(self.dirpath, "best.ckpt")
            shutil.copy(self.best_model_path, best_path, follow_symlinks=True)


class LRScheduler(pl.Callback):
    def __init__(self,
                 initial_lr=0.03,
                 use_cosine_scheduler=False,
                 schedule=None,
                 max_epochs=50):
        super().__init__()
        self.lr = initial_lr
        self.use_cosine_scheduler = use_cosine_scheduler
        if schedule:
            self.schedule = schedule
        else:
            self.schedule = (int(max_epochs * 1 / 3), int(max_epochs * 2 / 3))
        self.max_epochs = max_epochs

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        lr = self.lr

        if self.use_cosine_scheduler:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.max_epochs))
        else:  # stepwise lr schedule
            for milestone in self.schedule:
                lr *= 0.1 if epoch >= milestone else 1.

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr