from pytorch_lightning.core.lightning import LightningModule
from typing import MutableSequence


class LightningDataMixin(LightningModule):
    def __init__(self):
        super().__init__()

    def _cal_num_classes(self, dataset_name):
        def fn_num(dset):
            return self.dm.get_num_class(dset)

        if isinstance(dataset_name, str):
            return fn_num(dataset_name)
        elif isinstance(dataset_name, MutableSequence):
            return sum(fn_num(dset) for dset in dataset_name)
        else:
            return None

    def prepare_data(self):
        self.dm.prepare_data()

    def train_dataloader(self):
        return self.dm.train_dataloader(pl_trainer=self.trainer,
                                        use_ddp=self.trainer.use_ddp)

    def val_dataloader(self):
        if self.hparams.disable_validation:
            return None
        return self._eval_dataloader(self.hparams.data.val_dataset)

    def test_dataloader(self):
        if self.hparams.data.test_dataset is None:
            return self.val_dataloader()
        else:
            return self._eval_dataloader(self.hparams.data.test_dataset)

    def _eval_dataloader(self, dataset):
        if self.hparams.eval_mode == 'few_shot':
            return self._get_fewshot_loader(dataset)

        elif self.hparams.eval_mode == 'linear':
            return self._get_linear_loader(dataset)

    def _get_fewshot_loader(self, dataset):
        return self.dm.get_fewshot_dataloader(pl_trainer=self.trainer,
                                              use_ddp=self.trainer.use_ddp,
                                              aug=self.hparams.data.val_aug,
                                              datasets=dataset)

    def _get_linear_loader(self, dataset):
        base_loader = self.dm.get_simple_dataloader(
            dataset,
            aug=self.hparams.data.val_aug,
            pl_trainer=self.trainer,
            use_ddp=self.trainer.use_ddp,
            opt=self.hparams.data,
            shuffle=self.hparams.data.shuffle_val)
        return base_loader

    @property
    def batch_size(self) -> int:
        return self.hparams.data.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.hparams.data.batch_size = value