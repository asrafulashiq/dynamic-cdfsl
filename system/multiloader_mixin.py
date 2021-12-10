from typing import Any, List, MutableSequence
from pytorch_lightning.utilities.apply_func import move_data_to_device
import torch
import collections
from torch._six import string_classes


class MultiTrainLoaderMixin(object):
    def train_dataloader(self):
        if self.hparams.apply_unlabel:
            self.build_unlabel_loader()
        return super().train_dataloader()

    def build_unlabel_loader(self):
        datasets = self.hparams.unlabel_params.dataset
        if isinstance(datasets, str) and ',' in datasets:
            datasets = datasets.split(',')
        if not isinstance(datasets, MutableSequence):
            datasets = [datasets]
        _loaders = []
        for dset in datasets:

            limit_data = None
            if hasattr(self.hparams.unlabel_params, 'limit_data'):
                limit_data = self.hparams.unlabel_params.limit_data

            _dloader = self.dm.get_unlabel_dataloader(
                dataset_name=dset,
                batch_size=self.hparams.unlabel_params.batch_size,
                labels_to_chose=None,  # all labels
                labels_dict=None,  # true label
                aug=self.hparams.unlabel_params.unlabel_aug,
                limit_data=limit_data,
                unlabel_kwrgs={"filter_data": False})

            if self.trainer.use_ddp:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    _dloader.dataset, shuffle=True)
                _dloader = self.trainer.replace_sampler(_dloader, sampler)

            self.logger.experiment.info(
                f"unlabel loader size: {len(_dloader)}")

            _loaders.append(_dloader)

        self.num_unlabel = len(_loaders)
        self.unlabel_dataloaders = _loaders

    def get_unlabel_batch(self) -> List[Any]:
        def get_batch(i=0):
            try:
                batch = next(self.unlabel_dataloader_iters[i])
            except StopIteration:
                self.unlabel_dataloader_iters[i] = iter(
                    self.unlabel_dataloaders[i])
                batch = next(self.unlabel_dataloader_iters[i])
            batch = move_data_to_device(batch, self.device)
            return batch

        return custom_collate([get_batch(i) for i in range(self.num_unlabel)])

    def on_train_epoch_start(self) -> None:
        # replace seed for ddp to unlabel dataloader
        if self.hparams.apply_unlabel:
            try:
                for loader in self.unlabel_dataloaders:
                    loader.sampler.set_epoch(self.current_epoch)
            except Exception:
                pass
            self.unlabel_dataloader_iters = [
                iter(loader) for loader in self.unlabel_dataloaders
            ]

        return super().on_train_epoch_start()


def custom_collate(batch):
    """ modified from pytorch default collate """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(
        "custom_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}".format(elem_type))
