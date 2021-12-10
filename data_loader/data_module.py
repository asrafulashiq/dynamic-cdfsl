import torch
from utils.utils_system import get_rank
import pytorch_lightning as pl
from data_loader import utils_data
import os
from pytorch_lightning.utilities.distributed import rank_zero_only
from typing import MutableSequence
from omegaconf import OmegaConf
from loguru import logger
from data_loader.utils_data import get_split


class DataModule(object):
    def __init__(self, hparams, conf_path=None) -> None:
        self.hparams = hparams
        if conf_path is None:
            conf_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config_path.yaml")
        self.conf_data = OmegaConf.load(conf_path)
        for key in self.conf_data:
            self.conf_data[key]['data_path'] = os.path.expanduser(
                self.conf_data[key]['data_path'])

        self.count = 0

    def get_path(self, dname):
        # check if there is any split
        basename = dname.split("_")[0]
        return self.conf_data[basename]['data_path']

    def get_num_class(self, dname):
        # check if there is any split
        basename = dname.split("_")[0]
        return self.conf_data[basename]['num_class']

    def prepare_data(self, *args, **kwargs):
        # prepare indices
        def fn_prepare(_dataset, split_type=None):
            if isinstance(_dataset, MutableSequence) or isinstance(
                    _dataset, tuple):
                for dname in _dataset:
                    suff = get_split(dname, split_type)[1]
                    if suff:
                        utils_data.prepare_data_indices(dname,
                                                        self.get_path(dname),
                                                        split_type,
                                                        opt=self.hparams)
            elif isinstance(_dataset, str):
                suff = get_split(_dataset, split_type)[1]
                if suff:
                    utils_data.prepare_data_indices(_dataset,
                                                    self.get_path(_dataset),
                                                    split_type,
                                                    opt=self.hparams)
            else:
                pass

        if self.hparams.dataset:
            fn_prepare(self.hparams.dataset, self.hparams.train_split_type)

        if not self.hparams.disable_validation and self.hparams.val_dataset:
            fn_prepare(self.hparams.val_dataset, self.hparams.val_split_type)

    def train_dataloader(self,
                         pl_trainer=None,
                         use_ddp=False,
                         *args,
                         **kwargs):
        # if self.hparams.dataset is None:
        #     self.hparams.dataset = self.hparams.val_dataset
        datamgr = utils_data.SimpleDataManager(
            self.hparams.image_size,
            batch_size=self.hparams.batch_size,
            dataset_name=self.hparams.dataset,
            split_type=self.hparams.train_split_type,
            opt=self.hparams)

        if isinstance(self.hparams.dataset, MutableSequence) or isinstance(
                self.hparams.dataset, tuple):
            data_path = [self.get_path(_dat) for _dat in self.hparams.dataset]
        else:
            data_path = self.get_path(self.hparams.dataset)
        base_loader = datamgr.get_data_loader(
            data_path,
            aug=self.hparams.train_aug,
            consecutive_label=True,
            limit_data=self.hparams.limit_train_data)

        if use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(
                base_loader.dataset, shuffle=True)
            base_loader = pl_trainer.replace_sampler(base_loader, sampler)

        self._log(f"train loader size: {len(base_loader)}")
        return base_loader

    @rank_zero_only
    def _log(self, msg):
        logger.info(msg)

    def get_unlabel_dataloader(self,
                               dataset_name=None,
                               labels_to_chose=None,
                               labels_dict=None,
                               num_classes=None,
                               batch_size=None,
                               unlabel=True,
                               pl_trainer=None,
                               aug=None,
                               limit_data=None,
                               unlabel_kwrgs={}):
        self.count += 1
        if dataset_name is None:
            dataset_name = self.hparams.dataset

        if labels_to_chose is None:
            if hasattr(self, 'labels_to_chose'):
                labels_to_chose = self.labels_to_chose
        else:
            self.labels_to_chose = labels_to_chose

        datamgr = utils_data.SimpleDataManager(
            self.hparams.image_size,
            batch_size=self.hparams.batch_size
            if batch_size is None else batch_size,
            dataset_name=dataset_name,
            split_type=self.hparams.train_split_type,
            unlabel=unlabel,
            opt=self.hparams,
            seed=self.hparams.seed + self.count + get_rank())

        if isinstance(dataset_name, MutableSequence) or isinstance(
                dataset_name, tuple):
            data_path = [self.get_path(_dat) for _dat in dataset_name]
        else:
            data_path = self.get_path(dataset_name)
        base_loader, unlabel_dataset = datamgr.get_unlabel_loader(
            data_path,
            labels_to_chose=labels_to_chose,
            num_classes=num_classes,
            labels_dict=labels_dict,
            aug=self.hparams.train_aug if aug is None else aug,
            consecutive_label=True,
            drop_last=True,
            limit_data=limit_data,
            **unlabel_kwrgs)

        # store
        self.mgr = datamgr
        self.udset = unlabel_dataset

        self._log(f"unlabel batch: {len(base_loader)}")

        return base_loader

    def filter_unlabel_dataloader(self,
                                  unlabel_idx=None,
                                  unlabel_y=None,
                                  aug='true'):

        return self.mgr.filter_unlabel_dataloader(self.udset,
                                                  unlabel_idx,
                                                  unlabel_y,
                                                  aug=aug)

    def get_fewshot_dataloader(self,
                               pl_trainer=None,
                               use_ddp=False,
                               aug=None,
                               datasets=None,
                               *args,
                               **kwargs):
        # datasets = self.hparams.val_dataset
        if datasets is None or datasets == 'none':
            return None

        if not isinstance(datasets, MutableSequence):
            datasets = [datasets]

        list_loader = []
        for dset in datasets:
            few_shot_params = dict(n_way=self.hparams.n_way,
                                   n_shot=self.hparams.n_shot,
                                   n_query=self.hparams.n_query,
                                   num_episodes=self.hparams.num_episodes,
                                   aug=aug if aug else self.hparams.val_aug)
            if use_ddp:
                ddp_args = dict(num_replicas=pl_trainer.num_nodes *
                                pl_trainer.num_processes,
                                rank=pl_trainer.global_rank)
            else:
                ddp_args = {}
            if 'few_shot_params' in kwargs:
                few_shot_params.update(kwargs['few_shot_params'])
            datamgr = utils_data.SetDataManager(
                self.get_path(dset),
                self.hparams.image_size,
                dataset_name=dset,
                opt=self.hparams,
                split_type=self.hparams.val_split_type,
                **few_shot_params)
            novel_loader = datamgr.get_data_loader(use_ddp=use_ddp,
                                                   dist_args=ddp_args)
            list_loader.append(novel_loader)

        return list_loader

    def get_simple_dataloader(self,
                              dataset_name,
                              aug='true',
                              pl_trainer=None,
                              use_ddp=False,
                              opt=None,
                              drop_last=True,
                              shuffle=True):
        datamgr = utils_data.SimpleDataManager(
            self.hparams.image_size,
            batch_size=self.hparams.batch_size,
            dataset_name=dataset_name,
            split_type=self.hparams.train_split_type,
            opt=self.hparams)
        if isinstance(dataset_name, MutableSequence):
            data_path = [self.get_path(_dat) for _dat in dataset_name]
        else:
            data_path = self.get_path(dataset_name)

        base_loader = datamgr.get_data_loader(data_path,
                                              aug=aug,
                                              consecutive_label=True,
                                              drop_last=drop_last,
                                              shuffle=shuffle)
        if use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(
                base_loader.dataset, shuffle=True)
            base_loader = pl_trainer.replace_sampler(base_loader, sampler)

        return base_loader
