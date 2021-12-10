from collections import defaultdict
from typing import List, Optional
import torch
import numpy as np
from abc import abstractmethod
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from tqdm import tqdm
from itertools import chain
from loguru import logger
from data_loader import additional_data, helper
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchvision
import os
from data_loader.additional_data import ConcatProportionDataset
from data_loader.transforms import TransformLoader
from data_loader.helper import EpisodicBatchSampler, DistEpisodicBatchSampler
from utils.utils_system import get_rank

ImageFile.LOAD_TRUNCATED_IMAGES = True


def identity(x):
    return x


# NOTE: change path if you have different root
TMP_PATH = os.path.expanduser("data/cache_cdfsl")
DATA_ROOT = os.path.expanduser("data/cdfsl")

# NOTE as we are dealing with few-shot, no dataset has splits
dataset_no_split = [
    "ISIC",
    "ChestX",
    "EuroSAT",
    "CropDisease",  # Exception: crop-disease
    "miniImageNettest",
    "tieredImageNettest"
]


def get_split(dname, split_type=None):
    splits = dname.split("_")
    if len(splits) > 1:
        base, mode = splits[0], splits[-1]
    else:
        base, mode = splits[0], None
    # These datasets have no train/test split, manually create them
    # SUN397, ISIC, ChestX, EuroSAT, Omniglot, sketch, DeepWeeds, Resisc45
    data_indices_suffix = ""
    if split_type is not None:
        data_indices_suffix = "_" + split_type
    if any(x in dname for x in dataset_no_split):
        # mode = None
        if mode is not None and not data_indices_suffix:
            data_indices_suffix = "_partial"

    return base, data_indices_suffix, mode


def get_image_folder(dataset_name, data_path, split_type=None):

    base_dataset_name, data_indices_suffix, mode = get_split(
        dataset_name, split_type)

    def get_data(mode):
        return additional_data.__dict__[base_dataset_name](data_path,
                                                           mode=mode)

    if base_dataset_name in additional_data.__dict__.keys():
        if split_type is None:
            if mode is None:
                if base_dataset_name in dataset_no_split:
                    dset = get_data(mode)
                else:
                    dset = torch.utils.data.ConcatDataset(
                        (get_data("train"), get_data("test")))
            else:
                dset = get_data(mode)
        elif base_dataset_name in dataset_no_split:
            dset = get_data(mode)
        else:
            dset = torch.utils.data.ConcatDataset(
                (get_data("train"), get_data("test")))
    else:
        dset = ImageFolder(data_path)

    return dset, base_dataset_name, data_indices_suffix, mode


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 split_type=None,
                 transform=None,
                 target_transform=identity,
                 dataset_name=None,
                 consecutive_label=False,
                 opt=None,
                 raise_error=True,
                 unlabel_training=False):
        self.transform = transform
        self.unlabel_training = unlabel_training
        self.target_transform = target_transform
        self.rng = np.random.RandomState(seed=opt.seed)
        self.raise_error = raise_error

        self.data, base_dataset_name, data_indices_suffix, mode = get_image_folder(
            dataset_name, data_path, split_type)
        # loguru_log(str(get_image_folder.cache_info()))

        if split_type is not None:
            data_indices_suffix = "_" + split_type

        self.cls_to_idx = None
        if 'partial' in data_indices_suffix or 'disjoint' in data_indices_suffix or 'overlap' in data_indices_suffix:
            tmpfile = os.path.join(
                TMP_PATH, base_dataset_name +
                f"_indices{data_indices_suffix}_{mode}_{opt.split_fraction}_{opt.split_seed}.npy"
            )
            if not os.path.exists(tmpfile):
                prepare_data_indices(dataset_name,
                                     data_path,
                                     split_type,
                                     opt=opt)

            self.class_indices = np.load(tmpfile, allow_pickle=True).item()
            list_classes = list(sorted(self.class_indices.keys()))
            self.indices = list(
                chain.from_iterable(self.class_indices.values()))

            if consecutive_label:
                self.cls_to_idx = {
                    c: i
                    for i, c in enumerate(sorted(list_classes))
                }
            loguru_log(f"loading indices from {tmpfile}")
        else:
            self.indices = None

    def __getitem__(self, i):
        idx = i
        if self.indices is not None:
            idx = self.indices[i]

        try:
            img, target = self.data[idx]
        except FileNotFoundError:
            if self.raise_error:
                raise FileNotFoundError

            rand_idx = int(self.rng.choice(len(self.data)))
            img, target = self.data[rand_idx]

        if self.cls_to_idx is not None:
            target = self.cls_to_idx[target]

        img = self.transform(img)
        target = self.target_transform(target)

        if self.unlabel_training:
            return img, target, i

        return img, target

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.data)

    def filter_labels(self, labels_to_chose: List):
        """ only items with `labels_to_chose` lables are returned """
        # remove other label from class_indices
        _tmp = {}
        for lab in self.class_indices:
            if lab in labels_to_chose:
                _tmp[lab] = self.class_indices[lab]
        self.class_indices = _tmp
        self.indices = list(chain.from_iterable(self.class_indices.values()))

    @property
    def classes(self) -> List[int]:
        """get list of classes
        """
        return list(sorted(self.class_indices.keys()))

    def filter_proportional_label(self,
                                  max_per_class: Optional[int] = None,
                                  labels_to_chose: List = None):
        """make the maximum number of items per class to 
            be `max_per_class`
        """
        if max_per_class is None:
            return
        if max_per_class == -1:
            # get minimum number of items per class for each category
            max_per_class = min(len(v) for _, v in self.class_indices.items())
        _tmp = defaultdict(list)
        if not labels_to_chose:
            labels_to_chose = list(sorted(self.class_indices.keys()))
        for k in labels_to_chose:
            if k in self.class_indices:
                v = self.class_indices[k]

                if max_per_class <= 1:
                    size = int(len(v) * max_per_class)
                else:
                    size = max_per_class
                _replace = False if len(v) > max_per_class else True
                _tmp[k].extend(
                    list(self.rng.choice(v, size=size, replace=_replace)))
        self.class_indices = _tmp
        self.indices = list(chain.from_iterable(self.class_indices.values()))


def map_ind_to_label(dataset_name, data):
    tmpfile = os.path.join(TMP_PATH, dataset_name + f"_indices.npy")
    if not os.path.exists(tmpfile):
        sub_meta_indices = _get_ind_to_label(data, dataset_name)
        if not os.path.exists(TMP_PATH):
            os.makedirs(TMP_PATH, exist_ok=True)

        np.save(os.path.join(TMP_PATH, dataset_name + f"_indices.npy"),
                sub_meta_indices)


def _get_ind_to_label(data, dataset_name=None):
    sub_meta_indices = {}

    # Dummy dataset to be passed to DataLoader
    class LoaderInd:
        def __init__(self, data) -> None:
            self.data = data

        def __len__(self):
            return len(data)

        def __getitem__(self, index):
            try:
                _, label = self.data[index]
            except FileNotFoundError:
                return None, None
            return label, index

    _loader = torch.utils.data.DataLoader(LoaderInd(data),
                                          batch_size=None,
                                          batch_sampler=None,
                                          collate_fn=identity,
                                          num_workers=60,
                                          shuffle=False)
    for label, i in tqdm(_loader,
                         total=len(data),
                         desc=f"storing indices {dataset_name}: "):
        if label is None:
            continue
        if label not in sub_meta_indices:
            sub_meta_indices[label] = []
        sub_meta_indices[label].append(i)

    return sub_meta_indices


def prepare_data_indices(dataset_name, data_path, split_type, opt=None):
    base_dataset_name, data_indices_suffix, mode = get_split(
        dataset_name, split_type)
    indfile = os.path.join(TMP_PATH, base_dataset_name + f"_indices.npy")

    if not os.path.exists(indfile):
        data, *_ = get_image_folder(dataset_name, data_path, split_type)
        map_ind_to_label(base_dataset_name, data)
    if data_indices_suffix:
        tmpfile = os.path.join(
            TMP_PATH, base_dataset_name +
            f"_indices{data_indices_suffix}_{mode}_{opt.split_fraction}" +
            f"_{opt.split_seed}" + ".npy")
        if not os.path.exists(tmpfile):
            data_dict = np.load(indfile, allow_pickle=True).item()
            if "disjoint" in data_indices_suffix:
                helper.create_disjoint_indices(data_dict,
                                               base_dataset_name,
                                               num_split=4,
                                               min_way=opt.n_way,
                                               fraction=opt.split_fraction,
                                               seed=opt.split_seed)

            elif "sup" in data_indices_suffix or "unsup" in data_indices_suffix or "partial" in data_indices_suffix:
                helper.create_partial_data(data_dict,
                                           base_dataset_name,
                                           fraction=opt.split_fraction,
                                           seed=opt.split_seed)


@rank_zero_only
def loguru_log(msg):
    logger.info(msg)


class SetDataset(torch.utils.data.Dataset):
    """Dataset to generate few-shot episode"""
    def __init__(self,
                 data_path,
                 batch_size,
                 transform,
                 dataset_name=None,
                 opt=None,
                 split_type=None):

        base_dataset_name, data_indices_suffix, mode = get_split(
            dataset_name, split_type)
        if split_type is not None:
            data_indices_suffix = "_" + split_type
        if mode is None:
            mode = ""
            data_train = SimpleDataset(data_path,
                                       split_type,
                                       transform,
                                       dataset_name=base_dataset_name +
                                       "_train",
                                       opt=opt)
            data_test = SimpleDataset(data_path,
                                      split_type,
                                      transform,
                                      dataset_name=base_dataset_name + "_test",
                                      opt=opt)
            self.data = torch.utils.data.ConcatDataset((data_train, data_test))
        else:
            self.data = SimpleDataset(data_path,
                                      split_type,
                                      transform,
                                      dataset_name=base_dataset_name +
                                      f"_{mode}",
                                      opt=opt)
        tmpfile = os.path.join(
            TMP_PATH, base_dataset_name +
            f"{mode}_fs_indices_{data_indices_suffix}_{opt.split_fraction}_{opt.split_seed}.npy"
        )
        if not os.path.exists(tmpfile):
            self.sub_meta_indices = _get_ind_to_label(self.data,
                                                      base_dataset_name)
            np.save(tmpfile, self.sub_meta_indices)
        else:
            loguru_log(f"loading indices from {tmpfile}")
            self.sub_meta_indices = np.load(tmpfile, allow_pickle=True).item()

        self.cl_list = list(sorted(self.sub_meta_indices.keys()))

        # check if any data less than batch size
        self.rng = np.random.RandomState(seed=2183647 + opt.seed)
        for lab in self.sub_meta_indices:
            if len(self.sub_meta_indices[lab]) < batch_size:
                _orig = self.sub_meta_indices[lab]
                _needed = batch_size - len(_orig)
                _extra = self.rng.choice(_orig, size=_needed, replace=True)
                _new = np.concatenate((_orig, _extra), axis=0)
                self.sub_meta_indices[lab] = _new

        self.sub_dataloader = []

        self.sub_dataloader_iter = []

        self.sub_dataset = []
        self.batch_size = batch_size

        self.generator = torch.Generator().manual_seed(opt.seed + 214743647)

        sub_data_loader_params = dict(
            batch_size=batch_size,
            num_workers=0,  #use main thread only or may receive multiple batches
            pin_memory=False,
            drop_last=True,
            shuffle=True,
            generator=self.generator)

        for cl in self.cl_list:
            sub_dataset = SubDataset(self.data,
                                     self.sub_meta_indices[cl],
                                     cl,
                                     transform=None)

            self.sub_dataset.append(sub_dataset)

            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset,
                                            **sub_data_loader_params))
            self.sub_dataloader_iter.append(iter(self.sub_dataloader[-1]))
        loguru_log(
            f"loaded dataset {dataset_name}:: #class: {len(self.sub_meta_indices.keys())},"
            +
            f" #data: {sum(len(v) for _, v in self.sub_meta_indices.items())}")

    def __getitem__(self, i):
        try:
            return next(self.sub_dataloader_iter[i])
        except StopIteration:
            self.sub_dataloader_iter[i] = iter(self.sub_dataloader[i])
            return next(self.sub_dataloader_iter[i])

    def __len__(self):
        return len(self.sub_dataloader)


class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self,
                 image_size,
                 batch_size,
                 dataset_name=None,
                 split_type=None,
                 unlabel=False,
                 opt=None,
                 seed=0):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size,
                                            normalize_type=opt.normalize_type)
        self.dataset_name = dataset_name
        self.unlabel = unlabel

        self.split_type = split_type
        self.opt = opt

        self.rng = np.random.RandomState(seed=self.opt.seed + 111 + seed +
                                         get_rank())

    def get_unlabel_loader(
            self,
            data_path,
            labels_to_chose=None,
            num_classes=None,
            labels_dict=None,
            aug=True,
            return_data_idx=False,
            consecutive_label=False,
            drop_last=True,
            shuffle=True,
            limit_data=1000,
            filter_data=True):  #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)

        if isinstance(data_path, list):
            assert isinstance(self.dataset_name, list)
            assert len(data_path) == len(self.dataset_name)
            list_dataset = []
            for i, _ in enumerate(data_path):
                _dataset = self._get_dataset(data_path[i],
                                             transform,
                                             dataset_name=self.dataset_name[i])
                list_dataset.append(_dataset)
            # dataset = torch.utils.data.ConcatDataset(list_dataset)
            dataset = ConcatProportionDataset(list_dataset,
                                              return_data_idx=return_data_idx)
        else:
            dataset = self._get_dataset(data_path,
                                        transform,
                                        dataset_name=self.dataset_name,
                                        consecutive_label=consecutive_label)

        try:
            class_list = dataset.classes
        except AttributeError:
            class_list = []

        # class_common = []
        rem_class = class_list
        classes_to_chose = []
        if labels_to_chose is not None:
            # chose specific classes
            rem_class = list(set(class_list) - set(labels_to_chose))
            classes_to_chose = list(labels_to_chose)

        if num_classes is not None:
            # select `num_classes` classes
            if len(classes_to_chose) < num_classes:
                classes_to_chose = classes_to_chose + list(
                    self.rng.choice(rem_class,
                                    size=num_classes - len(classes_to_chose),
                                    replace=True))
        if len(classes_to_chose) > 0:
            dataset.filter_labels(classes_to_chose)

        if limit_data:
            if limit_data <= 1:
                limit_data = int(len(dataset) * limit_data)

            if classes_to_chose:
                max_per_class = int(limit_data / len(classes_to_chose))
            else:
                max_per_class = limit_data / len(dataset)

            dataset.filter_proportional_label(max_per_class=max_per_class,
                                              labels_to_chose=classes_to_chose)

        else:
            if filter_data:
                dataset.filter_proportional_label(max_per_class=-1)

        loguru_log(labels_to_chose)
        loguru_log(classes_to_chose)

        # if labels_dict is not None:
        dataset = DatasetLabelDict(dataset, labels_dict)

        loguru_log(
            f"loaded dataset {self.dataset_name}:: #data: {len(dataset)}")

        data_loader_params = dict(
            batch_size=self.batch_size,
            num_workers=self.opt.num_workers,
            pin_memory=False,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.opt.seed + 4233647),
            drop_last=drop_last)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  **data_loader_params)

        return data_loader, dataset

    def filter_unlabel_dataloader(self,
                                  dataset: SimpleDataset,
                                  unlabel_idx=None,
                                  unlabel_y=None,
                                  aug=True,
                                  shuffle=True,
                                  drop_last=True):
        # get selected indices only
        transform = self.trans_loader.get_composed_transform(aug)

        dataset = SubSetWithLabel(dataset, unlabel_idx, unlabel_y)
        dataset.set_transform(transform)

        loguru_log(f"class: {dataset.dataset.dataset.classes}")
        loguru_log(
            f"loaded dataset {self.dataset_name}:: #data: {len(dataset)}")

        data_loader_params = dict(
            batch_size=self.batch_size,
            num_workers=self.opt.num_workers,
            pin_memory=False,
            shuffle=shuffle,
            #   timeout=5,
            drop_last=drop_last)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  **data_loader_params)
        return data_loader

    def get_data_loader(
            self,
            data_path,
            aug=True,
            return_data_idx=False,
            consecutive_label=False,
            limit_data=None,
            drop_last=True,
            shuffle=True):  #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)

        if isinstance(data_path, list):
            assert isinstance(self.dataset_name, list)
            assert len(data_path) == len(self.dataset_name)
            list_dataset = []
            for i, _ in enumerate(data_path):
                _dataset = self._get_dataset(data_path[i],
                                             transform,
                                             dataset_name=self.dataset_name[i])
                list_dataset.append(_dataset)
            # dataset = torch.utils.data.ConcatDataset(list_dataset)
            dataset = ConcatProportionDataset(list_dataset,
                                              return_data_idx=return_data_idx)
        else:
            dataset = self._get_dataset(data_path,
                                        transform,
                                        dataset_name=self.dataset_name,
                                        consecutive_label=consecutive_label)

        if limit_data and limit_data != 1:
            if limit_data <= 1:
                limit_len = int(len(dataset) * limit_data)
            else:
                limit_len = int(limit_data)
            rng = np.random.RandomState(seed=self.opt.seed + 21474647 +
                                        get_rank())
            limit_indices = rng.choice(
                len(dataset),
                limit_len,
                replace=False if limit_len <= len(dataset) else True)
            dataset = torch.utils.data.Subset(dataset, limit_indices)

        loguru_log(
            f"loaded dataset {self.dataset_name}:: #data: {len(dataset)}")

        if self.batch_size == -1:
            self.batch_size = len(dataset)
        data_loader_params = dict(
            batch_size=self.batch_size,
            num_workers=self.opt.num_workers,
            pin_memory=False,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.opt.seed + 4743647),
            drop_last=drop_last)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  **data_loader_params)

        return data_loader

    def _get_dataset(self,
                     data_path,
                     transform,
                     dataset_name,
                     consecutive_label=False):
        dataset = SimpleDataset(data_path,
                                split_type=self.split_type,
                                transform=transform,
                                dataset_name=dataset_name,
                                consecutive_label=consecutive_label,
                                opt=self.opt,
                                unlabel_training=self.unlabel,
                                raise_error=False)

        return dataset


class DatasetLabelDict:
    def __init__(self, dataset: SimpleDataset, labels_dict=None):
        self.dataset = dataset
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        x, y, i = batch
        if self.labels_dict is not None:
            y = self.labels_dict.get(y, len(self.labels_dict) + 1)
        return x, y, idx


class SubSetWithLabel(torch.utils.data.Subset):
    def __init__(self,
                 dataset: DatasetLabelDict,
                 indices,
                 labels=None) -> None:
        super().__init__(dataset, indices)
        self.labelsU = labels

    def set_transform(self, tsfm):
        self.dataset.dataset.transform = tsfm

    def __getitem__(self, idx):
        batch = self.dataset[self.indices[idx]]
        x, y = batch[0], batch[1]
        if self.labelsU is None:
            return x, y
        else:
            y = self.labelsU[idx]
            return x, y


class SetDataManager(DataManager):
    def __init__(self,
                 data_path,
                 image_size=224,
                 n_way=5,
                 n_shot=5,
                 n_query=16,
                 num_episodes=100,
                 aug=False,
                 dataset_name=None,
                 opt=None,
                 split_type=None,
                 **kwargs):
        super().__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.opt = opt
        self.split_type = split_type
        self.batch_size = n_shot + n_query
        self.num_episodes = num_episodes

        self.trans_loader = TransformLoader(image_size)

        transform = self.trans_loader.get_composed_transform(aug)
        self.dataset = SetDataset(data_path,
                                  self.batch_size,
                                  transform,
                                  dataset_name=dataset_name,
                                  opt=opt,
                                  split_type=split_type,
                                  **kwargs)

    def get_data_loader(
        self,
        # aug=False,
        use_ddp=False,
        dist_args={},
    ):  #parameters that would change on train/val set

        if use_ddp is False and self.opt.replica_rank is None:
            batch_sampler = EpisodicBatchSampler(len(self.dataset),
                                                 self.n_way,
                                                 self.num_episodes,
                                                 seed=self.opt.seed)
            # worker_init_fn = init_seed
        else:
            if use_ddp:
                rank = dist_args["rank"]
                num_replicas = dist_args["num_replicas"]
            else:
                rank = self.opt.replica_rank
                num_replicas = self.opt.num_replica

            batch_sampler = DistEpisodicBatchSampler(
                len(self.dataset),
                n_way=self.n_way,
                num_episodess=self.num_episodes,
                num_replicas=num_replicas,
                rank=rank,
                seed=self.opt.seed)  # seed has to be fixed for all

            # def init_fn(x):
            #     init_seed(dist_args["rank"] + x)
            # worker_init_fn = init_fn

        data_loader_params = dict(batch_sampler=batch_sampler,
                                  num_workers=self.opt.num_workers,
                                  pin_memory=False)
        data_loader = torch.utils.data.DataLoader(self.dataset,
                                                  **data_loader_params)
        return data_loader


class SubDataset:
    def __init__(self,
                 data_orig,
                 indices,
                 cl,
                 transform=torchvision.transforms.ToTensor(),
                 target_transform=identity):
        self.sub_meta_indices = indices
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform
        self.data = data_orig

    def __getitem__(self, i):
        idx = self.sub_meta_indices[i]
        img, lab = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        target = self.target_transform(self.cl)

        # sanity check
        assert lab == self.cl

        return img, target

    def __len__(self):
        return len(self.sub_meta_indices)
