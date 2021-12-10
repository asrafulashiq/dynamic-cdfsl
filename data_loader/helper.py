import numpy as np
import os
from numpy.random import RandomState
import torch

TMP_PATH = os.path.expanduser("data/cache_cdfsl")


def create_partial_data(data_dict, dataset, fraction=0.5, seed=0):
    #NOTE: fraction is train split
    random_state = RandomState(seed)
    dict_train = {}
    dict_test = {}
    for k in data_dict:
        indices = data_dict[k]
        unsup_length = int(fraction * len(indices))

        random_state.shuffle(indices)
        dict_train[k] = indices[:unsup_length]
        dict_test[k] = indices[unsup_length:]

    np.save(
        os.path.join(TMP_PATH, dataset +
                     f"_indices_partial_train_{fraction}_{seed}.npy"),
        dict_train)
    np.save(
        os.path.join(TMP_PATH,
                     dataset + f"_indices_partial_test_{fraction}_{seed}.npy"),
        dict_test)


def create_disjoint_indices(data_dict,
                            dataset,
                            num_split=4,
                            min_way=5,
                            fraction=0.5,
                            seed=0):
    # NOTE fraction is split for train data
    num_classes = len(data_dict)
    for i_split in range(num_split):
        random_state = RandomState(seed + i_split)
        dict_unsupervised = {}
        dict_supervised = {}

        if num_classes >= 2 * min_way:
            unsupervised_classes = random_state.choice(num_classes,
                                                       int(num_classes *
                                                           fraction),
                                                       replace=False)
            supervised_classes = [
                c for c in range(num_classes) if c not in unsupervised_classes
            ]

        else:
            cls_list = np.arange(num_classes)
            random_state.shuffle(cls_list)
            num_unsup = max(int(num_classes * fraction), min_way)
            unsupervised_classes = cls_list[:num_unsup]
            supervised_classes = cls_list[-num_unsup:]

        for k in data_dict:
            if k in unsupervised_classes:
                dict_unsupervised[k] = data_dict[k]
            if k in supervised_classes:
                dict_supervised[k] = data_dict[k]

        np.save(
            os.path.join(
                TMP_PATH, dataset +
                f"_indices_disjoint_train_{fraction}_{seed+i_split}.npy"),
            dict_unsupervised)
        np.save(
            os.path.join(
                TMP_PATH, dataset +
                f"_indices_disjoint_test_{fraction}_{seed+i_split}.npy"),
            dict_supervised)


def create_overlap_data(data_dict, dataset, fraction=0.7, seed=0):
    num_classes = len(data_dict)
    random_state = RandomState(seed)

    dict_unsupervised = {}
    dict_supervised = {}

    min_way = int(num_classes * fraction)

    cls_list = np.arange(num_classes)
    random_state.shuffle(cls_list)

    num_unsup = min_way
    unsupervised_classes = cls_list[:num_unsup]
    supervised_classes = cls_list[-num_unsup:]

    for k in data_dict:
        if k in unsupervised_classes:
            dict_unsupervised[k] = data_dict[k]
        if k in supervised_classes:
            dict_supervised[k] = data_dict[k]

    print("Overlapped indices ",
          set(unsupervised_classes).intersection(set(supervised_classes)))

    np.save(
        os.path.join(TMP_PATH,
                     dataset + f"_indices_unsup_overlap_{fraction}.npy"),
        dict_unsupervised)
    np.save(
        os.path.join(TMP_PATH,
                     dataset + f"_indices_sup_overlap_{fraction}.npy"),
        dict_supervised)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, num_episodess, seed=0):
        self.n_classes = n_classes
        self.n_way = n_way
        self.num_episodess = num_episodess

        self.g = torch.Generator().manual_seed(seed + 2117483647)

    def __len__(self):
        return self.num_episodess

    def __iter__(self):
        for i in range(self.num_episodess):
            if self.n_classes >= self.n_way:
                yield torch.randperm(self.n_classes,
                                     generator=self.g)[:self.n_way]
            else:
                raise ValueError(
                    f"num-class {self.n_classes} is less than {self.n_way}")


class DistEpisodicBatchSampler(object):
    def __init__(self,
                 n_classes,
                 n_way,
                 num_episodess,
                 num_replicas=None,
                 rank=0,
                 seed=0):
        self.n_classes = n_classes
        self.n_way = n_way
        self.num_episodess = num_episodess

        # dist params
        self.num_replicas = num_replicas
        self.rank = rank

        self.g = torch.Generator().manual_seed(2117483647 + seed)

        # num sample to each replica
        self.num_samples = int(np.ceil(num_episodess / self.num_replicas))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            if self.n_classes >= self.n_way:
                for i in range(self.num_replicas):
                    out = torch.randperm(self.n_classes,
                                         generator=self.g)[:self.n_way]
                    if i == self.rank:
                        yield out
            else:
                raise ValueError(
                    f"num-class {self.n_classes} is less than {self.n_way}")
