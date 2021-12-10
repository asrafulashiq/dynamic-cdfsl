from typing import Any, Optional, Union
import hydra
import numpy as np
from omegaconf.dictconfig import DictConfig
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.step_result import recursive_gather, recursive_stack
import random
import torchvision
from omegaconf import OmegaConf
from contextlib import contextmanager
from .custom_metrics import *


def to_container(conf):
    return OmegaConf.to_container(conf, resolve=True)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def build_base_encoder(
        backbone='resnet10',
        pretrained=False,
        model_args: Optional[DictConfig] = None) -> torch.nn.Module:
    if backbone.lower() == "resnet10":
        # used in CDFSL benchmark
        from utils import backbone
        encoder = backbone.ResNet10(flatten=True)
    elif backbone.lower() == "resnet12":
        from utils.resnet12 import resnet12
        encoder = resnet12(avg_pool=True)
        encoder.final_feat_dim = 640
    elif "random" in backbone:
        encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        encoder.final_feat_dim = 5
    elif "resnet" in backbone:
        resnet_head = getattr(torchvision.models.resnet,
                              backbone)(pretrained=pretrained)

        encoder = nn.Sequential(*(list(resnet_head.children())[:-1] +
                                  [nn.Flatten()]))
        encoder.final_feat_dim = resnet_head.fc.in_features
    elif "deit" in backbone:
        encoder = hydra.utils.instantiate(model_args)
        encoder.final_feat_dim = encoder.num_features
    else:
        raise NotImplementedError(f"{backbone} not implemented")

    return encoder


class AverageMeter():
    def __init__(self, use_ddp=False):
        super().__init__()
        self.reset()
        self.use_ddp = use_ddp
        self.best_val = -1

    def add(
        self,
        value,
        n=1,
    ):
        if self.use_ddp is False:
            self.val = value
            self.sum += value
            self.var += value * value
            self.n += n
        else:
            var = value**2
            dist.all_reduce(value)
            dist.all_reduce(var)
            self.sum += value
            self.var += var
            self.n += n * dist.get_world_size()
            self.val = value / dist.get_world_size()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0

    @property
    def value(self):
        return self.val

    @property
    def mean(self):
        return (self.sum / self.n)

    @property
    def avg(self):
        return self.mean

    @property
    def max(self):
        mean_val = self.mean
        if self.use_ddp:
            dist.all_reduce(mean_val)
            mean_val /= dist.get_world_size()

        if mean_val > self.best_val:
            self.best_val = mean_val

        return self.best_val

    @property
    def std(self):
        acc_std = torch.sqrt(self.var / self.n - self.mean**2)
        return 1.96 * acc_std / np.sqrt(self.n)


def set_bn_to_eval(mod):
    for m in mod.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def aggregate_dict(inputs):
    results = recursive_gather(inputs, {})
    recursive_stack(results)
    return results


@contextmanager
def patch_conf(conf, new_conf={}):
    new_conf = OmegaConf.merge(conf, new_conf)
    try:
        yield new_conf
    finally:
        pass


@torch.no_grad()
def concat_all_ddp(tensor):
    try:
        world_size = torch.distributed.get_world_size()
    except AssertionError:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# moco utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def moco_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1. / batch_size))
        return res


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def reset_modules(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def set_bn_to_eval(module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def to_numpy(x: Union[torch.Tensor, Any]) -> Union[np.ndarray, Any]:
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        return x.data.numpy()
    else:
        return x