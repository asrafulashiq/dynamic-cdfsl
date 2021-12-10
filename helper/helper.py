import collections
from utils.utils_system import get_rank
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from importlib import import_module
import pytorch_lightning as pl
import os
import torch
from omegaconf import OmegaConf

# NOTE reguster resolver for step lr scheduler, requires omegaconf: 2.1.0.dev26
OmegaConf.register_new_resolver("multiply",
                                lambda x, y: int(float(x) * float(y)))


def config_init(params: DictConfig):

    params = OmegaConf.create(OmegaConf.to_yaml(params, resolve=True))

    pl.seed_everything(params.seed + get_rank())

    # FIXME hard-coded for now
    if params.system_name in (
            "ce", "ce_ep200") and "tieredImageNet" in params.data.dataset:
        params.trainer.max_epochs = 100

    params.data.disable_validation = params.disable_validation

    if params.launcher.name == "slurm":
        config_init_slurm(params)

    if params.trainer.log_every_n_steps == -1:
        params.trainer.log_every_n_steps = int(1e10)

    if params.data.val_dataset is not None and ',' in params.data.val_dataset:
        params.data.val_dataset = list(params.data.val_dataset.split(','))

    if isinstance(params.data.val_dataset, list) and len(
            params.data.val_dataset) == 1:
        params.data.val_dataset = params.data.val_dataset[0]

    if not params.test and not os.path.isdir(params.trainer.weights_save_path):
        if rank_zero_only.rank == 0:
            os.makedirs(params.trainer.weights_save_path, exist_ok=True)

    if params.test:
        params.trainer.log_every_n_steps = 5

    if params.resume and os.path.exists(
            os.path.join(params.weights_save_path, 'last.ckpt')):
        params.trainer.resume_from_checkpoint = os.path.join(
            params.trainer.weights_save_path, 'last.ckpt')
        if params.test: params.ckpt = params.trainer.resume_from_checkpoint

    return params


def config_init_slurm(params):
    assert params.trainer.num_nodes == params.launcher.nodes
    assert (params.trainer.gpus == -1
            or params.trainer.gpus == params.launcher.gpus)


def refine_args(params):
    def fn(name):
        # convert comma-separated dataset to list
        if hasattr(params.trainer, name) and isinstance(
                getattr(params.trainer, name), str):
            val = getattr(params.trainer, name)
            _split = val.split(',')
            if len(_split) > 1:
                setattr(params.trainer, name, _split)

    fn('dataset')
    fn('val_dataset')

    # set backend properly
    if not isinstance(params.trainer.gpus, int):
        if params.trainer.gpus == '-1':
            params.trainer.gpus = -1
        else:
            _split = params.trainer.gpus.split(',')
            if len(_split) == 1:
                params.trainer.gpus = int(params.trainer.gpus)
            else:
                list_of_gpu = [int(k) for k in _split if k.isdigit()]
                params.trainer.gpus = list_of_gpu

    if ((isinstance(params.trainer.gpus, list)
         and len(params.trainer.gpus) > 1) or
        (isinstance(params.trainer.gpus, int)
         and params.trainer.gpus > 1)) or (params.trainer.gpus == -1):
        params.trainer.accelerator = 'ddp'

    return params


def load_system(system_name):
    if system_name is not None:
        module = import_module(f"system.system_{system_name}", __package__)
        lt_system = module.LightningSystem
    else:
        raise ValueError(f"should provide system!!")
    return lt_system


def nested_dict_to_dict(cfg: dict) -> dict:
    """Convert nested dictionary to one dictionary with '.' keys
    """
    out = {}
    for k, v in cfg.items():
        if isinstance(v, collections.abc.Mapping):
            v = nested_dict_to_dict(dict(v))
            for nk, nv in v.items():
                out[k + "." + nk] = nv
        else:
            if isinstance(v, collections.Sequence):
                v = list(v)
            out[k] = v
    return out


def dict_to_nested_dict(cfg: dict) -> dict:
    """Convert dictionary  with '.' keys to nested dictionary
    """
    out = {}
    for k, v in cfg.items():
        if "." in k:
            node = out
            ksplits = k.split(".")
            for level_k in ksplits[:-1]:
                if level_k not in node:
                    node[level_k] = {}
                node = node[level_k]
            node[ksplits[-1]] = v
        else:
            out[k] = v
    return out


def tune_config_create(tune_params: collections.abc.Mapping) -> dict:
    results = {}
    tune_params = nested_dict_to_dict(dict(tune_params))
    for k, v in tune_params.items():
        if isinstance(v, collections.abc.Sequence):
            v = tune.choice(list(v))
        elif isinstance(v, collections.abc.Mapping):
            v = tune_config_create(dict(v))
        results[k] = v
    return results


class TuneReportCallback(Callback):
    def __init__(self, metric="acc_mean") -> None:
        super().__init__()
        self.metric = metric

    def search_in_dict(self, metric, results):
        out = None
        for k, v in results.items():
            if metric in k:
                out = v
        return out

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx):
        # out = self.search_in_dict(self.metric, trainer.callback_metrics)
        # if out is None:
        #     out = self.search_in_dict(self.metric, trainer.logged_metrics)
        out = outputs[self.metric]
        if isinstance(out, torch.Tensor): out = out.item()
        tune.report(**{self.metric: out})

    def on_test_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.on_validation_batch_end(trainer, pl_module, *args, **kwargs)
