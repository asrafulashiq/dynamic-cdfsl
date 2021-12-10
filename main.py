from typing import List
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks.base import Callback
import torch
import pytorch_lightning as pl
import argparse
from utils.custom_logger import CustomLogger
from data_loader.data_module import DataModule
from helper import load_system
from helper import config_init, refine_args
from helper.helper_slurm import run_cluster


def main(params: DictConfig, LightningSystem: pl.LightningModule, *args,
         **kwargs):
    params = config_init(params)
    params = refine_args(params)

    datamodule = DataModule(params.data)

    # Init PyTorch Lightning model ⚡
    model = LightningSystem(params, datamodule)

    if params.ckpt is not None and params.ckpt != 'none':
        if params.load_base:
            model.load_base(params.ckpt)
        else:
            ckpt = torch.load(
                params.ckpt,
                map_location=lambda storage, loc: storage)['state_dict']
            model.load_state_dict(ckpt, strict=not params.load_flexible)

    logger = CustomLogger(save_dir=params.logger.save_dir,
                          name=params.logger.name,
                          version=params.logger.version,
                          test=params.test,
                          disable_logfile=params.disable_logfile)

    # Init PyTorch Lightning callbacks ⚡
    callbacks: List[Callback] = [
        hydra.utils.instantiate(callback_conf)
        for _, callback_conf in params["callbacks"].items()
    ] if "callbacks" in params else []

    trainer = pl.Trainer.from_argparse_args(
        argparse.Namespace(**params.trainer),
        logger=logger,
        callbacks=callbacks,
        limit_test_batches=params.trainer.limit_val_batches)

    if params.test:
        out = trainer.test(model)
        return out
    else:
        return trainer.fit(model)


@hydra.main(config_name="config", config_path="configs")
def hydra_main(cfg: DictConfig):
    lt_system = load_system(cfg.system_name)

    if cfg.launcher.name == "local":
        # add Lightning parse
        main(cfg, lt_system)
    elif cfg.launcher.name == "slurm":
        # submit job to slurm
        run_cluster(cfg, main, lt_system)
    elif cfg.launcher.name == "submitit_eval":
        from helper.helper_submitit_eval import submitit_eval_main
        submitit_eval_main(cfg, lt_system)


if __name__ == "__main__":
    hydra_main()
