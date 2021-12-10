from typing import Any, Callable, Dict
import submitit
from omegaconf import DictConfig
from rich import print
import os
import pytorch_lightning as pl
import hydra


def submitit_main(cfg_launcher: DictConfig,
                  fn_main: Callable,
                  params: DictConfig = None,
                  *args,
                  verbose=True,
                  **kwargs) -> submitit.Job:

    folder = os.path.join(cfg_launcher.log_root, params.model_name)
    # executor = submitit.SlurmExecutor(folder=folder, max_num_timeout=0)
    executor = hydra.utils.instantiate(cfg_launcher.executor, folder=folder)

    time = str(cfg_launcher.time)
    if not ':' in time:
        time = f"{int(time):02d}:00:00"

    executor.update_parameters(job_name=params.model_name,
                               **cfg_launcher.params_submitit,
                               time=time,
                               setup=list(cfg_launcher.setup))

    job = executor.submit(fn_main, params, *args, **kwargs)

    if verbose:
        print("_" * 60)
        print("_" * 60)
        print("job_id :", job.job_id)
        print("sbatch script : ", job.paths.submission_file)
        print("stderr : ", job.paths.stderr)
        print("stdout : ", job.paths.stdout)

    return job
