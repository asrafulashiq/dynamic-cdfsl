import os
import pickle
import time
import numpy as np
from loguru import logger
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm
from typing import Any, Callable, Dict, List
from helper.helper_submitit import submitit_main
from omegaconf import DictConfig
from utils.custom_logger import CustomLogger
from helper.tuner import HPJob, convert_dict_to_filestring, merge_conf
from main import main
from omegaconf import OmegaConf

logger.configure(handlers=[
    dict(
        sink=lambda msg: tqdm.write(msg, end=''),
        level='DEBUG',
        colorize=True,
        format="<green>{time: MM-DD at HH:mm}</green> <level>{message}</level>",
        enqueue=True),
])


def submitit_eval_main(params: DictConfig,
                       LightningSystem: pl.LightningModule):

    # params.data.num_episodes = int(
    #     np.ceil(params.data.num_episodes / params.launcher.num_runs))

    search_space = [{
        "data.replica_rank": i
    } for i in range(params.launcher.num_runs)]

    params.data.num_replica = params.launcher.num_runs
    custom_logger = CustomLogger(save_dir=params.logger.save_dir,
                                 name=params.model_name,
                                 version=params.logger.version,
                                 test=params.test,
                                 disable_logfile=params.disable_logfile)

    print(f"log to {custom_logger.log_dir}")

    hp_tuner = HPRandomTuner(main,
                             params,
                             LightningSystem,
                             submitit_main,
                             search_space,
                             resume=params.launcher.resume,
                             logger=custom_logger)

    if params.launcher.mode == "submit":
        hp_tuner.submit()
    else:
        hp_tuner.reload_existing()
        hp_tuner.print_results()


class HPRandomTuner(object):
    def __init__(self,
                 fnc_main: Callable,
                 cfg: DictConfig,
                 LightningSystem: pl.LightningModule,
                 submitter: Callable,
                 search_space: List,
                 resume: bool,
                 logger: CustomLogger = None) -> None:
        self.fnc_main = fnc_main
        self.lt_system = LightningSystem
        self.submitter = submitter

        self.search_space = search_space
        self.params = cfg

        # class specific args
        self._save_file = None

        self.logger = logger
        self.csv_paths = []

        if resume:
            self.reload_existing()

    def _submit_one_job(self, conf_to_tune: Dict) -> HPJob:
        param_job = merge_conf(conf_to_tune, self.params)

        param_job.model_name = (param_job.model_name + "_rand-" +
                                convert_dict_to_filestring(conf_to_tune))
        param_job.launcher.log_root = os.path.join(param_job.launcher.log_root,
                                                   "fs_submitit")
        param_job.logger.version = 0

        submitit_job = submitit_main(param_job.launcher,
                                     self.fnc_main,
                                     params=param_job,
                                     LightningSystem=self.lt_system,
                                     verbose=False,
                                     return_path=True)
        hpjob = HPJob(submitit_job, conf=conf_to_tune)
        self.logger.experiment.info(OmegaConf.to_yaml(hpjob.get_info()))
        self.csv_paths.append(
            os.path.join("test_" + param_job.logger.save_dir,
                         param_job.model_name, f"version_0", "metrics.csv"))
        return hpjob

    def submit(self):
        """submit all jobs

        Args:
            as_new (bool, optional): whether to run previously completed job. Defaults to False.
        """
        for each_conf in tqdm(self.search_space, desc="Submitted"):
            self._submit_one_job(each_conf)
            time.sleep(1)
        self.save()

    def print_results(self):
        dataframes = []
        for i, path in enumerate(self.csv_paths):
            df = pd.read_csv(path).dropna()
            dataframes.append(df)
            _str = f"{i}:\n" + "*" * 50 + "\n" + str(df.tail(n=1))
            self.logger.log(_str)
        dataframe = pd.concat(dataframes).reset_index()
        dataframe['accuracy'] = dataframe['accuracy'].astype(float)
        average = dataframe['accuracy'].mean()
        std = 1.96 * dataframe['accuracy'].std() / np.sqrt(dataframe.shape[0])
        self.logger.log_metrics({"acc_mean": average, "std": std}, step=-1)

    @property
    def file_name_to_save(self):
        if self._save_file:
            return self._save_file
        os.makedirs(self.params.launcher.save_tune_dir, exist_ok=True)
        self._save_file = os.path.join(self.params.launcher.save_tune_dir,
                                       self.params.model_name + ".pkl")
        return self._save_file

    def reload_existing(self):
        """load previous results
        """
        db_file = self.file_name_to_save
        if os.path.exists(db_file):
            with open(db_file, 'rb') as fp:
                self.csv_paths = pickle.load(fp)
                logger.info("loading exisiting database")
                logger.debug(f"completed job: {len(self.csv_paths)}")

    def save(self):
        with open(self.file_name_to_save, "wb") as fp:
            pickle.dump(self.csv_paths, fp)
