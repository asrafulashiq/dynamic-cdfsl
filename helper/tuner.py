"""Hyperparameter tuning class using submitit
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import pytorch_lightning as pl
from omegaconf import DictConfig
from helper.helper_submitit import submitit_main
import copy
import submitit
from omegaconf import OmegaConf
from omegaconf import open_dict
import time
import os
from tqdm import tqdm
import pickle
import pandas as pd
from utils.custom_logger import CustomLogger


class HPJob:
    def __init__(self,
                 job: Optional[submitit.Job] = None,
                 conf=None,
                 done=False,
                 result=None,
                 job_info=None) -> None:
        self.job = job
        self.conf = sorted_dict(conf) if conf else conf
        self.result = result

        self.job_info = job_info
        if self.job is not None:
            self.job_info = self.get_info()

        self._done = done

    def check_done(self):
        if self.job is None:
            return True
        else:
            status = self.job.done()
            if status:
                self.result = self.job.result()
                self._done = True
                return True
            else:
                return False

    @property
    def done(self):
        return self._done

    def get_info(self):
        if self.job_info is not None:
            return self.job_info
        return {
            "job_id": self.job.job_id,
            "sbatch script": str(self.job.paths.submission_file),
            "stderr": str(self.job.paths.stderr),
            "stdout": str(self.job.paths.stdout)
        }

    def serialize(self):
        if not self.check_done():
            return None
        else:
            return HPJob(job=None,
                         conf=self.conf,
                         done=True,
                         result=self.result,
                         job_info=self.job_info)


class HPTuner(object):
    def __init__(self,
                 fnc_main: Callable,
                 cfg: DictConfig,
                 LightningSystem: pl.LightningModule,
                 submitter: Callable,
                 search_space: List,
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

    def _submit_one_job(self, conf_to_tune: Dict) -> HPJob:
        param_job = merge_conf(conf_to_tune, self.params)

        param_job.model_name = (param_job.model_name + "-" +
                                convert_dict_to_filestring(conf_to_tune))
        param_job.launcher.log_root = os.path.join(param_job.launcher.log_root,
                                                   "tune")
        param_job.logger.version = 0
        param_job.data.num_episodes = param_job.tune.num_episodes

        submitit_job = submitit_main(param_job.launcher,
                                     self.fnc_main,
                                     params=param_job,
                                     LightningSystem=self.lt_system,
                                     verbose=False)
        hpjob = HPJob(submitit_job, conf=conf_to_tune)
        self.logger.log(OmegaConf.to_yaml(hpjob.get_info()))
        self.csv_paths.append({
            "path":
            os.path.join("test_" + param_job.logger.save_dir,
                         param_job.model_name, f"version_{0}", "metrics.csv"),
            "conf":
            conf_to_tune
        })
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
        items = []
        for i, dat in enumerate(self.csv_paths):
            path = dat["path"]
            conf = dat["conf"]
            df = pd.read_csv(path).dropna()
            average = df['accuracy'].astype(float).mean()
            items.append((average, conf))
        self._print(items)
        best_res, best_conf = self.best_result(items)
        self.logger.log("\nBest" + "*" * 100 + "\n" +
                        str(OmegaConf.to_yaml(best_conf)) + "\nScore: " +
                        str(best_res))

        return best_res, best_conf

    def _print(self, items):
        items_sorted = sorted(items, key=lambda x: x[0])

        _str = "\n" + ("*" * 60) + "\n"
        num_best = self.params.tune.num_best
        num_best = num_best if num_best else len(items_sorted)
        for res, conf in items_sorted[-num_best:]:
            _str += f"Result: {res:.2f}\n" + ("-" * 20) + "\n"
            _str += f"{OmegaConf.to_yaml(conf)}\n"
        self.logger.log(_str)

    def best_result(self, items) -> Tuple[dict, float]:
        if self.params.tune.mode_metric == 'max':
            best_item = max(items, key=lambda x: x[0])
        else:
            best_item = min(items, key=lambda x: x[0])
        return best_item

    @property
    def file_name_to_save(self):
        if self._save_file:
            return self._save_file
        os.makedirs(self.params.tune.save_tune_dir, exist_ok=True)
        self._save_file = os.path.join(self.params.tune.save_tune_dir,
                                       self.params.model_name + ".pkl")
        return self._save_file

    def reload_existing(self):
        """load previous results
        """
        db_file = self.file_name_to_save
        if os.path.exists(db_file):
            with open(db_file, 'rb') as fp:
                self.csv_paths = pickle.load(fp)
                self.logger.log("loading exisiting database")
                self.logger.log(f"completed job: {len(self.csv_paths)}")

    def save(self):
        with open(self.file_name_to_save, "wb") as fp:
            pickle.dump(self.csv_paths, fp)


# class HPTuner:
#     def __init__(
#             self,
#             fnc_main: Callable,
#             cfg: DictConfig,
#             LightningSystem: pl.LightningModule,
#             submitter: Callable,
#             search_space: List,
#             resume: bool = True,
#     ) -> None:
#         self.fnc_main = fnc_main
#         self.lt_system = LightningSystem
#         self.submitter = submitter

#         self.search_space = search_space
#         self.params = cfg

#         # class specific args
#         self.hpjob_info: Dict[str, HPJob] = {}

#         self._save_file = None

#         if resume:
#             self.reload_existing()

#     def submit(self):
#         """submit all jobs

#         Args:
#             as_new (bool, optional): whether to run previously completed job. Defaults to False.
#         """
#         for each_conf in tqdm(self.search_space, desc="Submitted"):
#             conf_str = str(each_conf)
#             if conf_str in self.hpjob_info:
#                 continue
#             hpjob = self._submit_one_job(each_conf)
#             self.hpjob_info[conf_str] = hpjob
#             time.sleep(2)

#     def wait(self):
#         pbar = tqdm(total=len(self.search_space), desc="Job completion: ")
#         num_finished = self._check_finished()
#         while num_finished < len(self.search_space):
#             time.sleep(10)

#             _prev = num_finished
#             num_finished = self._check_finished()

#             pbar.update(n=num_finished - _prev)
#         pbar.close()
#         self.store_db()  # store results

#     def print_results(self):
#         items = [(float(hpjob.result[0][self.params.tune.metric]), hpjob.conf)
#                  for hpjob in self.hpjob_info.values() if hpjob.done]
#         items_sorted = sorted(items, key=lambda x: x[0])

#         _str = "\n" + ("*" * 60) + "\n"
#         num_best = self.params.tune.num_best
#         num_best = num_best if num_best else len(items_sorted)
#         for res, conf in items_sorted[-num_best:]:
#             _str += f"Result: {res:.2f}\n" + ("-" * 20) + "\n"
#             _str += f"{OmegaConf.to_yaml(conf)}\n"
#         logger.info(_str)

#     def best_result(self) -> Tuple[dict, float]:
#         items = [(float(hpjob.result[0][self.params.tune.metric]), hpjob.conf)
#                  for hpjob in self.hpjob_info.values() if hpjob.done]
#         if self.params.tune.mode_metric == 'max':
#             best_item = max(items, key=lambda x: x[0])
#         else:
#             best_item = min(items, key=lambda x: x[0])
#         return best_item

#     def _check_finished(self) -> int:
#         for _, hpjob in self.hpjob_info.items():
#             if hpjob.done is False:
#                 if hpjob.check_done():
#                     # save current state
#                     logger.info(f"Job {hpjob.job.job_id} completed")
#                     logger.info("\n" + str(hpjob.result))

#                     self.store_db()
#         num_finished = sum(hpjob.done for _, hpjob in self.hpjob_info.items())
#         return num_finished

#     def _submit_one_job(self, conf_to_tune: Dict) -> HPJob:
#         param_job = merge_conf(conf_to_tune, self.params)
#         param_job.model_name = (param_job.model_name + "-" +
#                                 convert_dict_to_filestring(conf_to_tune))
#         param_job.launcher.log_root = os.path.join(param_job.launcher.log_root,
#                                                    "tune")
#         # disable log to file to avoid clutter
#         # param_job.disable_logfile = True
#         param_job.data.num_episodes = param_job.tune.num_episodes

#         submitit_job = submitit_main(param_job.launcher,
#                                      self.fnc_main,
#                                      params=param_job,
#                                      LightningSystem=self.lt_system,
#                                      verbose=False)
#         hpjob = HPJob(submitit_job, conf=conf_to_tune)
#         if self.params.tune.verbose:
#             logger.info(OmegaConf.to_yaml(hpjob.get_info()))
#         return hpjob

#     def reload_existing(self):
#         """load previous results
#         """
#         db_file = self.file_name_to_save
#         if os.path.exists(db_file):
#             with open(db_file, 'rb') as fp:
#                 self.hpjob_info = pickle.load(fp)
#                 logger.info("loading exisiting database")
#                 logger.debug(f"completed job: {len(self.hpjob_info)}")

#     @property
#     def file_name_to_save(self):
#         if self._save_file:
#             return self._save_file
#         os.makedirs(self.params.tune.save_tune_dir, exist_ok=True)
#         self._save_file = os.path.join(self.params.tune.save_tune_dir,
#                                        self.params.model_name + ".pkl")
#         return self._save_file

#     def store_db(self):
#         """store current results in database
#         """
#         db = {}
#         for k, hpjob in self.hpjob_info.items():
#             if hpjob.done:
#                 db[k] = hpjob.serialize()

#         with open(self.file_name_to_save, "wb") as fp:
#             pickle.dump(db, fp)
#         # logger.info(f"Saved loader")


def merge_conf(config: dict, params: DictConfig) -> DictConfig:
    """merge conf with params and return a new params

    Args:
        config (dict): new key-value dict to update
        params (DictConfig): original parameters

    Returns:
        DictConfig: updated parameters
    """
    params_copy = copy.deepcopy(params)
    with open_dict(params_copy):
        for k, v in config.items():
            OmegaConf.update(params_copy, k, v, merge=True)
    return params_copy


def convert_dict_to_filestring(ddict: Dict) -> str:
    _str = ""
    ddict = sorted_dict(ddict)
    for k, v in ddict.items():
        _str += f"-{str(v)}"
    _str = _str.replace("[", "_").replace("]", "_")
    _str = _str.replace("\"", "").replace("\'", "")
    return _str


def sorted_dict(x: Sequence) -> Dict:
    """sort dictionary by key
    """
    return dict(sorted(x.items(), key=lambda item: item[0]))
