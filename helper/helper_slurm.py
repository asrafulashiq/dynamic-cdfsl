"""Write helper script to submit job to slurm
"""

import os, sys
import subprocess
import platform
import datetime
from typing import Callable, List
from colorama import init, Fore
from omegaconf.dictconfig import DictConfig
import pytorch_lightning

init(autoreset=True)

SLURM_CMD = """#!/bin/bash

# set a job name
#SBATCH --job-name={job_name}
#################

# a file for job output, you can check job progress
#SBATCH --output={output}
#################

# a file for errors
#SBATCH --error={error}
#################

# time needed for job
#SBATCH --time={time}
#################

# gpus per node
#SBATCH --gres=gpu:{num_gpus}
#################

# number of requested nodes
#SBATCH --nodes={num_nodes}
#################

# slurm will send a signal this far out before it kills the job
{auto_submit}
#################

# Have SLURM send you an email when the job ends or fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={email}

# #task per node
#SBATCH --ntasks-per-node={ntasks_per_node}
#################

# #cpu per task/gpu
#SBATCH --cpus-per-task={cpus_per_task}
#################

# memory per cpu
#SBATCH --mem-per-cpu={mem_per_cpu}
#################

# extra stuff
{extra}


export PYTHONFAULTHANDLER=1
export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)

{module}

srun {main_cmd}
"""


def get_max_trial_version(path: str):
    files = os.listdir(path)
    version_files = [f for f in files if 'trial_' in f]
    if len(version_files) > 0:
        # regex out everything except file version for ve
        versions = [int(f_name.split('_')[1]) for f_name in version_files]
        max_version = max(versions)
        return max_version + 1
    else:
        return 0


def layout_path(params: DictConfig):
    # format the logging folder path
    slurm_out_path = os.path.join(params.log_root, params.job)

    # when err logging is enabled, build add the err logging folder
    err_path = os.path.join(slurm_out_path, 'slurm_err_logs')
    if not os.path.exists(err_path):
        os.makedirs(err_path)

    # when out logging is enabled, build add the out logging folder
    out_path = os.path.join(slurm_out_path, 'slurm_out_logs')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # place where slurm files log to
    slurm_files_log_path = os.path.join(slurm_out_path, 'slurm_scripts')
    if not os.path.exists(slurm_files_log_path):
        os.makedirs(slurm_files_log_path)
    return out_path, err_path, slurm_files_log_path


def get_argv() -> str:
    argv = sys.argv

    def _convert() -> str:
        for x in argv:
            if ("[" in x) or ("," in x):
                x = "'" + x + "'"
            yield x

    return " ".join(list(_convert()))


def run_cluster(cfg: DictConfig, fn_main: Callable,
                lt_system: pytorch_lightning.LightningModule):

    slurm_params = cfg.launcher
    if slurm_params.job is None:
        slurm_params.job = cfg.model_name

    if slurm_params.from_slurm:
        fn_main(cfg, lt_system)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
        extra = ""
        out_path, err_path, slurm_files_log_path = layout_path(slurm_params)
        trial_version = get_max_trial_version(out_path)

        slurm_params.time = str(slurm_params.time)
        if not ':' in slurm_params.time:
            slurm_params.time = f"{int(slurm_params.time):02d}:00:00"
        if slurm_params.auto is False:
            auto_walltime = ""
        else:
            auto_walltime = f"#SBATCH --signal=USR1@150"

        loaded_module = ''
        extra = ""

        if slurm_params.partition is not None:
            extra = f"#SBATCH --partition {slurm_params.partition} \n"
            raise ValueError("Do not specify partition in AIMOS")

        node = platform.processor()
        if node == "x86_64":
            PYTHON = "/gpfs/u/home/LLLD/LLLDashr/scratch/miniconda3x86_64/envs/fs_cdfsl/bin/python"
        elif node == "ppc64le":
            PYTHON = "/gpfs/u/home/LLLD/LLLDashr/scratch/miniconda3ppc64le/envs/fs_cdfsl/bin/python"
            # extra = extra + "#SBATCH --partition dcs,rpi\n"

        # extra = extra + "conda activate fs_cdfsl \n"  # FIXME check

        python_cmd = get_argv()

        full_command = f"{PYTHON} {python_cmd} launcher.from_slurm=true "
        outpath = os.path.join(out_path,
                               f'trial_{trial_version}_{timestamp}_%j.out')
        error = os.path.join(err_path,
                             f'trial_{trial_version}_{timestamp}_%j.err')
        cmd_to_sbatch = SLURM_CMD.format(
            job_name=slurm_params.job,
            output=outpath,
            error=error,
            time=slurm_params.time,
            num_gpus=slurm_params.gpus,
            num_nodes=slurm_params.nodes,
            auto_submit=auto_walltime,
            email=slurm_params.email,
            ntasks_per_node=slurm_params.gpus,
            cpus_per_task=slurm_params.cpus_per_task,
            mem_per_cpu=slurm_params.mem_per_cpu,
            extra=extra,
            module=loaded_module,
            main_cmd=full_command,
        )
        # print(Fore.LIGHTWHITE_EX + cmd_to_sbatch)
        script = "{}/{}.sh".format(slurm_files_log_path, slurm_params.job)
        with open(script, 'w') as f:
            print(cmd_to_sbatch, file=f, flush=True)

        p = subprocess.Popen(['sbatch', script],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        stdout, _ = p.communicate()
        stdout = stdout.decode("utf-8")
        job_id = stdout.split(" ")[-1].strip()
        print(f"Job {job_id} is submitted.")

        print("sbatch script: ", str(script))
        print(Fore.LIGHTGREEN_EX + f"stderr : ", error.replace("%j", job_id))
        print(Fore.LIGHTYELLOW_EX + f"stdout : ",
              outpath.replace("%j", job_id))
        print("-" * 60, "\n\n\n")
