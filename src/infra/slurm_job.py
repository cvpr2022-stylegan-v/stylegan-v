#!/path/to/python
"""
Must be launched from the released project dir
"""

import os;
import time
import random
import subprocess
from shutil import copyfile

import hydra
from omegaconf import DictConfig, OmegaConf

# Unfortunately, (AFAIK) we cannot pass arguments normally (to parse them with argparse)
# that's why we are reading them from env
SLURM_JOB_ID = os.getenv('SLURM_JOB_ID')
project_dir = os.getenv('project_dir')

# Printing the environment
print('PROJECT DIR:', project_dir)
print(f'SLURM_JOB_ID: {SLURM_JOB_ID}')
print('HOSTNAME:', subprocess.run(['hostname'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

@hydra.main(config_name=os.path.join(project_dir, 'experiment_config.yaml'))
def main(cfg: DictConfig):
    os.chdir(project_dir)

    target_data_dir_base = os.path.dirname(cfg.dataset.target_path)
    if os.path.islink(target_data_dir_base):
        os.makedirs(os.readlink(target_data_dir_base), exist_ok=True)
    else:
        os.makedirs(target_data_dir_base, exist_ok=True)

    copyfile(cfg.dataset.source_path, cfg.dataset.target_path)
    print('Data has been copied! Starting the training...')

    training_cmd = open('training_cmd.sh').read()
    print('<=== TRAINING COMMAND ===>')
    print(training_cmd)
    os.system(training_cmd)


if __name__ == "__main__":
    main()
