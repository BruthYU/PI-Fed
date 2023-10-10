import hydra
import os
import pickle
import tempfile
from datetime import datetime
from typing import *
import hydra
import struct
import mlflow
import optuna
import torch
import copy
from omegaconf import DictConfig, OmegaConf
from dataloader import get_shuffled_dataloder
from utils import BColors, myprint as print, suggest_float, suggest_int
from optuna import Trial
import logging

from clients import *
def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10 ** digits)

    return uuid.uuid_value
LOG = logging.getLogger(__name__)





def load_dataloader(cfg: DictConfig) -> Dict[int, Dict[str, Any]]:
    basename_data = f'seq={cfg.seq.name}_bs={cfg.seq.batch_size}_seed={cfg.seed}'
    dirpath_data = os.path.join(hydra.utils.get_original_cwd(), 'data')
    # load data
    filepath_pkl = os.path.join(dirpath_data, f'{basename_data}.pkl')
    if os.path.exists(filepath_pkl):
        with open(filepath_pkl, 'rb') as f:
            dict__idx_task__dataloader = pickle.load(f)
        # endwith

        print(f'Loaded from {filepath_pkl}', bcolor=BColors.OKBLUE)
    else:
        dict__idx_task__dataloader = get_shuffled_dataloder(cfg)
        with open(filepath_pkl, 'wb') as f:
            pickle.dump(dict__idx_task__dataloader, f)

    # compute hash
    num_tasks = len(dict__idx_task__dataloader.keys())
    hash = []
    for idx_task in range(num_tasks):
        name = dict__idx_task__dataloader[idx_task]['fullname']
        ncls = dict__idx_task__dataloader[idx_task]['ncls']
        num_train = len(dict__idx_task__dataloader[idx_task]['train'].dataset)
        num_val = len(dict__idx_task__dataloader[idx_task]['val'].dataset)
        num_test = len(dict__idx_task__dataloader[idx_task]['test'].dataset)

        msg = f'idx_task: {idx_task}, name: {name}, ncls: {ncls}, num: {num_train}/{num_val}/{num_test}'
        hash.append(msg)
    # endfor
    hash = '\n'.join(hash)

    # check hash
    filepath_hash = os.path.join(dirpath_data, f'{basename_data}.txt')
    if os.path.exists(filepath_hash):
        with open(filepath_hash, 'rt') as f:
            hash_target = f.read()
        # endwith

        assert hash_target == hash

        print(f'Succesfully matched to {filepath_hash}', bcolor=BColors.OKBLUE)
        print(hash)
    else:
        # save hash
        with open(filepath_hash, 'wt') as f:
            f.write(hash)
    return dict__idx_task__dataloader


def train(cfg: DictConfig):
    dict__idx_task__dataloader = load_dataloader(cfg)
    num_tasks = len(dict__idx_task__dataloader.keys())
    list__name = [dict__idx_task__dataloader[idx_task]['name'] for idx_task in range(num_tasks)]
    list__ncls = [dict__idx_task__dataloader[idx_task]['ncls'] for idx_task in range(num_tasks)]
    inputsize = dict__idx_task__dataloader[0]['inputsize']  # type: Tuple[int, ...]

    task_train(dict__idx_task__dataloader, cfg.pi, None)
    pass

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    LOG.info(f'\n[CONIFG] : \n{OmegaConf.to_yaml(cfg)}')
    train(cfg)



if __name__ == '__main__':
    OmegaConf.register_new_resolver('uuid', lambda : uuid())
    main()