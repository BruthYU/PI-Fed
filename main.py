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
from models import *
from fed_trainer import *
from server import Eval_PI_Fed
def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10 ** digits)

    return uuid.uuid_value
LOG = logging.getLogger(__name__)



'''
seq={dataset_name}_{subtask_num}_bs={batchsize}_seed={seed}.pkl
'''

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


def continual_fed_train(cfg: DictConfig):
    print(f'device: {cfg.device}', bcolor=BColors.OKBLUE)
    # load dataset
    dict__idx_task__dataloader = load_dataloader(cfg)
    num_tasks = len(dict__idx_task__dataloader.keys())
    list__name = [dict__idx_task__dataloader[idx_task]['name'] for idx_task in range(num_tasks)]
    list__ncls = [dict__idx_task__dataloader[idx_task]['ncls'] for idx_task in range(num_tasks)]
    inputsize = dict__idx_task__dataloader[0]['inputsize']  # type: Tuple[int, ...]

    # load model
    root_state_dict = None
    root_mask = None
    client_cfg = None
    if cfg.fed.task == 'img_cls':
        client_cfg = {
        'device': cfg.device,
        'list__ncls': list__ncls,
        'inputsize': inputsize,
        'lr': cfg.lr,
        'lr_factor': cfg.lr_factor,
        'lr_min': cfg.lr_min,
        'epochs_max': cfg.epochs_max,
        'patience_max': cfg.patience_max,
        'backbone': cfg.backbone.name,
        'nhid': cfg.nhid,
        'idx_task': 0,
        'epochs_client': cfg.epochs_client,
        'lamb': 0,
        'eps': cfg.eps
        }
    task_name = cfg.seq.name
    drop_cfg = cfg.appr.tuned[task_name]
    if client_cfg['backbone'] in ['alexnet']:
        client_cfg['drop1'] = drop_cfg.drop1
        client_cfg['drop2'] = drop_cfg.drop2


    eval_server = Eval_PI_Fed(client_args=client_cfg)

    '''
    For every subtask, the trainer will assign num_client[task_id] clients
    '''
    for task_id in range(num_tasks):
        LOG.info(f'------------[Train On Task {task_id}]----------------')
        task_dataloader = dict__idx_task__dataloader[task_id]['train']
        client_cfg['idx_task'] = task_id
        trainer = fed_task_train(task_dataloader, cfg, client_cfg, root_state_dict, root_mask)
        trainer.train()
        root_state_dict, root_mask = trainer.server.avg_state_dict, trainer.server.mask
        eval_server.set_status(root_state_dict, task_id)
        avg_acc = []
        for t_prev in range(task_id + 1):
            results_test = eval_server.test(t_prev,dict__idx_task__dataloader[t_prev]['test'])
            loss_test = results_test['loss_test']
            acc_test = results_test['acc_test']
            avg_acc.append(acc_test)
            LOG.info(f'Test task {t_prev} | loss_test: {loss_test} | acc_test: {acc_test}')
        LOG.info(f'[Task learned : {task_id+1}] [Current average acc: {sum(avg_acc)/(task_id + 1)}]')


# TODO scheduler

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    LOG.info(f'\n[CONIFG] : \n{OmegaConf.to_yaml(cfg)}')
    continual_fed_train(cfg)



if __name__ == '__main__':
    OmegaConf.register_new_resolver('uuid', lambda : uuid())
    main()