import copy
import clients.alg_client as alg_client
import server.alg_server as alg_server

import torch
from omegaconf import DictConfig, OmegaConf
from typing import *
import numpy as np
import time
from torch import Tensor, nn, optim
import utils
from utils import my_accuracy
from copy import deepcopy
import logging
LOG = logging.getLogger(__name__)
class fed_task_train():
    def __init__(self, task__dataloader: dict, cfg: DictConfig, client_cfg: Dict, root_state_dict = None, root_mask = None):
        self.task_id = client_cfg['idx_task']
        self.task__dataloader = task__dataloader
        self.cfg = cfg
        self.client_cfg = client_cfg
        self.root_state_dict = root_state_dict


        self.num_clients = cfg.fed.num_clients[self.task_id]
        assert self.num_clients > 0, "At least one client is required."
        self.tags = self.switch_tag(self.num_clients, len(task__dataloader))


        server = getattr(alg_server, cfg.fed.alg)
        if cfg.fed.alg == 'PI_Fed':
            self.server = server(client_cfg, root_state_dict, root_mask)
        else:
            self.server = server(client_cfg, root_state_dict)


        # Initialization
        client = getattr(alg_client, cfg.fed.alg)
        if cfg.fed.alg == 'PI_Fed':
            self.clients = [client(client_cfg, root_state_dict=self.server.avg_state_dict,
                                   root_mask = self.server.mask) for i in range(self.num_clients)]
        else:
            self.clients = [client(client_cfg, root_state_dict=self.server.avg_state_dict) for i in range(self.num_clients)]




    '''
    switch client based on num_batches during the training of a subtask (per epoch)
    '''
    def switch_tag(self, num_clients, num_batches):
        assert num_clients < num_batches, "Exception: num clients > num_batches !"
        tags = []
        batch_per_client = num_batches // num_clients
        for client_id in range(num_clients):
            tags.append((client_id + 1) * batch_per_client)
        tags[-1] += num_batches % num_clients
        return tags

    def client_epoch_reset(self):
        for i in range(self.num_clients):
            self.clients[i].client_epoch_reset(self.server.avg_state_dict)

    # Global epochs = aggregation times
    def train(self):
        for client_id in range(self.num_clients):
            self.clients[client_id].model.train()
        for epoch in range(self.client_cfg['epochs_max']):
            self.client_epoch_reset()
            self.train_epoch()
            # Synchronize the global weight
            self.sever_epoch()
            LOG.info(f'Epoch {epoch}, Average Loss: {self.server.avg_loss}')

        if self.cfg.fed.alg == 'PI_Fed':
            list_client_module_mask = self.clients_complete_learning()
            self.server.aggregate_mask(list_client_module_mask)



    # Local epochs for clients
    def train_epoch(self):
        for c_epoch in range(self.client_cfg['epochs_client']):
            client_id = 0
            for idx_batch, (x, y) in enumerate(self.task__dataloader):
                if idx_batch in self.tags:
                    client_id += 1
                self.clients[client_id].batch_train(x, y)


    def sever_epoch(self):
        info = []
        for client_id in range(self.num_clients):
            info.append(self.clients[client_id].client_info())
        self.server.compute_global_weight(info)




    '''
    Operations after the training of each subtask 
    '''
    def clients_complete_learning(self):

        range_tasks = range(self.task_id + 1)
        for t in range_tasks:
            for client in self.clients:
               # zero_grad()
               client.pre_complete_learning()
            client_id = 0

            # one more backward pass (only for importance computation, not for weight update)
            for idx_batch, (x, y) in enumerate(self.task__dataloader):
                '''
                Key function of importance computation
                '''
                self.clients[client_id].batch_complete(x, y, t)
                if idx_batch in self.tags:
                    client_id += 1
            for client in self.clients:
                client.model_register_grad(t)

        list_client_module_mask = []
        for i in range(self.num_clients):
            list_client_module_mask.append(self.clients[i].post_complete_learning())
        return list_client_module_mask










