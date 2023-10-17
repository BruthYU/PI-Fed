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

class fed_task_train():
    def __init__(self, task__dataloader: dict, cfg: DictConfig, client_cfg: Dict, root_state_dict = None, root_mask = None):
        self.task_id = client_cfg['task_id']
        self.task__dataloader = task__dataloader
        self.cfg = cfg
        self.client_cfg = client_cfg
        self.root_state_dict = root_state_dict


        self.num_clients = cfg.fed.num_clients[self.task_id]
        assert self.num_clients > 0, "At least one client is required."
        self.tags = self.switch_tag(self.num_clients, len(self.task__dataloader))

        server = getattr(alg_server,cfg.fed.alg)
        self.server = server(client_cfg,root_state_dict,root_mask)


        # Initialization
        client = getattr(alg_client, cfg.fed.alg)
        if cfg.fed.alg == 'PI_Fed':
            self.clients = [client(client_cfg, root_state_dict=self.server.avg_state_dict, root_mask = self.server.avg_mask)] * self.num_clients




    def switch_tag(self, num_clients, num_batches):
        tags = [0]
        for client_id in range(num_clients):
            tags.append(tags[-1] + num_batches / num_clients - 1)
        tags[-1] += num_batches % num_clients
        return tags[1:-1]

    def client_epoch_reset(self):
        for i in range(self.num_clients):
            self.clients[i].client_epoch_reset(self.server.avg_state_dict)

    def train(self):
        for client_id in range(self.num_clients):
            self.clients[client_id].model.train()
        for epoch in range(self.client_cfg['epoch_max']):
            self.client_epoch_reset()
            self.train_epoch()
            self.sever_epoch()

        list_client_module_mask = self.clients_complete_learning()
        self.server.average_mask(list_client_module_mask)




    def train_epoch(self):
        for epoch in self.client_cfg['client_epochs']:
            client_id = 0
            for idx_batch, (x, y) in enumerate(self.task__dataloader):
                self.clients[client_id].batch_train(x, y)
                if idx_batch in self.tags:
                    client_id += 1



    def sever_epoch(self):
        client_models = []
        client_losses = []
        for client_id in range(self.num_clients):
            info = self.clients[client_id].client_info()
            client_models.append(info['model'])
            client_losses.append(info['loss'])
        self.server.avg_state_dict = self.server.average_weights(client_models)



    def clients_complete_learning(self):
        for i in range(self.num_clients):
            self.clients[i].pre_complete_learning()
        range_tasks = range(self.task_id + 1)
        for t in range_tasks:
            client_id = 0
            for idx_batch, (x, y) in enumerate(self.task__dataloader):
                self.clients[client_id].batch_complete(x, y, t)
                if idx_batch in self.tags:
                    client_id += 1
        list_client_module_mask = []
        for i in range(self.num_clients):
            list_client_module_mask.append(self.clients[i].post_complete_learning())
        return list_client_module_mask










