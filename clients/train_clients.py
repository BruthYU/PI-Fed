import copy
import clients.alg_client as alg_client
import torch
from omegaconf import DictConfig, OmegaConf
from typing import *
from torch import Tensor, nn, optim
def switch_tag(num_clients, num_batches):
    tags = [0]
    for client_id in range(num_clients):
        tags.append(tags[-1] + num_batches/num_clients -1)
    tags[-1] += num_batches % num_clients
    return tags[1:-1]



def fed_task_train(task_id, task__dataloader: dict, cfg: DictConfig, client_cfg: Dict,root_state_dict):
    num_clients = cfg.fed.num_clients
    assert num_clients > 0 , "At least one client is required."
    tags = switch_tag(num_clients,len(task__dataloader))
    client_models = []
    client_losses = []

    # Initialization
    client = getattr(alg_client, cfg.fed.alg)
    clients = [client(client_cfg,root_state_dict)] * num_clients


    # federated task train
    for client_id in range(num_clients):
        clients[client_id].model.train()

    for epoch in range(client_cfg['epoch_max']):
        client_id = 0
        for idx_batch, (x, y) in enumerate(task__dataloader):
            client[client_id].batch_train(x,y)
            if idx_batch in tags:
                client_id += 1

    for client_id in range(num_clients):
        info = clients[client_id].client_info()
        client_models.append(info['model'])
        client_losses.append(info['loss'])

    return client_models, client_losses




