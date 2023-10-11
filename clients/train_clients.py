import copy
import clients.alg_client as alg_client
import torch
from omegaconf import DictConfig, OmegaConf

def switch_tag(num_clients, num_batches):
    tags = [0]
    for client_id in range(num_clients):
        tags.append(tags[-1] + num_batches/num_clients -1)
    tags[-1] += num_batches % num_clients
    return tags[1:-1]



def task_train(task__dataloader: dict, task_cfg: DictConfig, root_model):
    num_clients = task_cfg.num_clients
    assert num_clients > 0 , "At least one client is required."
    tags = switch_tag(num_clients,len(task__dataloader))

    client = getattr(alg_client,task_cfg.alg)

    current_client = client(root_model)
    client_models = []
    client_losses = []

    for idx_batch, (x, y) in enumerate(task__dataloader):
        client.batch_train(x,y)
        if idx_batch in tags:
            info = current_client.client_info()
            client_models.append(info['model'].state_dict())
            client_losses.append(info['loss'])
            current_client = client(root_model)
    return client_models, client_losses




    client_model = None
    for idx in range(num_clients):
        node = client(root_model)


