from typing import *
from torch import Tensor, nn, optim
from copy import deepcopy
import torch
class AbstractServer:
    def __init__(self, device: str, idx_task: int, lamb: float,**kwargs):
        self.device = device
        self.idx_task = idx_task
        self.lamb = lamb


    '''
    Get a copy of the averaged state dict before calibration
    '''
    def average_weights(self, client_models):
        weights_avg = deepcopy(client_models[0])

        for key in weights_avg.keys():
            for i in range(1, len(client_models)):
                weights_avg[key] += client_models[i][key]
            weights_avg[key] = torch.div(weights_avg[key], len(client_models))

        return weights_avg
    def average_loss(self, list_losses):
        return sum(list_losses)/len(list_losses)


# def __init__(self, device: str, list__ncls: List[int], inputsize: Tuple[int, ...],
#              lr: float, lr_factor: float, lr_min: float, epochs_max: int, patience_max: int,
#              lamb: float, **kwargs):

