from typing import *
from torch import Tensor, nn, optim
from copy import deepcopy
import torch
class AbstractServer:
    def __init__(self, device: str, idx_task: int, **kwargs):
        self.device = device
        self.idx_task = idx_task


        # dataloader
        # self.list__ncls = list__ncls
        # self.inputsize = inputsize
        #
        # # variables
        # self.lr = lr
        # self.lr_factor = lr_factor
        # self.lr_min = lr_min
        # self.epochs_max = epochs_max
        # self.patience_max = patience_max
        # self.lamb = lamb

    def average_weights(self, client_models):
        weights_avg = deepcopy(client_models[0])

        for key in weights_avg.keys():
            for i in range(1, len(client_models)):
                weights_avg[key] += client_models[i][key]
            weights_avg[key] = torch.div(weights_avg[key], len(client_models))

        return weights_avg


# def __init__(self, device: str, list__ncls: List[int], inputsize: Tuple[int, ...],
#              lr: float, lr_factor: float, lr_min: float, epochs_max: int, patience_max: int,
#              lamb: float, **kwargs):

