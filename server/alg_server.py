import numpy as np
from copy import deepcopy
from server.abst_server import AbstractServer
from typing import *
from models import *
from torch import Tensor, nn, optim
class PI_Fed(AbstractServer):
    def __init__(self, client_args: Dict[str, Any], root_state_dict = None, root_mask = None):
        super().__init__(**client_args)

        self.avg_state_dict = root_state_dict
        self.avg_mask = root_mask
        self.avg_loss = 0


    def average_mask(self, list_client_module_mask):
        avg_mask = deepcopy(list_client_module_mask[0])

        for module_name, history_mask in avg_mask.items():
            for name in history_mask[self.idx_task].keys():
                for i in range(1,len(list_client_module_mask)):
                    avg_mask[module_name][self.idx_task][name] += list_client_module_mask[i][module_name][self.idx_task][name]
                avg_mask[module_name][self.idx_task][name] = torch.div(avg_mask[module_name][self.idx_task][name],len(list_client_module_mask))
        self.avg_mask = avg_mask









