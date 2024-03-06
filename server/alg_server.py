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
        self.mask = root_mask
        self.avg_loss = 0


    def standardize_pm1(self, x: Tensor) -> Tensor:
        if torch.all(x == 0):
            pass
        else:
            x = self.standardize(x)
        # endif
        ret = torch.tanh(x)

        return ret

    def standardize(cls, x: Tensor) -> Tensor:
        sh = x.shape
        x = x.view(-1)

        ret = (x - x.mean()) / x.std()

        return ret.view(*sh)

    def mask_mean(self):
        value = []
        for key in self.mask.keys():
            mask_dict = self.mask[key]
            for weight_key in mask_dict[self.idx_task - 1].keys():
                value.append(mask_dict[self.idx_task - 1][weight_key].view(-1))
        cat_value = torch.cat(value,dim=0)
        mean_value = cat_value.mean()
        return mean_value

    def modify_state_dict(self, new_state_dict):
        if self.mask is None:
            self.avg_state_dict = new_state_dict
        else:
            modification = 1 - self.mask_mean()

            for key in new_state_dict.keys():
                diff = self.avg_state_dict[key] - new_state_dict[key]
                if 'target_module' in key:
                    [prefix, suffix] = key.split('.target_module.')
                    diff = diff * (1 - self.mask[prefix][self.idx_task-1][suffix]).cuda()
                if 'classifier' in key:
                    diff = diff * modification
                self.avg_state_dict[key] = self.avg_state_dict[key]- diff




    '''
    list_client_module_mask = self.clients_complete_learning()
    fed_train.py -> self.train()
    '''
    def aggregate_mask(self, list_client_module_mask):
        mask = deepcopy(list_client_module_mask[0])
        for module_name, history_mask in mask.items():
            for name in history_mask[self.idx_task].keys():
                for i in range(1,len(list_client_module_mask)):
                    mask[module_name][self.idx_task][name] += list_client_module_mask[i][module_name][self.idx_task][name]
                v = self.standardize_pm1(mask[module_name][self.idx_task][name]).abs()
                if self.idx_task > 0:
                    v = torch.max(v,self.mask[module_name][self.idx_task-1][name])
                mask[module_name][self.idx_task][name] = v
        self.mask = mask










