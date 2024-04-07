import copy

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

    def compute_global_weight(self, info: list):
        client_models = []
        client_losses = []
        for info_item in info:
            client_models.append(info_item['model'])
            client_losses.append(info_item['loss'])
        self.avg_loss = self.average_loss(client_losses)
        new_state_dict = self.average_weights(client_models)
        self.modify_state_dict(new_state_dict)

class FedAvg(AbstractServer):
    def __init__(self, client_args: Dict[str, Any], root_state_dict = None):
        super().__init__(**client_args)
        self.avg_state_dict = root_state_dict
        self.avg_loss = 0
    def compute_global_weight(self, info: list):
        client_models = []
        client_losses = []
        for info_item in info:
            client_models.append(info_item['model'])
            client_losses.append(info_item['loss'])
        self.avg_loss = self.average_loss(client_losses)
        self.avg_state_dict = self.average_weights(client_models)

class FedNova(AbstractServer):
    def __init__(self, client_args: Dict[str, Any], root_state_dict = None):
        super().__init__(**client_args)
        self.avg_state_dict = root_state_dict
        self.avg_loss = 0

    def nova_average(self, client_models, client_coeffs):
        norm_grads = []
        for model, coeff in zip(client_models, client_coeffs):
            norm_grad = copy.deepcopy(self.avg_state_dict)
            for key in model:
                norm_grad[key] = torch.div(self.avg_state_dict[key]-model[key], coeff)
            norm_grads.append(norm_grad)

        nova_grad = copy.deepcopy(self.avg_state_dict)
        for i, norm_grad in enumerate(norm_grads):
            for key in norm_grad:
                if i == 0:
                    nova_grad[key] = norm_grad[key]/len(client_models)
                else:
                    nova_grad[key] = nova_grad[key] + norm_grad[key]/len(client_models)

        avg_coeff = sum(client_coeffs)/len(client_models)
        for key in self.avg_state_dict:
            self.avg_state_dict[key] -= avg_coeff * nova_grad[key]



    def compute_global_weight(self, info: list):
        client_models = []
        client_losses = []
        client_coeffs = []
        for info_item in info:
            client_models.append(info_item['model'])
            client_losses.append(info_item['loss'])
            client_coeffs.append(info_item['coeff'])
        self.avg_loss = self.average_loss(client_losses)
        self.nova_average(client_models, client_coeffs)

#TODO

class FedSCAFFOLD(AbstractServer):
    def __init__(self, client_args: Dict[str, Any], root_state_dict = None):
        super().__init__(**client_args)
        self.avg_loss = 0
        self.avg_state_dict = root_state_dict

        # server control variate
        self.scv = ModelSPG(**client_args).to(self.device)
        self.scv.load_state_dict(root_state_dict)
        self.scv_state = self.scv.state_dict()

        #



    def scaffold_agg(self, client_state_dicts, client_new_ccvs):
        new_avg_state = copy.deepcopy(self.avg_state_dict)
        new_scv_state = copy.deepcopy(self.avg_state_dict)
        for i, (model, client_ccv) in enumerate(zip(client_state_dicts, client_new_ccvs)):
            for key in model:
                if i == 0:
                    new_scv_state[key] = model[key] / len(client_state_dicts)
                    new_scv_state[key] = client_ccv[key] / len(client_state_dicts)
                else:
                    new_avg_state[key] = new_avg_state[key] + model[key] / len(client_state_dicts)
                    new_scv_state[key] = new_scv_state + client_ccv[key] / len(client_state_dicts)
        self.scv_state = self.scv.state_dict()
        self.scv.load_state_dict(new_scv_state)
        self.avg_state_dict = new_avg_state






    def compute_global_weight(self, info: list):
        client_state_dicts = []
        client_losses = []
        client_new_ccvs = []
        for info_item in info:
            client_state_dicts.append(info_item['state_dict'])
            client_losses.append(info_item['loss'])
            client_new_ccvs.append(info_item['new_ccv_state'])
        self.avg_loss = self.average_loss(client_losses)
        new_avg_state, new_scv_state  = self.scaffold_agg(client_state_dicts,client_new_ccvs)
        return

















