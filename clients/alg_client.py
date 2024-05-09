import numpy as np

from clients.abst_client import AbstractClient
from typing import *
from models import *
from torch import Tensor, nn, optim
from torch.nn.utils import clip_grad_norm_
import copy
import torchinfo
'''
client_info() --> rec() in https://github.com/rruisong/pytorch_federated_learning
'''
class PI_Fed(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict = None, root_mask = None):
        super().__init__(**client_args)
        self.client_args = client_args
        self.model = ModelSPG(**client_args).to(self.device)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)
        if root_mask is not None:
            self.model.load_mask(root_mask)


        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)




        # self.list__target_train = []
        # self.list__output_train = []


    # Use AbstractClient Methods
    def batch_train(self,x,y):
        self.batch_num += 1
        x = x.to(self.device)
        y = y.to(self.device)
        args = {'idx_task': self.client_args['idx_task']}
        output, misc = self.model(x, args=args)
        loss = self.compute_loss(output=output, target=y, misc=misc)


        self.loss += loss


        # optim
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(),max_norm=2.,norm_type=2)

        self.optimizer.step()


    '''
    '''
    def batch_complete(self, x, y, t):
        idx_task = self.client_args['idx_task']
        args = {'idx_task': t}
        x = x.to(self.device)
        y = y.to(self.device)
        out, _ = self.model.__call__(x, args=args)
        if t == idx_task:
            lossfunc = nn.CrossEntropyLoss()
        else:
            lossfunc = OtherTasksLoss()
        loss = lossfunc(out, y)
        loss.backward()
        #clip_grad_norm_(self.model.parameters(), max_norm=2., norm_type=2)

    def model_register_grad(self, t):
        idx_task = self.client_args['idx_task']
        for name_module, module in self.model.named_modules():
            if isinstance(module,SPG):
                grads = {}

                for name_param, param in module.target_module.named_parameters():
                    if param.grad is not None:
                        grad = param.grad.data.clone().cpu()
                    else:
                        grad = 0
                    # endif

                    if name_param not in grads.keys():
                        grads[name_param] = 0
                    # endif

                    grads[name_param] += grad
                module.register_grad(idx_task=idx_task, t=t, grads=grads)




    def modify_grads(self, args):
        self.model.modify_grads(args= args)


    def client_epoch_reset(self, avg_state_dict):
        self.loss = 0
        self.batch_num = 0
        if avg_state_dict is not None:
            self.model.load_state_dict(avg_state_dict)



    def pre_complete_learning(self) -> None:
        self.model.zero_grad()




    def post_complete_learning(self) -> Dict[str, Dict]:

        dict_module_mask = {}

        for name, module in self.model.named_modules():
            if isinstance(module, SPG):
                module.compute_mask(idx_task=self.client_args['idx_task'])
                dict_module_mask[name] = module.history_mask
                # [module_name][idx_task][param_name][mask_tensor]

        return dict_module_mask

    def client_info(self):
        # mem_info =  torchinfo.summary(self.model)
        info = {'model': self.model.state_dict(), 'loss': self.loss / self.batch_num}
        return info




class FedAvg(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        # Importance Computation will not be used
        self.model = ModelSPG(**client_args).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)
        self.client_args = client_args

    def batch_train(self, x, y):
        self.batch_num += 1
        x = x.to(self.device)
        y = y.to(self.device)
        args = {'idx_task': self.client_args['idx_task']}
        output, misc = self.model(x, args=args)
        loss = self.compute_loss(output=output, target=y, misc=misc)
        # print(f'batch_num: {self.batch_num}, loss: {loss}')
        self.loss += loss


        # optim
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=2., norm_type=2)
        self.optimizer.step()

    def client_epoch_reset(self, avg_state_dict):
        self.loss = 0
        self.batch_num = 0
        if avg_state_dict is not None:
            self.model.load_state_dict(avg_state_dict)

    def client_info(self):
        info = {'model': self.model.state_dict(), 'loss': self.loss / self.batch_num}
        return info

class FedNova(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        # Importance Computation will not be used
        self.model = ModelSPG(**client_args).to(self.device)

        #Nova
        self.rho = 0.9
        self._momentum = self.rho
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr,momentum=self._momentum)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)
        self.client_args = client_args


    def batch_train(self, x, y):
        self.batch_num += 1
        x = x.to(self.device)
        y = y.to(self.device)
        args = {'idx_task': self.client_args['idx_task']}
        output, misc = self.model(x, args=args)
        loss = self.compute_loss(output=output, target=y, misc=misc)
        # print(f'batch_num: {self.batch_num}, loss: {loss}')
        self.loss += loss


        # optim
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=2., norm_type=2)
        self.optimizer.step()

    def client_epoch_reset(self, avg_state_dict):
        self.loss = 0
        self.batch_num = 0
        if avg_state_dict is not None:
            self.model.load_state_dict(avg_state_dict)

    def client_info(self):
        tau = self.batch_num
        coeff = (tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)) / (1 - self.rho)
        info = {'model': self.model.state_dict(), 'loss': self.loss / self.batch_num, 'coeff':coeff}
        return info


class SCAFFOLD(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        assert root_state_dict is not None, "root_state_dict is required (None)"
        # Importance Computation will not be used
        self.model = ModelSPG(**client_args).to(self.device)
        self.model.load_state_dict(root_state_dict)
        # server control variate
        self.scv = ModelSPG(**client_args).to(self.device)
        self.scv.load_state_dict(root_state_dict)
        self.scv_state_dict = self.scv.state_dict()
        # client control variate
        self.ccv = ModelSPG(**client_args).to(self.device)
        self.ccv.load_state_dict(root_state_dict)
        self.ccv_state_dict = self.ccv.state_dict()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        self.client_args = client_args

    def client_epoch_reset(self, model_state_dict, scv_state):
        """
        SCAFFOLD client updates local models and server control variate
        :param model_state_dict:
        :param scv_state:
        """
        self.loss = 0
        self.batch_num = 0
        self.scv.load_state_dict(scv_state)
        self.scv_state_dict = self.scv.state_dict()
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)


    def batch_train(self, x, y):
        self.batch_num += 1
        x = x.to(self.device)
        y = y.to(self.device)
        args = {'idx_task': self.client_args['idx_task']}
        output, misc = self.model(x, args=args)
        loss = self.compute_loss(output=output, target=y, misc=misc)
        # print(f'batch_num: {self.batch_num}, loss: {loss}')
        self.loss += loss


        # optim
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=2., norm_type=2)
        self.optimizer.step()

        state_dict = self.model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key] - self.lr * (self.scv_state_dict[key] - self.ccv_state_dict[key])
        self.model.load_state_dict(state_dict)



    def client_info(self):
        delta_model_state = copy.deepcopy(self.model.state_dict())
        global_state_dict = copy.deepcopy(self.model.state_dict())
        new_ccv_state = copy.deepcopy(self.ccv.state_dict())
        delta_ccv_state = copy.deepcopy(new_ccv_state)
        state_dict = self.model.state_dict()
        for key in state_dict:
            new_ccv_state[key] = self.ccv_state_dict[key] - self.scv_state_dict[key] + (global_state_dict[key] - state_dict[key]) / (self.batch_num * self.lr)
            delta_ccv_state[key] = new_ccv_state[key] - self.ccv_state_dict[key]
            delta_model_state[key] = state_dict[key] - global_state_dict[key]
        self.ccv.load_state_dict(new_ccv_state)
        self.ccv_state_dict = self.ccv.state_dict()
        info = {'state_dict': state_dict, 'loss': self.loss / self.batch_num, 'new_ccv_state':new_ccv_state}
        return info


