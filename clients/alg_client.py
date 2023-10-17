import numpy as np

from clients.abst_client import AbstractClient
from typing import *
from models import *
from torch import Tensor, nn, optim
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                         mode='min',
                                                         factor=1.0 / self.lr_factor,
                                                         patience=max(self.patience_max - 1, 0),
                                                         min_lr=self.lr_min,
                                                         verbose=True,
                                                         )
        self.list__target_train = []
        self.list__output_train = []



    # Use AbstractClient Methods
    def batch_train(self,x,y):
        self.batch_num += 1
        x = x.to(self.device)
        y = y.to(self.device)
        args = {'idx_task': self.client_args['idx_task']}
        output, misc = self.model(x, args=args)
        loss = self.compute_loss(output=output, target=y, misc=misc)
        self.loss += loss
        self.batch_num += 1
        self.list__target_train.append(y)
        self.list__output_train.append(output)

        # optim
        self.optimizer.zero_grad()
        loss.backward()
        self.modify_grads(args)
        self.optimizer.step()

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
        self.model.modify_grad(args= args)


    def client_epoch_reset(self, avg_state_dict):
        self.list__target_train = []
        self.list__output_train = []
        self.loss = 0
        self.batch_num = 0
        if avg_state_dict is not None:
            self.model.load_state_dict(avg_state_dict)
        if self.loss != 0 :
            self.scheduler.step()


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











class FedAvg(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)

    def batch_train(self, x, y):
        pass