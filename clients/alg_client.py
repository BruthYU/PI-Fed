import numpy as np

from clients.abst_client import AbstractClient
from typing import *
from models import *
from torch import Tensor, nn, optim
class PI_Fed(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        self.model = ModelSPG(**client_args).to(self.device)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)
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
        output, misc = self.model(x)
        loss = self.compute_loss(output=output, target=y, misc=misc)
        self.loss += loss
        self.batch_num += 1
        self.list__target_train.append(y)
        self.list__output_train.append(output)

        # optim
        self.optimizer.zero_grad()
        loss.backward()
        self.modify_grads()
        self.optimizer.step()

    def modify_grads(self):
        self.model.modify_grad()


    def client_epoch_reset(self, avg_state_dict):
        self.list__target_train = []
        self.list__output_train = []
        self.loss = 0
        self.batch_num = 0
        if avg_state_dict is not None:
            self.model.load_state_dict(avg_state_dict)

    def complete_learning(self, idx_task,):
        def complete_learning(self, idx_task: int, **kwargs) -> None:
            dl = kwargs['dl_train']

            self.model.compute_importance(idx_task=idx_task, dl=dl)








class FedAvg(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)

    def batch_train(self, x, y):
        pass