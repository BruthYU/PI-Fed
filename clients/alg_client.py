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

        self.metric = {
            'patience': 0,
            'loss_val_best': np.inf,
            'acc_val_best': -np.inf,
            'loss_train_best': np.inf,
            'acc_train_best': -np.inf,
            'state_dict_best': self.copy_model(),
            'epoch_best': 0,
        }

    # Use AbstractClient Methods
    def batch_train(self,x,y):
        self.batch_num += 1
        x = x.to(self.device)
        y = y.to(self.device)



class FedAvg(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)

    def batch_train(self, x, y):
        pass