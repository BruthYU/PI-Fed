from typing import *
from torch import Tensor, nn, optim
class AbstractClient:
    def __init__(self, device: str, list__ncls: List[int], inputsize: Tuple[int, ...],
                 lr: float, lr_factor: float, lr_min: float, epochs_max: int, patience_max: int,
                 lamb: float, **kwargs):
        self.device = device

        # dataloader
        self.list__ncls = list__ncls
        self.inputsize = inputsize

        # variables
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.epochs_max = epochs_max
        self.patience_max = patience_max
        self.lamb = lamb

        # misc
        self.criterion = nn.CrossEntropyLoss()
        self.model = NotImplemented  # type: nn.Module

    def batch_train(self,x,y):
        raise NotImplementedError("Batch_train() is not implemented.")

    def client_info(self):
        info = {'model': self.model.state_dict(), 'loss':self.loss/self.batch_num}
        return info
