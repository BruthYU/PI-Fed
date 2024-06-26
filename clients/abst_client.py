from typing import *
from torch import Tensor, nn, optim
from copy import deepcopy
from abc import abstractmethod
class AbstractClient:
    def __init__(self, device: str, list__ncls: List[int], inputsize: Tuple[int, ...],
                 lr: float, lr_factor: float, lr_min: float, epochs_max: int, patience_max: int,
                 lamb: float, eps: float,**kwargs):
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
        self.eps = eps

        # misc
        self.criterion = nn.CrossEntropyLoss()
        self.model = NotImplemented  # type: nn.Module


        #info
        self.loss = 0
        self.batch_num = 0



    def batch_train(self,x,y):
        raise NotImplementedError("Batch_train() is not implemented.")

    @abstractmethod
    def client_info(self):
        pass

    def compute_loss(self, output: Tensor, target: Tensor, misc: Dict[str, Any]) -> Tensor:
        reg = misc['reg']

        loss_all = self.criterion(output, target) + self.lamb * reg

        return loss_all
    def copy_model(self) -> Dict[str, Tensor]:
        return deepcopy(self.model.state_dict())


