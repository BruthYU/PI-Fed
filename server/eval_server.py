import numpy as np
from copy import deepcopy
from server.abst_server import AbstractServer
from typing import *
from models import *
from torch import Tensor, nn, optim
import utils

'''
Latest Accuracy, Average Accuracy
'''

class Eval(AbstractServer):
    def __init__(self,client_args: Dict[str, Any]):
        super().__init__(**client_args)
        self.criterion = nn.CrossEntropyLoss()
        self.model = ModelSPG(**client_args).to(self.device)
        self.root_state_dict = None

    def set_status(self, root_state_dict, idx_task_learned):
        self.root_state_dict = root_state_dict
        self.idx_task = idx_task_learned
        self.model.load_state_dict(root_state_dict)


    def test(self, t_prev, dl_test):
        assert self.root_state_dict is not None, "root_state_dict is not loaded !"
        results_test = self._eval_commom(t_prev, dl_test)

        results = {
            'idx_task_learned':self.idx_task,
            'idx_task_tested_now': t_prev,
            'loss_test': results_test['loss'],
            'acc_test': results_test['acc'],
            }

        return results
    def _eval_commom(self, idx_task, dl_test: DataLoader) -> Dict[str, float]:
        self.model.eval()
        list__target, list__output = [], []
        loss = 0  # type: Tensor

        args = {
            'idx_task': idx_task,
        }

        with torch.no_grad():
            for idx_batch, (x, y) in enumerate(dl_test):
                x = x.to(self.device)
                y = y.to(self.device)

                output, misc = self.model(x, args=args)
                loss += self.compute_loss(output=output, target=y, misc=misc)
                list__target.append(y)
                list__output.append(output)
            # endfor
        # endwith

        acc = utils.my_accuracy(torch.cat(list__target, dim=0),
                                torch.cat(list__output, dim=0)).item()

        results = {
            'loss': loss.item(),
            'acc': acc,
        }

        return results

    def compute_loss(self, output: Tensor, target: Tensor, misc: Dict[str, Any]) -> Tensor:
        reg = misc['reg']

        loss_all = self.criterion(output, target) + self.lamb * reg

        return loss_all

