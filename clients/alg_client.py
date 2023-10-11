from clients.abst_client import AbstractClient
from typing import *
from models import *
class PI_Fed(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        self.model = ModelSPG(**client_args).to(self.device)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)
    def batch_train(self,x,y):
        pass

class FedAvg(AbstractClient):
    def __init__(self, client_args: Dict[str, Any], root_state_dict):
        super().__init__(**client_args)
        if root_state_dict is not None:
            self.model.load_state_dict(root_state_dict)

    def batch_train(self, x, y):
        pass