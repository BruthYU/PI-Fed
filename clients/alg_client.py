from clients.abst_client import AbstractClient

class PI_Fed(AbstractClient):
    def __init__(self, root_model):
        super().__init__(root_model)
    def batch_train(self,x,y):
        pass

class FedAvg(AbstractClient):
    def __init__(self, root_model):
        super().__init__(root_model)

    def batch_train(self, x, y):
        pass