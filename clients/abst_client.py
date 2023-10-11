import copy
class AbstractClient:
    def __init__(self, root_model):
        self.model = copy.deepcopy(root_model)
        self.device = self.model.device
        self.loss = 0
        self.batch_num = 0

    def batch_train(self,x,y):
        raise NotImplementedError("Batch_train() is not implemented.")

    def client_info(self):
        info = {'model': self.model.state_dict(), 'loss':self.loss/self.batch_num}
        return info
