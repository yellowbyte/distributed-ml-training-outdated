import torch
import torch.nn as nn
from torch.distributed.rpc import RRef


class ToyModelBase(nn.Module):
    def __init__(self, group=1):
        super(ToyModelBase, self).__init__()
        # self.net1 = torch.nn.Linear(10, 10)
        # self.relu = torch.nn.ReLU()
        # self.net2 = torch.nn.Linear(10, 5)
        self.group = group
        self._lock = threading.Lock()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        # TODO: issue here
        return [RRef(p) for p in self.parameters()]


class ToyModel1(ToyModelBase):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ToyModel1, self).__init__()
        self.net1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()

    def forward(self, out):
        out = out.to_here()
        out = self.relu(self.net1(out))
        return out.cpu()

class ToyModel2(ToyModelBase):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ToyModel2, self).__init__()
        self.net2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, out):
        out = out.to_here()
        out = self.net2(out)
        return out.cpu()

class ToyModel(ToyModelBase):
    def __init__(self): 
        super(ToyModelBase,self).__init__()
        input_size = 784
        hidden_size=500
        num_classes=2
        self.net1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #x = self.relu(self.net1(x))
        x = self.net2(self.relu(self.net1(x)))
        return x

