import torch
import torch.nn as nn
from torch.distributed.rpc import RRef


class MODELBASE(nn.Module):
    def __init__(self, group=1):
        super(MODELBASE, self).__init__()
        self.group = group

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        # TODO: issue here
        return [RRef(p) for p in self.parameters()]


INIT_BODY = [
    "self.net1 = torch.nn.Linear(input_size, hidden_size)",
    "self.relu = torch.nn.ReLU()",
    "self.net2 = torch.nn.Linear(hidden_size, num_classes)",
]
FORWARD_BODY = [
    "x = self.relu(x)",
    "x = self.net1(x)",
    "x = self.net2(x)",
]
#FORWARD_BODY = [
#    "x = self.net2(self.relu(self.net1(x)))",
#]

# assume each element in the list is a split-able point
#FORWARD_BODY = [
#    "x = self.relu(self.net1(x))",
#    "x = self.net2(x)",
#]


INIT_HEADING = ["def __init__(self):"] + \
               ["super(MODEL, self).__init__()"] + \
               ["input_size=784"] + \
               ["hidden_size=500"] + \
               ["num_classes=2"]
INIT_ENDING = []


FORWARD_HEADING = ["def forward(self, x):"]
FORWARD_ENDING = ["return x"]
# to_here is only needed for actual rpc (RRef object): 
# need to put it back for actual split
#FORWARD_HEADING = ["def forward(self, x1):"] + ["x1 = x1.to_here()"]


SHAPES = {
    # forward_body index: (input size,distribution)
    0: (100,784,"rand"),     # element is between [0,1] for images
    1: (500,100,"randint"),  # element is a positive number for relu
}
