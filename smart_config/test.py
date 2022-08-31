# testing .grad_fn.next_functions to get internal computational graph

import torch
a = torch.randn(3, requires_grad=True)
b = 3*a + 1
c = torch.relu(b)
d = c.sum()
