import torch
a = torch.randn(3, requires_grad=True)
b = 3*a + 1
c = torch.relu(b)
d = c.sum()
