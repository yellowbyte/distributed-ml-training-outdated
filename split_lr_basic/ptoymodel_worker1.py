import os
import threading
import time
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import torch.distributed as dist

from torch.profiler import profile, record_function, ProfilerActivity


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
    def __init__(self):
        super(ToyModel1, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x_rref):
        #with torch.autograd.profiler.profile(record_shapes=True) as prof:
        x = x_rref.to_here()
        start = time.time()
        net1 = self.net1(x)
        print("net1:", time.time()-start)
        start = time.time()
        out = self.relu(net1)
        print("relu:", time.time()-start)
        #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return out.cpu()

class ToyModel2(ToyModelBase):
    def __init__(self):
        super(ToyModel2, self).__init__()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x_rref):
        x = x_rref.to_here()
        with self._lock:
            out = self.net2(x)
        return out.cpu()

class DistToyModel(nn.Module):
    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistToyModel, self).__init__()
        self.split_size = split_size

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ToyModel1
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ToyModel2
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        print("forwarding? Dist class")
        for x in iter(xs.split(self.split_size, dim=0)):
            print("split size", self.split_size)
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        print("parameter rref???? Dist class")
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        print("rref param len", len(remote_params))
        print("parameter rref finish???", remote_params)
        return remote_params



num_batches = 3



def run_master(split_size):

    # put the two model parts on worker1 and worker2 respectively
    print("Check run master")
    model = DistToyModel(split_size, ["worker1", "worker2"])
    print(model)
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )


    for i in range(num_batches):
        print(f"Processing batch {i}")
        # generate random inputs and labels
        inputs = torch.randn(20, 10)
        labels = torch.randn(20, 5)

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            # start forward
            print("checkpoint forward")
            outputs = model(inputs)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)

if __name__ == "__main__":
    os.environ['GLOO_SOCKET_IFNAME'] = 'enp71s0'
    os.environ['TP_SOCKET_IFNAME'] = 'enp71s0'
    os.environ['MASTER_ADDR'] = '128.195.41.40'
    os.environ['MASTER_PORT'] = '29412'
    print(os.environ.get('GLOO_SOCKET_IFNAME'))

    #with torch.autograd.profiler.profile(record_shapes=True) as prof:
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)
    rpc.init_rpc("worker1", rank=1, world_size=3, rpc_backend_options=options)
    print("check whether worker is init")
    #print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))

    rpc.shutdown()
