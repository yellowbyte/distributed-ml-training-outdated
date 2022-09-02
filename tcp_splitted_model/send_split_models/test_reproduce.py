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


import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# set some hyperparameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
lr = 0.001




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


# class ToyModel1(ToyModelBase):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(ToyModel1, self).__init__()
#         self.net1 = torch.nn.Linear(input_size, hidden_size)
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, out):
#         out = out.to_here()
#         #with self._lock:
#         out = self.relu(self.net1(out))
#         return out.cpu()
#
# class ToyModel2(ToyModelBase):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(ToyModel2, self).__init__()
#         self.net2 = torch.nn.Linear(hidden_size, num_classes)
#
#     def forward(self, out):
#         out = out.to_here()
#         #with self._lock:
#         out = self.net2(out)
#         return out.cpu()


class DistToyModel(nn.Module):
    def __init__(self, split_size, input_size, hidden_size, num_classes, workers, *args, **kwargs):
        super(DistToyModel, self).__init__()
        self.split_size = split_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        #meta_models(2)

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            #models[0],
            TM0,
            # ToyModel1,
            args = (self.input_size, self.hidden_size, self.num_classes)
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            #models[1],
            TM1,
            #ToyModel2,
            args=(self.input_size, self.hidden_size, self.num_classes)
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            #fut_z = z_fut.then(lambda fut: fut.value())
            #fut_z = z_fut.add_done_callback(lambda fut: fut.value())
            # print("feature", z_fut.value())

            out_futures.append(z_fut)
        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        print("rref param len", len(remote_params))
        return remote_params


num_batches = 3
num_epochs = 1

# MNIST dataset    os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '29500'
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Let's check the data and labels
image, label = next(iter(train_loader))
print(image.shape, label.shape)


def run_master(split_size):

    # put the two model parts on worker1 and worker2 respectively
    print("Check run master")
    #model = DistToyModel(split_size, ["worker1", "worker2"])
    model = DistToyModel(split_size, input_size, hidden_size, num_classes, ["worker1", "worker2"])
    print("Initialize model")
    loss_fn = nn.CrossEntropyLoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.001,
    )
    print("finish initializing")


    total_step = len(train_loader)
    for epoch in range(num_epochs):
        print("Current Epoch :{}".format(epoch))
        for i, (images, label) in enumerate(train_loader):
            images = images.reshape(-1, 28 * 28)  # reshape tensor matrix to a vector form
            label = label

            with dist_autograd.context() as context_id:
                print(images)
                output = model(images)
                dist_autograd.backward(context_id, [loss_fn(output, label)])
                opt.step(context_id)
    # Test the model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28)
        labels = labels
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# if __name__ == "__main__":
    # model1 = ToyModel1()
    # model2 = ToyModel2()
    # loss_fn = nn.MSELoss()
    # optimizer1 = optim.SGD(model1.parameters(), lr=0.001)
    # optimizer2 = optim.SGD(model2.parameters(), lr=0.001)
    #
    # optimizer1.zero_grad()
    # optimizer2.zero_grad()
    # # outputs = model2(torch.randn(20, 10))
    #
    # loss_fn(outputs, labels).backward()
    # optimizer.step()


    ################################################################################
    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)
    # os.environ['GLOO_SOCKET_IFNAME'] = 'wlp72s0'
    # os.environ['TP_SOCKET_IFNAME'] = 'wlp72s0'
    # os.environ['MASTER_ADDR'] = '192.168.0.195'
    # os.environ['MASTER_PORT'] = '29411'
    # print(os.environ.get('GLOO_SOCKET_IFNAME'))
    # rpc.init_rpc("master", rank=0, world_size=3, rpc_backend_options=options)
    # # rpc.init_rpc("worker1", rank=1, world_size=3, rpc_backend_options=options)
    # # rpc.init_rpc("worker2", rank=2, world_size=3, rpc_backend_options=options)
    # print("check whether worker is init")
    #
    #
    # print("splitting")
    # run_master(1)


def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = '192.168.0.195'
    os.environ['MASTER_PORT'] = '29414'
    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=700, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


def track_free_cpu_mem(cpu_free_mem_input):
    # get cpu usage
    oom_counter = 0
    starttime = time.time()
    while True:
        time.sleep(5.0 - ((time.time() - starttime) % 5.0))
        result = subprocess.run(
            # -b: batch-mode operation
            # -n 1: number of iterations limit
            ["adb", "shell", "top", "-b", "-n", "1"], capture_output=True, text=True
        ).stdout
        mem_usage = [l for l in result.split('\n')][1]
        cpu_free_mem = float(mem_usage.split()[5][:-1])
        cpu_free_mem_input = cpu_free_mem
        print("get cpu free memory: {}".format(cpu_free_mem))
        if cpu_free_mem < 0.1:
            oom_counter += 1
        if oom_counter > 5:
            print("Out of Memeory: Need to resplit the model! Process terminated.")
            break
# program to create class dynamically

# constructor
def constructor(self, arg):
    self.constructor_arg = arg

# method
def displayMethod(self, arg):
    print(arg)

# class method
@classmethod
def classMethod(cls, arg):
    print(arg)

# creating class dynamically
Geeks = type("Geeks", (object, ), {
    # constructor
    "__init__": constructor,

    # data members
    "string_attribute": "Geeks 4 geeks !",
    "int_attribute": 1706256,

    # member functions
    "func_arg": displayMethod,
    "class_func": classMethod
})

# creating objects
obj = Geeks("constructor argument")
print(obj.constructor_arg)
print(obj.string_attribute)
import subprocess


def meta_split(layers_num):
    init_body_whole = [
        "self.net1 = torch.nn.Linear(input_size, hidden_size)\n    self.relu = torch.nn.ReLU()",
        "self.net2 = torch.nn.Linear(hidden_size, num_classes)",
    ]
    forward_body_whole = [
        "out = self.relu(self.net1(out))\n    return out.cpu()",
        "out = self.net2(out)\n    return out.cpu()",
    ]
    devices_init = list()
    devices_body = list()

    # divide in half
    # TODO: split based on function argument
    devices_init.append(init_body_whole[0])
    devices_init.append(init_body_whole[1])
    devices_body.append(forward_body_whole[0])
    devices_body.append(forward_body_whole[1])

    return (devices_init,devices_body)


import socket, pickle


devices_num = 2
classes = list()
devices_init,devices_body = meta_split(2)

toy_models = list()
for i in range(devices_num):
    # create model that initialize/execute specific layers
    init_str = ["def __init__(self, input_size, hidden_size, num_classes):"] + ["super(TM"+str(i)+", self).__init__()"] + [devices_init[i]]
    forward_str = ["def forward(self, out):"] + ["out = out.to_here()"] + [devices_body[i]]

    #exec("global TM"+str(i))
    exec("\n    ".join(init_str))
    exec("\n    ".join(forward_str))
    exec("TM"+str(i)+" = type('TM'+str(i),(ToyModelBase,),{'__init__': globals()['__init__'],'forward': globals()['forward']})")
    toy_models.append(("\n    ".join(init_str), "\n    ".join(forward_str), "TM"+str(i)+" = type('TM"+str(i)+"',(ToyModelBase,),{'__init__': globals()['__init__'],'forward': globals()['forward']})"))

if __name__ == "__main__":
    # model1 = ToyModel1()
    # model2 = ToyModel2()
    # loss_fn = nn.MSELoss()
    # optimizer1 = optim.SGD(model1.parameters(), lr=0.001)
    # optimizer2 = optim.SGD(model2.parameters(), lr=0.001)
    #
    # optimizer1.zero_grad()
    # optimizer2.zero_grad()
    # # outputs = model2(torch.randn(20, 10))
    #
    # loss_fn(outputs, labels).backward()
    # optimizer.step()

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)
    # os.environ['GLOO_SOCKET_IFNAME'] = 'wlp72s0'
    # os.environ['TP_SOCKET_IFNAME'] = 'wlp72s0'
    os.environ['GLOO_SOCKET_IFNAME'] = 'enp71s0'
    os.environ['TP_SOCKET_IFNAME'] = 'enp71s0'
    # os.environ['MASTER_ADDR'] = '192.168.0.195'
    os.environ['MASTER_ADDR'] = '128.195.41.40'
    os.environ['MASTER_PORT'] = '29412'
    #os.environ['MASTER_ADDR'] = '128.195.41.40'
    os.environ['WORKER1_PORT'] = '29402'
    os.environ['WORKER2_PORT'] = '29400'

    print(os.environ.get('GLOO_SOCKET_IFNAME'))

    # Sending Pickled Class
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #s.bind((os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])))
    #s.listen(5)
    #conn, addr = s.accept()
    #print('Connected by', addr)
    HEADERSIZE = 10
    #d = {1: "hi", 2: "there"}
    data = {
        # constructor
        "__init__": "globals()['__init__']",
        # member functions
        "forward": "globals()['forward']"
    }
    #breakpoint()
    # msg = pickle.dumps(toy_models[0])
    # msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    # print(msg)
    # conn.send(msg)
    #data = conn.recv(16)
    #breakpoint()
    s.connect((os.environ['MASTER_ADDR'], int(os.environ['WORKER1_PORT'])))
    msg = pickle.dumps(toy_models[0])
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    print(msg)
    s.send(msg)
    s.close()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((os.environ['MASTER_ADDR'], int(os.environ['WORKER2_PORT'])))
    msg = pickle.dumps(toy_models[1])
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    print(msg)
    s.send(msg)
    s.close()

    # Perform Distributed Learning
    rpc.init_rpc("master", rank=0, world_size=3, rpc_backend_options=options)
    run_master(1)
    print("after run_master")
