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

from torchvision.models.resnet import Bottleneck


#########################################################
#           Define Model Parallel ResNet50              #
#########################################################

# In order to split the ResNet50 and place it on two different workers, we
# implement it in two model shards. The ResNetBase class defines common
# attributes and methods shared by two shards. ResNetShard1 and ResNetShard2
# contain two partitions of the model layers respectively.

#
num_classes = 10
#
#
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=1000,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()

        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                      base_width=self.base_width, dilation=self.dilation,
                                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]
#
#
# class ResNetShard1(ResNetBase):
#     """
#     The first part of ResNet.
#     """
#     def __init__(self, device, *args, **kwargs):
#         super(ResNetShard1, self).__init__(
#             Bottleneck, 64, num_classes=num_classes, *args, **kwargs)
#
#         self.device = device
#         self.seq = nn.Sequential(
#             nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
#             self._norm_layer(self.inplanes),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             self._make_layer(64, 3),
#             self._make_layer(128, 4, stride=2)
#         ).to(self.device)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#
#     def forward(self, x_rref):
#         x = x_rref.to_here().to(self.device)
#         with self._lock:
#             out =  self.seq(x)
#         return out.cpu()
#
#
# class ResNetShard2(ResNetBase):
#     """
#     The second part of ResNet.
#     """
#     def __init__(self, device, *args, **kwargs):
#         super(ResNetShard2, self).__init__(
#             Bottleneck, 512, num_classes=num_classes, *args, **kwargs)
#
#         self.device = device
#         self.seq = nn.Sequential(
#             self._make_layer(256, 6, stride=2),
#             self._make_layer(512, 3, stride=2),
#             nn.AdaptiveAvgPool2d((1, 1)),
#         ).to(self.device)
#
#         self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)
#
#     def forward(self, x_rref):
#         x = x_rref.to_here().to(self.device)
#         with self._lock:
#             out = self.fc(torch.flatten(self.seq(x), 1))
#         return out.cpu()
#
#
class DistResNet50(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """
    def __init__(self, split_size, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.split_size = split_size

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.rpc_sync(
            workers[0],
            ResNetShard1,
            args = args,
            kwargs = kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.rpc_sync(
            workers[1],
            ResNetShard2,
            args = args,
            kwargs = kwargs
        )

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params

import torchvision
from torchvision import transforms
#########################################################
#                   Run RPC Processes                   #
#########################################################
input_size = 784
hidden_size = 500
num_epochs = 5
batch_size = 100
lr = 0.1
num_batches = 3
num_epochs = 1

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
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
# image, label = next(iter(train_loader))
# print(image.shape, label.shape)
# #breakpoint()


def run_master(split_size):
    print("check run master resnet")
    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(split_size, ["worker1", "worker2"])
    loss_fn = nn.CrossEntropyLoss()
    opt = DistributedOptimizer(
        optim.Adam,
        model.parameter_rrefs(),
        lr=0.05,
    )

    # one_hot_indices = torch.LongTensor(batch_size) \
    #                        .random_(0, num_classes) \
    #                        .view(batch_size, 1)

    # for i in range(num_batches):
    #     print(f"Processing batch {i}")
    #     # generate random inputs and labels
    #     # inputs = torch.randn(batch_size, 3, image_w, image_h)
    #     # labels = torch.zeros(batch_size, num_classes) \
    #     #               .scatter_(1, one_hot_indices, 1)
    #
    #     # The distributed autograd context is the dedicated scope for the
    #     # distributed backward pass to store gradients, which can later be
    #     # retrieved using the context_id by the distributed optimizer.
    #     with dist_autograd.context() as context_id:
    #         outputs = model(inputs)
    #         dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
    #         opt.step(context_id)

    # total_step = len(train_loader)
    # curr_lr = learning_rate
    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(train_loader):
    #         images = images.to(device)
    #         labels = labels.to(device)
    #
    #         # Forward pass
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         if (i + 1) % 100 == 0:
    #             print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
    #                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    #
    #     # Decay learning rate
    #     if (epoch + 1) % 20 == 0:
    #         curr_lr /= 3
    #         update_lr(optimizer, curr_lr)
    #

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        print("Current Epoch :{}".format(epoch))
        for i, (images, label) in enumerate(train_loader):
            images = images  # reshape tensor matrix to a vector form
            label = label

            with dist_autograd.context() as context_id:
                # print(images)
                output = model(images)
                dist_autograd.backward(context_id, [loss_fn(output, label)])
                opt.step(context_id)
            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}]"
                      .format(epoch + 1, num_epochs, i + 1, total_step))
    # Test the model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images
        labels = labels
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


def run_worker(rank, world_size, num_split):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29500'
    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

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
# num_batches = 3
# batch_size = 120
# image_w = 128
# image_h = 128

#
# def run_master(split_size):
#
#     # put the two model parts on worker1 and worker2 respectively
#     model = DistResNet50(split_size, ["worker1", "worker2"])
#     loss_fn = nn.MSELoss()
#     opt = DistributedOptimizer(
#         optim.SGD,
#         model.parameter_rrefs(),
#         lr=0.05,
#     )
#
#     one_hot_indices = torch.LongTensor(batch_size) \
#                            .random_(0, num_classes) \
#                            .view(batch_size, 1)
#
#     for i in range(num_batches):
#         print(f"Processing batch {i}")
#         # generate random inputs and labels
#         inputs = torch.randn(batch_size, 3, image_w, image_h)
#         labels = torch.zeros(batch_size, num_classes) \
#                       .scatter_(1, one_hot_indices, 1)
#
#         # The distributed autograd context is the dedicated scope for the
#         # distributed backward pass to store gradients, which can later be
#         # retrieved using the context_id by the distributed optimizer.
#         with dist_autograd.context() as context_id:
#             outputs = model(inputs)
#             dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
#             opt.step(context_id)
#
#
# def run_worker(rank, world_size, num_split):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '29500'
#
#     # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
#     options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)
#
#     if rank == 0:
#         rpc.init_rpc(
#             "master",
#             rank=rank,
#             world_size=world_size,
#             rpc_backend_options=options
#         )
#         run_master(num_split)
#     else:
#         rpc.init_rpc(
#             f"worker{rank}",
#             rank=rank,
#             world_size=world_size,
#             rpc_backend_options=options
#         )
#         pass
#
#     # block until all rpcs finish
#     rpc.shutdown()

classes = list()
init_body = [
    "self.cov2d = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)",
    "self.layer_norm = self._norm_layer(self.inplanes)",
    "self.relu = nn.ReLU(inplace=True)",
    "self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)",
    "self.make_layer1 = self._make_layer(64, 3)",
    "self.make_layer2 = self._make_layer(128, 4, stride=2)",
    "self.make_layer3 = self._make_layer(256, 6, stride=2)",

    "self.make_layer4 = self._make_layer(512, 3, stride=2)",
    "self.avgpool = nn.AdaptiveAvgPool2d((1, 1))",
    "self.fc = nn.Linear(512 * self._block.expansion, num_classes)",
    ""
]


forward_body = [
    "x1 = self.cov2d(x1)",
    "x1 = self.layer_norm(x1)",
    "x1 = self.relu(x1)",
    "x1 = self.maxpool(x1)",
    "x1 = self.make_layer1(x1)",
    "x1 = self.make_layer2(x1)",
    "x1 = self.make_layer3(x1)",
    "x1 = self.make_layer4(x1)",
    "x1 = self.avgpool(x1)",
    "x1 = self.fc(torch.flatten(x1, 1))",
    ""
]

dependencies = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    7:7,
    8:8,
    9:9,
    10:10
}


def extract_layers(init_body_whole, forward_body_whole, dependencies, start_index, end_index):
    devices_init = init_body_whole[dependencies[start_index]:dependencies[end_index]]
    devices_body = forward_body_whole[start_index:end_index]

    # "self.cov2d = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)",
    # "self.layer_norm = self._norm_layer(self.inplanes)",
    # "self.relu = nn.ReLU(inplace=True)",
    # "self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)",
    # "self.make_layer1 = self._make_layer(64, 3)",
    # "self.make_layer2 = self._make_layer(128, 4, stride=2)",
    # "self.make_layer3 = self._make_layer(256, 6, stride=2)",
    #
    # "self.make_layer4 = self._make_layer(512, 3, stride=2)",
    # "self.avgpool = nn.AdaptiveAvgPool2d((1, 1))",
    # initialize the inplane number
    inplane_num_l = []
    inplane_num= 0
    for lines in devices_init:
        if "self._make_layer" not in lines:
            continue
        else:
            inplane_num_l.append(int(lines.split("(")[1].split(",")[0]))
    if len(inplane_num_l) != 0:
        inplane_num = inplane_num_l[-1]

    return ("\n    ".join(devices_init),"\n    ".join(devices_body), inplane_num)


def create_model(devices_init, devices_body,inplane_num, i):
    # create model that initialize/execute specific layers
    init_str = ["def __init__(self, device, *args, **kwargs):"] + ["super(RN"+str(i)+", self).__init__(Bottleneck, "+str(inplane_num)+", num_classes=num_classes, *args, **kwargs)"] + [devices_init]
    forward_str = ["def forward(self, x1):"] + ["x1 = x1.to_here()"] + [devices_body] + ["return x1.cpu()"]
    #print('@@@ INIT: ', init_str)
    #print("@@@ FORWARD: ", forward_str)
    #exec("global TM"+str(i))
    exec("\n    ".join(init_str))
    exec("\n    ".join(forward_str))
    #exec("global __init__")
    #exec("global forward")
    exec("TM = type('RN"+str(i)+"',(ResNetBase,),{'__init__': locals()['__init__'],'forward': locals()['forward']})")
    exec("global RN"+str(i))
    globals()['RN'+str(i)] = locals()['TM']


import socket, pickle


if __name__=="__main__":
    # world_size = 3
    # for num_split in [1, 2, 4, 8]:
    #     tik = time.time()
    #     mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
    #     tok = time.time()
    #     print(f"number of splits = {num_split}, execution time = {tok - tik}")
    # dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
    #                         rank=3, world_size=3)


    os.environ['MASTER_ADDR'] = '192.168.0.195'
    os.environ['MASTER_PORT'] = '29411'
    os.environ['GLOO_SOCKET_IFNAME'] = 'wlp72s0'
    os.environ['TP_SOCKET_IFNAME'] = 'wlp72s0'
    os.environ['WORKER1_PORT'] = '29402'

    # Get Splitted Model
    HEADERSIZE = 10
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((os.environ['MASTER_ADDR'], int(os.environ['WORKER1_PORT'])))
    s.listen(5)
    conn, addr = s.accept()

    #s.connect((os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])))
    done = False
    while True:
        if done:
            break
        full_msg = b''
        new_msg = True
        while True:
            msg = conn.recv(16)
            if new_msg:
                #print("new msg len:", msg[:HEADERSIZE])
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            #print(f"full message length: {msglen}")

            full_msg += msg

            #print(len(full_msg))

            if len(full_msg) - HEADERSIZE == msglen:
                print("full msg recvd")
                print(full_msg[HEADERSIZE:])
                data = pickle.loads(full_msg[HEADERSIZE:])
                start_index, end_index = data.split(":")
                start_index = int(start_index)
                end_index = int(end_index)
                new_msg = True
                full_msg = b""
                done = True
                break

    #breakpoint()
    conn.close()

    devices_init1, devices_body1, inplane_num1 = extract_layers(init_body, forward_body, dependencies, start_index, end_index)
    create_model(devices_init1, devices_body1, inplane_num1, 0)
    #breakpoint()

    # Perform Distributed Training
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)
    rpc.init_rpc("worker1", rank=1, world_size=3, rpc_backend_options=options)
    # rref1 = rpc.rpc_sync("worker1", _run_trainer())
    rpc.shutdown()