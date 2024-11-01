import torch
import torch.nn as nn
from torch.distributed.rpc import RRef

from torchvision.models.resnet import Bottleneck


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


INIT_BODY = [
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
]

FORWARD_BODY = [
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
]

INIT_HEADING = ["def __init__(self, device, *args, **kwargs):"] + \
               ["super(MODELBASE, self).__init__(Bottleneck, 64, num_classes=num_classes, *args, **kwargs)"] + \
               ["self.device = 'cpu'"]
INIT_ENDING = []
FORWARD_HEADING = ["def forward(self, x1):"] + \
                  ["x1 = x1.to_here()"]
FORWARD_ENDING = ["return x.cpu()"]

SHAPES = {
    0: (10,3,32,32,"randn","float"),  
    1: (64,128,"randint"),
    2: (500,100,"randint"),
    3: (500,100,"randint"),
    4: (500,100,"randint"),
    5: (500,100,"randint"),
    6: (500,100,"randint"),
}


