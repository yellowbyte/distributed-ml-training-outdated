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
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10
}
