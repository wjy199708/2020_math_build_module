from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import torch
from PIL import Image
import numpy as np
from torchsummary import summary


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_class=1):
        # bn_size (int): 这个是在block里一个denselayer里两个卷积层间的channel数 需要bn_size*growth_rate
        super(DenseNet, self).__init__()

        self.num_class = num_class
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(1, num_init_features, 7, 2, 3, bias=False)),  # 64, 1/2
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(3, 2, 1))
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features,
                                bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i+1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config)-1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module("transition%d" % (i+1), trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))
        self.classifier = nn.Linear(num_features, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(features, (features.shape[2], features.shape[3]), 1).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out, features


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features,
                                           bn_size*growth_rate, 1, 1, bias=False))

        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(
            bn_size*growth_rate, growth_rate, 3, 1, 1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i+1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features,
                                          num_output_features, 1, 1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, 2))


if __name__ == "__main__":
    device=torch.device('cuda')
    dense_net = DenseNet().to(device)
    x = torch.randint(0, 255, size=(2, 1, 1280, 720),
                      device=device, dtype=torch.float32)
    summary(dense_net,(1,1280,720),device='cuda')

    out, features = dense_net(x)
    print(out.shape)
    print(out)
    # print(torch.argmax(out,dim=1,keepdim=True))
    # print(torch.gather(out,dim=1,index=torch.argmax(out,dim=1,keepdim=True)))
