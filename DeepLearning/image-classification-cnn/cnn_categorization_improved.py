import torch
import torch.nn as nn
from torchvision import transforms

class AugmentedModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])

    def forward(self, x):
        if self.training:
            x = self.augment(x)
            return self.net(x)
        else:
            out1 = self.net(x)
            out2 = self.net(torch.flip(x, dims=[3]))
            return (out1 + out2) / 2

def cnn_categorization_improved(netspec_opts):
    net = nn.Sequential()
    in_channels = 3

    layer_types = netspec_opts['layer_type']
    kernel_sizes = netspec_opts['kernel_size']
    num_filters = netspec_opts['num_filters']
    strides = netspec_opts['stride']

    conv_count = 0
    bn_count = 0
    relu_count = 0
    pool_count = 0
    drop_count = 0

    for i in range(len(layer_types)):
        if layer_types[i] == 'conv':
            conv_count += 1
            padding = (kernel_sizes[i] - 1) // 2
            name = f'conv_{conv_count}' if conv_count < len([l for l in layer_types if l == 'conv']) else 'pred'
            net.add_module(name, nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=padding))
            in_channels = num_filters[i]
        elif layer_types[i] == 'bn':
            bn_count += 1
            net.add_module(f'bn_{bn_count}', nn.BatchNorm2d(num_filters[i]))
        elif layer_types[i] == 'relu':
            relu_count += 1
            net.add_module(f'relu_{relu_count}', nn.ReLU())
        elif layer_types[i] == 'pool':
            pool_count += 1
            net.add_module(f'pool_{pool_count}', nn.AvgPool2d(kernel_sizes[i], strides[i], padding=0))
        elif layer_types[i] == 'dropout':
            drop_count += 1
            net.add_module(f'drop_{drop_count}', nn.Dropout2d(0.15))

    return AugmentedModel(net)