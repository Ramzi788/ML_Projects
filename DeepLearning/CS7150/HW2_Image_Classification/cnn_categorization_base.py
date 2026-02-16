from torch import nn

def cnn_categorization_base(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()

    # add layers as specified in netspec_opts to the network
    in_channels = 3
    layer_types = netspec_opts['layer_type']
    layer_names = ['conv_1', 'bn_1', 'relu_1', 'conv_2', 'bn_2', 'relu_2',
             'conv_3', 'bn_3', 'relu_3', 'pool_3', 'pred']
    kernel_sizes = netspec_opts['kernel_size']
    num_filters = netspec_opts['num_filters']
    strides = netspec_opts['stride']
    
    for i in range(len(layer_types)):
        if layer_types[i] == 'conv':
            padding = (kernel_sizes[i] - 1) // 2
            net.add_module(layer_names[i], nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_sizes[i], stride=strides[i], padding=padding))
            in_channels = num_filters[i]
        elif layer_types[i] == 'bn':
            net.add_module(layer_names[i], nn.BatchNorm2d(num_filters[i-1]))
        elif layer_types[i] == 'relu':
            net.add_module(layer_names[i], nn.ReLU())
        elif layer_types[i] == 'pool':
            net.add_module(layer_names[i], nn.MaxPool2d(kernel_sizes[i], stride=strides[i], padding=0))

    return net
