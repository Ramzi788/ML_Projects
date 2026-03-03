from torch import nn
from sum_layer import Sum


class SemanticSegmentationBase(nn.Module):
    def __init__(self, netspec_opts):
        """

        Creates a fully convolutional neural network for the base semantic segmentation model. Given that there are
        several layers, we strongly recommend that you keep your layers in an nn.ModuleDict as described in
        the assignment handout. nn.ModuleDict mirrors the operations of Python dictionaries.

        You will specify the architecture of the module in the constructor. And then define the forward
        propagation in the forward method as described in the handout.

        Arguments
        ---------
        netspec_opts: (dictionary), the architecture of the base semantic network. netspec_opts has the keys
                                    1. kernel_size: (list) of size L where L is the number of layers
                                        representing the kernel sizes
                                    2. layer_type: (list) of size L indicating the type of each layer
                                    3. num_filters: (list) of size L representing the number of filters for each layer
                                    4. stride: (list) of size L indicating the striding factor of each layer
                                    5. input: (List) of size L containing the layer number of the inputs for each layer.

        """
        super(SemanticSegmentationBase, self).__init__()

        self.netspec_opts = netspec_opts
        self.net = nn.ModuleDict()

        layer_names = [
            'conv_1', 'bn_1', 'relu_1',
            'conv_2', 'bn_2', 'relu_2',
            'conv_3', 'bn_3', 'relu_3',
            'conv_4', 'bn_4', 'relu_4',
            'conv_5', 'upsample_4x', 'skip_6', 'sum_6', 'upsample_2x'
        ]
        self.layer_names = layer_names

        layer_type = netspec_opts['layer_type']
        kernel_size = netspec_opts['kernel_size']
        num_filters = netspec_opts['num_filters']
        stride = netspec_opts['stride']
        layer_input = netspec_opts['input']

        num_layers = len(layer_type)

        out_channels = {0: 3}

        for i in range(num_layers):
            name = layer_names[i]
            lt = layer_type[i]

            if lt == 'conv':
                inp_layer = layer_input[i]
                in_ch = out_channels[inp_layer]
                k = kernel_size[i]
                s = stride[i]
                pad = k // 2
                self.net[name] = nn.Conv2d(in_ch, num_filters[i], k, stride=s, padding=pad)
                out_channels[i + 1] = num_filters[i]

            elif lt == 'bn':
                self.net[name] = nn.BatchNorm2d(num_filters[i])
                inp_layer = layer_input[i]
                out_channels[i + 1] = out_channels[inp_layer]

            elif lt == 'relu':
                self.net[name] = nn.ReLU(inplace=True)
                inp_layer = layer_input[i]
                out_channels[i + 1] = out_channels[inp_layer]

            elif lt == 'convt':
                inp_layer = layer_input[i]
                in_ch = out_channels[inp_layer]
                k = kernel_size[i]
                s = stride[i]
                pad = (k - s) // 2
                self.net[name] = nn.ConvTranspose2d(
                    in_ch, num_filters[i], k, stride=s, padding=pad,
                    groups=in_ch, bias=False
                )
                out_channels[i + 1] = num_filters[i]

            elif lt == 'skip':
                inp_layer = layer_input[i]
                in_ch = out_channels[inp_layer]
                k = kernel_size[i]
                s = stride[i]
                self.net[name] = nn.Conv2d(in_ch, num_filters[i], k, stride=s, padding=0)
                out_channels[i + 1] = num_filters[i]

            elif lt == 'sum':
                self.net[name] = Sum()
                inp1, inp2 = layer_input[i]
                out_channels[i + 1] = out_channels[inp1]


    def forward(self, x):
        """
        Define the forward propagation of the base semantic segmentation model here. Starting with the input, pass
        the output of each layer to the succeeding layer until the final layer. Return the output of final layer
        as the predictions.

        Arguments
        ---------
        x: (Tensor) of size (B x C X H X W) where B is the mini-batch size, C is the number of
            channels and H and W are the spatial dimensions. X is the input activation volume.

        Returns
        -------
        out: (Tensor) of size (B x C' X H x W) where C' is the number of classes.

        """
        
        # implement the forward propagation as defined in the handout
        layer_input = self.netspec_opts['input']
        layer_type = self.netspec_opts['layer_type']
        outputs = {0: x}

        for i, name in enumerate(self.layer_names):
            layer_num = i + 1 
            lt = layer_type[i]
            inp_spec = layer_input[i]

            if lt == 'sum':
                inp1_num, inp2_num = inp_spec
                outputs[layer_num] = self.net[name](outputs[inp1_num], outputs[inp2_num])
            else:
                outputs[layer_num] = self.net[name](outputs[inp_spec])

        # return the final activation volume
        out = outputs[len(self.layer_names)]
        return out
