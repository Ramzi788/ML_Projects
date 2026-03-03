from torch import nn

# import all other functions you may need


class SemanticSegmentationImproved(nn.Module):
    def __init__(self, netspec_opts):
        """

        Creates a fully convolutional neural network for the improve semantic segmentation model.


        Arguments
        ---------
        netspec_opts: (dictionary), the architecture of the base semantic network.

        """
        super(SemanticSegmentationImproved, self).__init__()
        
        # implement the improvement model architecture

    def forward(self, x):
        """
        Define the forward propagation of the improvement model.

        Arguments
        ---------
        x: (Tensor) of size (B x C X H X W) where B is the mini-batch size, C is the number of
            channels and H and W are the spatial dimensions. X is the input activation volume.

        Returns
        -------
        out: (Tensor) of size (B x C' X H x W), where C' is the number of classes.

        """

        # implement the forward propagation

        # return the final activation volume
        return out
