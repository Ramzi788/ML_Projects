from torch import nn


class Sum(nn.Module):
    def __init__(self):
        """
        An element-wise addition layer
        """
        super(Sum, self).__init__()

    def forward(self, x1, x2):
        """
        The forward propagation. A simple element-wise addition

        Arguments
        ---------
         x1: (Tensor), one of the input volumes
         x2: (Tensor), the other input volume in the addition

        Returns
        ------
        The element-wise sum of x1 and x2

        """
        return x1 + x2

