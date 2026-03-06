from torch import nn

# import all other functions you may need
from sum_layer import Sum


class ResBlock(nn.Module):
    """Residual block with 3 conv layers, BN, ReLU, and a residual connection."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu3 = nn.ReLU(inplace=True)

        self.match_channels = None
        if in_ch != out_ch:
            self.match_channels = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0),
                nn.BatchNorm2d(out_ch)
            )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = x
        if self.match_channels is not None:
            residual = self.match_channels(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + residual
        out = self.relu3(out)

        if self.dropout is not None:
            out = self.dropout(out)
        return out


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
        self.netspec_opts = netspec_opts
        num_classes = netspec_opts.get('num_classes', 36)

        # Smaller filters + heavier dropout to reduce overfitting
        self.enc1 = ResBlock(3, 48, dropout=0.2)
        self.pool1 = nn.Sequential(nn.Conv2d(48, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))

        self.enc2 = ResBlock(48, 96, dropout=0.2)
        self.pool2 = nn.Sequential(nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))

        self.enc3 = ResBlock(96, 192, dropout=0.3)
        self.pool3 = nn.Sequential(nn.Conv2d(192, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))

        self.bottleneck = ResBlock(192, 384, dropout=0.5)

        self.up3 = nn.ConvTranspose2d(384, 192, 4, stride=2, padding=1, bias=False)
        self.up3_bn = nn.BatchNorm2d(192)
        self.up3_relu = nn.ReLU(inplace=True)
        self.skip3 = nn.Sequential(nn.Conv2d(192, 192, 1, stride=1, padding=0), nn.BatchNorm2d(192))
        self.sum3 = Sum()
        self.dec3 = ResBlock(192, 192, dropout=0.3)

        self.up2 = nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1, bias=False)
        self.up2_bn = nn.BatchNorm2d(96)
        self.up2_relu = nn.ReLU(inplace=True)
        self.skip2 = nn.Sequential(nn.Conv2d(96, 96, 1, stride=1, padding=0), nn.BatchNorm2d(96))
        self.sum2 = Sum()
        self.dec2 = ResBlock(96, 96, dropout=0.2)

        self.up1 = nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1, bias=False)
        self.up1_bn = nn.BatchNorm2d(48)
        self.up1_relu = nn.ReLU(inplace=True)
        self.skip1 = nn.Sequential(nn.Conv2d(48, 48, 1, stride=1, padding=0), nn.BatchNorm2d(48))
        self.sum1 = Sum()
        self.dec1 = ResBlock(48, 48, dropout=0.2)

        self.classifier = nn.Conv2d(48, num_classes, 1, stride=1, padding=0)

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
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        d3 = self.up3_relu(self.up3_bn(self.up3(b)))
        s3 = self.skip3(e3)
        d3 = self.sum3(d3, s3)
        d3 = self.dec3(d3)

        d2 = self.up2_relu(self.up2_bn(self.up2(d3)))
        s2 = self.skip2(e2)
        d2 = self.sum2(d2, s2)
        d2 = self.dec2(d2)

        d1 = self.up1_relu(self.up1_bn(self.up1(d2)))
        s1 = self.skip1(e1)
        d1 = self.sum1(d1, s1)
        d1 = self.dec1(d1)

        out = self.classifier(d1)
        # return the final activation volume
        return out