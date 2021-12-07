import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# Model here
def conv3x3unit(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3unit(self.in_channels, self.out_channels)
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3unit(self.out_channels, self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)
        
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class Conv_bn_relu(nn.Module):
    """
    A helper Module that performs a convolution, a batchnorm and 
    a relu based on request
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, do_relu=True):
        super(Conv_bn_relu, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.do_relu = do_relu

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size,
                              stride = stride, padding = padding)
        self.norm1 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        if self.do_relu:
            x = F.relu(x)
        return x
    

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3unit(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3unit(self.out_channels, self.out_channels)
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3unit(self.out_channels, self.out_channels)
        self.norm2 = nn.BatchNorm2d(self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x


class UNet(BaseModel):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, scaling_factor = 800.0, in_channels=2, depth=5,
                 start_filts=32, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.intermediate = []
        
        self.dropout = nn.Dropout(p = 0.5)

        #self.hardtanh = nn.Hardtanh(0.0, scaling_factor)
        # create the encoder pathway and add to a list
        for i in range(depth):
            if i < 3:
                ins = self.in_channels if i == 0 else outs
                outs = self.start_filts*(2**i)
                pooling = True if i < depth-1 else False

                down_conv = DownConv(ins, outs, pooling=pooling)
                self.down_convs.append(down_conv)
            else:
                if i == 3:
                    self.intermediate.append(Conv_bn_relu(128, 512, 3, stride=1, padding=1, do_relu=True))
                    self.intermediate.append(Conv_bn_relu(512, 512, 3, stride=1, padding=1, do_relu=False))
                    self.intermediate.append(Conv_bn_relu(128, 512, 1, stride=1, padding=0, do_relu=False))
                else:
                    self.intermediate.append(Conv_bn_relu(512, 512, 3, stride=1, padding=1, do_relu=True))
                    self.intermediate.append(Conv_bn_relu(512, 512, 3, stride=1, padding=1, do_relu=True))
                    self.intermediate.append(Conv_bn_relu(512, 512, 1, stride=1, padding=0, do_relu=True))

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        self.up_convs_1, self.up_convs_2, self.up_convs_3, self.up_convs_4, self.up_convs_5, self.up_convs_6 = [], [], [], [], [], []
        self.all_upconvs = [self.up_convs_1, self.up_convs_2, self.up_convs_3, self.up_convs_4, self.up_convs_5, self.up_convs_6]
        for chn in range(6): # need a 4 channel output
            outs = 512
            for i in range(depth-2):
                ins = outs
                outs = ins // 4 if i == 0 else ins // 2 
                up_conv = UpConv(ins, outs, up_mode=up_mode,
                    merge_mode=merge_mode)
                self.all_upconvs[chn].append(up_conv)

            self.all_upconvs[chn].append(conv1x1(outs, 1)) # changed from 1 to 2 for CE loss

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.intermediate = nn.ModuleList(self.intermediate)
        self.up_convs_1 = nn.ModuleList(self.up_convs_1)
        self.up_convs_2 = nn.ModuleList(self.up_convs_2)
        self.up_convs_3 = nn.ModuleList(self.up_convs_3)
        self.up_convs_4 = nn.ModuleList(self.up_convs_4)
        self.up_convs_5 = nn.ModuleList(self.up_convs_5)
        self.up_convs_6 = nn.ModuleList(self.up_convs_6)
        # for chn in range(4):
        #     self.all_upconvs[chn] = nn.ModuleList(self.all_upconvs[chn])

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x, verbose = False):
        encoder_outs = []
        intermediate_out_1 = []
        intermediate_out_2 = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        # up to now, we have 128 channel 
        for i, module in enumerate(self.intermediate):
            if i == 0 or i == 3:
                intermediate_out_1.append(x)
                x = module(x)
                intermediate_out_2.append(x)
            if i == 1 or i == 4:
                x = module(x)
            if i == 2 or i == 5:
                current_out = module(intermediate_out_1[-1])
                x = current_out + x
                x = self.dropout(F.relu(x))
                
        intermediate_final = x

        if verbose:
            print('unet encoder output', x.shape)
        
        channel_outputs = []
        for chn in range(6): # change 4 to 1
            length = len(self.all_upconvs[chn])
            for i, module in enumerate(self.all_upconvs[chn]):
                if i < length - 1:
                    before_pool = encoder_outs[-(i+1)]
                    if i == 0:
                        x = module(before_pool, intermediate_final)
                    else:
                        x = module(before_pool, x)
                else:
                    x = module(x)
            channel_outputs.append(x)
        
        x = torch.cat(channel_outputs, 1)
        #x = self.hardtanh(x) # added for mse dice loss

        if verbose:
            print('unet decoder output', x.shape)

        return x







