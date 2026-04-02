# --------------------------------------------------------------
# ResNet.py
# ResBlock : Basic residual block (InstanceNorm + LeakyReLU)
# ResNet   : Lightweight ResNet used as the unrolled regularizer
# --------------------------------------------------------------

import torch.nn as nn

from .modules import conv_layer, ResNetBlocksModule, ResBlock

class ResNet(nn.Module):
    def __init__(self, device, in_ch=2, num_of_resblocks=5, out_ch=2):
        super().__init__()
        self.device = device
        kernel_size = 3
        features    = 64
        filter1     = [kernel_size, in_ch,     features]
        filter2     = [kernel_size, features,  features]
        filter3     = [kernel_size, features,  out_ch]

        self.layer1 = conv_layer(filter_size=filter1, activation_type='None')
        self.layer2 = ResNetBlocksModule(
            device=self.device, filter_size=filter2,
            num_blocks=num_of_resblocks)
        self.layer3 = conv_layer(filter_size=filter2, activation_type='None')
        self.layer4 = conv_layer(filter_size=filter3, activation_type='None')

    def forward(self, input_x):
        l1_out = self.layer1(input_x)
        l2_out = self.layer2(l1_out)
        l3_out = self.layer3(l2_out)
        temp   = l3_out + l1_out
        return self.layer4(temp)
