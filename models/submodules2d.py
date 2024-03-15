# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function
import torch.nn as nn
from models.submodules import SubModule, convbn_2d_lrelu,deconvbn_2d_lrelu
import torch 
from nets.warp import disp_warp
import torch.nn.functional as F
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp + 1, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp + 1, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.1, inplace=True))
# GANet feature
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 mdconv=False):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            # if mdconv:
                # self.conv2 = DeformConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
            # else:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
                                       stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x
class HourglassRefinement(SubModule):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(HourglassRefinement, self).__init__()

        # Left and warped error
        in_channels = 2
        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(1, 16)  # on low disparity

        self.conv_start = conv2d(32, 32)

        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2,padding=1)
        
        self.attn = MultiheadSelfAttention(96, 16)
        
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, low_disp, left_img, right_img):
        scale_factor = left_img.size(-1) / low_disp.size(-1)
        if scale_factor == 1.0:
            disp = low_disp
        else:
            disp = F.interpolate(low_disp, size=left_img.size()[-2:], mode='bilinear')
            disp = disp * scale_factor
        warped_right = disp_warp(right_img, disp)[0]  # [B, C, H, W]
        error = warped_right - left_img  # [B, C, H, W]
        concat1 = torch.cat((error, left_img), dim=1)  # [B, 6, H, W]
        conv1 = self.conv1(concat1)  # [B, 16, H, W]
        conv2 = self.conv2(disp)  # [B, 16, H, W]
        x = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x

        rem3 = self.attn(rem3)
        
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        residual_disp = self.final_conv(x)  # [B, 1, H, W]

        disp = F.relu(disp + residual_disp, inplace=True)  # [B, 1, H, W]
        # disp = disp.squeeze(1)  # [B, H, W]

        return disp            

class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_channel, head_number):
        super(MultiheadSelfAttention, self).__init__()
        self.head_number = head_number
        self.input_channel = input_channel
        self.head_dim = input_channel // head_number

        self.query_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)
        self.key_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)
        self.value_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)

        self.out_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Split the input into multiple heads
        query = self.query_linear(x).view(batch_size, self.head_number, self.head_dim, height, width).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, self.head_number, self.head_dim, height, width).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, self.head_number, self.head_dim, height, width).transpose(1, 2)

        # Scale the query
        query = query / (self.head_dim ** (1/4))

        # Compute the attention weights
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply the attention weights to the value
        attn_output = torch.matmul(attn_weights, value)

        # Merge the heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.input_channel, height, width)

        # Apply the output linear layer
        output = self.out_linear(attn_output)
        return output