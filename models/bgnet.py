# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function
from models.feature_extractor_fast import feature_extraction
from models.submodules3d import CoeffsPredictor
from models.submodules2d import HourglassRefinement
from models.submodules import SubModule, convbn_2d_lrelu, convbn_3d_lrelu,convbn_2d_Tanh,convbn_3d
from nets.warp import disp_warp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# 从双边网格中提取特征
# 首先对输入的 guidemap 进行了置换和连续化处理，
# 然后使用 torch.cat 和 unsqueeze 对 guidemap 进行了扩展，
# 最后使用 F.grid_sample 对双边网格进行了采样
class Slice(SubModule):
    def __init__(self):
        super(Slice, self).__init__()
    def forward(self, bilateral_grid, wg, hg, guidemap): 
        guidemap = guidemap.permute(0,2,3,1).contiguous() #[B,C,H,W]-> [B,H,W,C]
        # [N, H/2, W/2, 1] [N, H/2, W/2, 1] [N, H/2, W/2, 1]
        # gudiemap_guide tensor([[[[wg,hg,guidemap], ..., ...]]], [B, H, W, C+C+C]
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # N x 1 x H x W x 3
        """
        https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        https://www.paddlepaddle.org.cn/documentation/docs/zh///api/paddle/nn/functional/grid_sample_cn.html
        (function) def grid_sample(
                        input: Tensor,
                        grid: Tensor,
                        mode: str = ...,
                        padding_mode: str = ...,
                        align_corners: Any | None = ...
                    ) -> Tensor
        根据 grid[n, h, w](二维向量, 其值为 x 坐标和 y 坐标) 在 input 中进行 bilinear interpolation
        提供一个 input 的 Tensor 以及一个对应的 flow-field grid (比如光流，体素流等)
        然后根据 grid 中每个位置提供的坐标信息, (这里指input中pixel的坐标)
        将 input 中对应位置的像素值填充到grid指定的位置, 得到最终的输出
        grid[n, h, w] 是一个二维向量，它用于指定输入图像中的像素位置 x 和 y. 这些位置将被用来插值输出值 output[n, :, h, w]。
        在4D输入的情况下, grid 的形状为 (N, H_out, W_out, 2)，其中：
            N 是批次大小(batch size)
            H_out 和 W_out 是输出图像的高度和宽度。
            2 表示二维向量，其中第一个元素是 x 坐标，第二个元素是 y 坐标。
        在5D输入的情况下, grid 的形状为 (N, D_out, H_out, W_out, 3)，其中：
            D_out 是输出图像的深度(depth)。
            3 表示三维向量，其中前两个元素是 x 和 y 坐标，第三个元素是 z 坐标。
        Q: 插值计算是在输入张量中计算的吗?
        1) 插值计算是在输入张量中进行的。torch.nn.functional.grid_sample 函数使用输入值和来自网格的像素位置来计算输出。
            这些像素位置由网格中的 (x, y) 坐标指定，用于插值输出值。因此，插值计算是在输入张量的像素上进行的，以生成输出。
        Q: grid提供了什么? 坐标吗?
        2) grid 提供了坐标信息。grid 中的每个 (x, y) 坐标指定了输入像素的位置，用于插值输出值。
            这些坐标是归一化的采样像素位置，范围应在 [-1, 1] 内。例如, x = -1, y = -1 是输入的左上角像素, x = 1, y = 1 是输入的右下角像素
        Q: 从最终结果来看,grid_sample实现什么?将输入tensor扭曲为grid的样子?
        3) 实现目的是将输入张量按照网格的样式进行扭曲。具体来说：
            首先，你提供了一个输入张量，它是你想要进行扭曲的原始数据。
            然后，你提供了一个网格，其中的每个 (x, y) 坐标指定了输入像素的位置，用于插值输出值。
            grid_sample 函数会根据网格中的坐标信息，对输入张量进行插值，生成输出。
            这个过程类似于将输入张量的像素位置按照网格的指导进行变换，从而实现了扭曲的效果。
        Q: grid[n, h, w]是如何得到x和y坐标的?
        4) grid 是一个网格，其形状为 (N, H_out, W_out, 2)(4D情况)或 (N, D_out, H_out, W_out, 3)(5D情况)。
            对于每个输出位置 output[n, :, h, w], grid[n, h, w] 是一个大小为2的向量, 其中的两个值分别表示 x 和 y 坐标。
            这些坐标是归一化的采样像素位置，范围应在 [-1, 1] 内。例如：
            x = -1, y = -1 对应输入的左上角像素。
            x = 1, y = 1 对应输入的右下角像素。
            因此, grid 中的 (x, y) 坐标指定了输入像素的位置，用于插值输出值 output[n, :, h, w]
        """
        # [N, 1, 44, H/8, W/8] [N, 1, H/2, W/2, 3]
        coeff = F.grid_sample(bilateral_grid, guidemap_guide,align_corners = False)
        return coeff.squeeze(2) #[B,1,H,W]
# 32 channels -> 16 channels -> 1 channel
class GuideNN(SubModule):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = convbn_2d_lrelu(32, 16, 1, 1, 0)   # 引入非线性
        self.conv2 = convbn_2d_Tanh(16, 1, 1, 1, 0)     # 限制在[-1,1]之间

    def forward(self, x):
        return self.conv2(self.conv1(x))
# 计算 correlation
def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

# 构建 cost volume
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    #[B,G,D,H,W]
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
def correlation(fea1, fea2):
    B, C, H, W = fea1.shape
    cost = (fea1 * fea2).mean(dim=1)
    assert cost.shape == (B, H, W)
    return cost
# 视差回归
def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp + 1, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp + 1, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

class BGNet(SubModule):
    def __init__(self):
        super(BGNet, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        # self.refinement_net = HourglassRefinement()
      
        self.feature_extraction = feature_extraction()
        self.coeffs_disparity_predictor = CoeffsPredictor()

        self.dres0 = nn.Sequential(convbn_3d_lrelu(44, 32, 3, 1, 1),
                                   convbn_3d_lrelu(32, 16, 3, 1, 1))
        # self.dres0 = hourglass(44)
        self.guide = GuideNN()
        self.slice = Slice()

        self.weight_init()

    def forward(self, left_input, right_input):
        # Fig.2: Feature extraction
        left_low_level_features_1, left_gwc_feature  = self.feature_extraction(left_input)
        _,                         right_gwc_feature = self.feature_extraction(right_input)
        # [Batch size, Channels , Height, Weight]
        # left_low_level_features_1: [B, 32, H/2, W/2]
        # left_gwc_feature: [B, 352, H/8, W/8]
        
        # Fig.2: Guidance map
        guide = self.guide(left_low_level_features_1) # [B, 32, H/2, W/2] -> [B, 1, H/2, W/2]
        
        # torch.cuda.synchronize()
        # start = time.time()
        
        #  构建 cost volume: refimg_fea, targetimg_fea, maxdisp, num_groups
        cost_volume = build_gwc_volume(left_gwc_feature, right_gwc_feature, 25, 44) # [B, 44, 25, H/8, W/8]
        # Fig.2: 3D convolution
        cost_volume = self.dres0(cost_volume)   # [B, 16, 25, H/8, W/8]        
        # coeffs: [B,D,G,H,W]
        # [B, 25, 44, H/8, W/8]
        # HourGlass: 3D conv and 3D deconv
        coeffs = self.coeffs_disparity_predictor(cost_volume)
        # 分割 coeffs 为 25 个视差层，maxdisp = 25
        # list_coeffs是一个包含了多个张量的列表
        # 每个张量都是coeffs的一个子张量，形状为[B, 1, G, H, W]
        list_coeffs = torch.split(coeffs,1,dim = 1)
        device = list_coeffs[0].get_device()
        # [B, 1, H/2, W/2]
        N, _, H, W = guide.shape
        # hg和wg的形状都是(H, W)
        # 在hg中，每一行的所有元素都是相同的，表示行坐标
        # 在wg中，每一列的所有元素都是相同的，表示列坐标
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1, [B,H,W,1]
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        slice_dict = []
        # torch.cuda.synchronize()
        # start = time.time()
        # [B, 1, G, H/8, W/8] [B, H/2, W/2, 1] [B, H/2, W/2, 1] [B, 1, H/2, W/2]
        # guide 每一次 G(x, y) 是相同的, 但是 bilaterial grid 是不同的
        #? 进行三次双线性插值/三线性插值
        #? 与双边滤波的区别是什么?为什么要命名为双边网格
        for i in range(25):
            slice_dict.append(self.slice(list_coeffs[i], wg, hg, guide)) #[B,1,H,W]
        slice_dict_a = []
        slice_dict_b = []
        
        #todo: 理解这个trick
        # wa,wb 可以将两个相邻的视差维度的特征进行线性插值，从而获得更平滑的成本体积
        index = torch.arange(0,97)  # tensor([ 0 ,..., 96])
        index_float = index/4.0
        index_a = torch.floor(index_float)
        index_b = index_a + 1
        # 限制 index_a, index_b 的范围在 [0,24] 之间
        index_a = torch.clamp(index_a, min=0, max= 24)
        index_b = torch.clamp(index_b, min=0, max= 24)
        
        wa = index_b - index_float
        wb = index_float - index_a

        list_float = []  
        
        # reshape wa, wb to (1,n,1,1)
        # torch.Size([1, 97, 1, 1])
        wa  = wa.view(1,-1,1,1)
        wb  = wb.view(1,-1,1,1)
        wa = wa.to(device)
        wb = wb.to(device)
        wa = wa.float()
        wb = wb.float()
        
        # 把0-24的视差范围扩展到0-96，总视差范围应该是192，所以这是一半的特征。不过他的扩展方式很奇怪，给我的感觉就是把25个视差维度特征混合了一下，拼成了97维
        for i in range(97):
            inx_a = i//4
            inx_b = inx_a + 1
            inx_b  = min(inx_b,24)
            slice_dict_a.append(slice_dict[inx_a])
            slice_dict_b.append(slice_dict[inx_b])
        # 将 0-24 恢复为 0-96
        final_cost_volume = wa * torch.cat(slice_dict_a,dim = 1) + wb * torch.cat(slice_dict_b,dim = 1)
        slice = self.softmax(final_cost_volume)
        disparity_samples = torch.arange(0, 97, dtype=slice.dtype, device=slice.device).view(1, 97, 1, 1)
        # 把经过代价聚合过程的每个视差等级的 feature maps 通道压缩成1
        # 每个视差等级只有一个 feature map, softmax 为每个视差等级都计算一个概率
        # 每个像素的视差 d 乘以改层视差等级的概率，累加得出最后该像素的视差值，从而生成视差图
        # 最终在 [H/2,W/2] 的尺寸上预测了 0-96 的视差概率分布，得到了每个点的预测视差 half_disp
        # 然后对 2*half_disp 进行双线性插值，得到[H,W]尺寸上的每点预测视差
        disparity_samples = disparity_samples.repeat(slice.size()[0], 1, slice.size()[2], slice.size()[3])
        # sum(d * softmax(C_H))
        half_disp = torch.sum(disparity_samples * slice,dim = 1).unsqueeze(1)
        out2 = F.interpolate(half_disp * 2.0, scale_factor=(2.0, 2.0),
                                      mode='bilinear',align_corners =False).squeeze(1)
                                            
        return out2,out2


class attention_block(nn.Module):
    def __init__(self, channels_3d, num_heads=8, block=4):
        """
        ws 1 for stand attention
        """
        super(attention_block, self).__init__()
        self.block = block
        self.dim_3d = channels_3d
        self.num_heads = num_heads
        head_dim_3d = self.dim_3d // num_heads  # 每个head的维度
        self.scale_3d = head_dim_3d ** -0.5 # 缩放因子
        # 全连接层可以将输入数据的特征数从 self.dim_3d 扩大到 self.dim_3d 的3倍
        # 以计算查询、键和值这三个部分
        self.qkv_3d = nn.Linear(self.dim_3d, self.dim_3d * 3, bias=True)
        self.final1x1 = torch.nn.Conv3d(self.dim_3d, self.dim_3d, 1)


    def forward(self, x):
        # pad input to be a multiple of block
        B, C, D, H0, W0 = x.shape
        pad_l = pad_t = 0
        pad_r = (self.block[2] - W0 % self.block[2]) % self.block[2]
        pad_b = (self.block[1] - H0 % self.block[1]) % self.block[1]
        # pad zeros at edge to make the input shape to be a multiple of block
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        B, C, D, H, W = x.shape
        d, h, w = D//self.block[0], H//self.block[1], W//self.block[2]
        # x: [B, C, D, H, W] -> [B, d, h, w, self.block[0], self.block[1], self.block[2], C]
        x = x.view(B, C, d,self.block[0], h, self.block[1], w, self.block[2]).permute(0, 2, 4, 6, 3, 5, 7, 1)
        # self.qkv_3d(x) 线性变换，将输入数据的特征数从 self.dim_3d 扩大到 self.dim_3d 的3倍
        # C // self.num_heads 是每个注意力头的通道数
        # (3, B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], C // self.numheads)
        # reshape: tensor 一维展开后再重新组织
        # permute: tensor 维度换位
        qkv_3d = self.qkv_3d(x).reshape(B, d*h*w, self.block[0]*self.block[1]*self.block[2], 3, self.num_heads,
                                            C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # assign q, k, v
        # (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], C // self.numheads)
        q_3d, k_3d, v_3d = qkv_3d[0], qkv_3d[1], qkv_3d[2]
        # 计算Query(q_3d) 和 Key(k_3d)相似性
        # @: torch.matmul
        # 要求第一个矩阵的列数（最后一个维度）等于第二个矩阵的行数（倒数第二个维度）
        # a: (2, 3, 4) , b: (4, 5)
        # (2, 3, 4) @ (4, 5) -> (2, 3, 5)
        # 使用广播机制将 b 扩充为 (2, 4, 5)
        # 对每个(3, 4) 和 (4, 5)做矩阵乘法，得到(3, 5)的矩阵
        # 然后把两个(3, 5)的矩阵拼接起来，得到(2, 3, 5)的矩阵
        # (a,b,c,d,e,f) @ (a,b,c,d,f,e) -> (a,b,c,d,e,e)
        # 转置最后和倒数第二个维度后进行矩阵乘法，得到注意力权重系数
        # q_3d: (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], C // self.numheads) 
        # k_3d: (B, d*h*w, self.numheads, C // self.numheads, self.block[0]*self.block[1]*self.block[2]) 
        # attn: (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], self.block[0]*self.block[1]*self.block[2])
        attn = (q_3d @ k_3d.transpose(-2, -1)) * self.scale_3d
        if pad_r > 0 or pad_b > 0:
            mask = torch.zeros((1, H, W), device=x.device)
            # 最后 pad_b 行和所有的列填充为 1
            mask[:, -pad_b:, :].fill_(1)
            # 所有的行的最后 pad_r 列填充为 1
            mask[:, :, -pad_r:].fill_(1)
            mask = mask.reshape(1, h, self.block[1], w, self.block[2]).transpose(2, 3).reshape(1,  h*w, self.block[1]*self.block[2])
            # unsqueeze: 在指定位置增加一个维度
            attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)  # 1, _h*_w, self.block*self.block, self.block*self.block
            #  -1000.0 在 softmax 操作后的值接近于 0, 有效地屏蔽掉某些位置
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
            # 在第二维上重复 d 次, 在第三维上重复 self.block[0] 次, 在第四维上重复 self.block[0] 次
            attn = attn + attn_mask.repeat(1, d, self.block[0], self.block[0]).unsqueeze(2)
        # 对 atten 进行归一化处理
        attn = torch.softmax(attn, dim=-1)
        # 根据权重系数对 Value(v_3d) 进行加权求和
        # 将 attn 张量的每一行（最后一个维度，因为 dim=-1）
        # 转换为一个概率分布，每一行的所有元素的和都为 1
        # x -> (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], self.block[0]*self.block[1]*self.block[2], C // self.numheads)
        # x -> (B, d, h ,w, self.num_heads, self.block[0], self.block[1], self.block[2], -1)
        # x -> (B, self.num_heads, -1, d, self.block[0], h, self.block[1], w, self.block[2])
        # tensor 中的 -1 表示自动计算该维度的大小
        x = (attn @ v_3d).view(B, d, h ,w, self.num_heads, self.block[0], self.block[1], self.block[2], -1).permute(0,4,8,1,5,2,6,3,7)
        # 注意到: d, h, w = D//self.block[0], H//self.block[1], W//self.block[2]
        x = x.reshape(B, C, D, H, W)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :, :H0, :W0]
        return self.final1x1(x)

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        #! attention block
        # 16 heads
        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6