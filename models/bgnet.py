# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function
from models.feature_extractor_fast import feature_extraction
from models.submodules3d import CoeffsPredictor
from models.submodules2d import HourglassRefinement
from models.submodules import SubModule, convbn_2d_lrelu, convbn_3d_lrelu,convbn_2d_Tanh
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
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # N x 1 x H x W x 3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide,align_corners =False)
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
        # [B, 44, 25, H/8, W/8]
        cost_volume = build_gwc_volume(left_gwc_feature, right_gwc_feature, 25, 44)
        
        # Fig.2: 3D convolution
        cost_volume = self.dres0(cost_volume)
        
        # coeffs:[B,D,G,H,W]
        # [B, 25, 44, H/8, W/8]
        # HourGlass: 3D conv and 3D deconv
        coeffs = self.coeffs_disparity_predictor(cost_volume)
        # 分割 coeffs 为 25 个视差层，maxdisp = 25
        list_coeffs = torch.split(coeffs,1,dim = 1)
        
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
        device = list_coeffs[0].get_device()
        # reshape wa, wb to (1,n,1,1)
        wa  = wa.view(1,-1,1,1)
        wb  = wb.view(1,-1,1,1)
        wa = wa.to(device)
        wb = wb.to(device)
        wa = wa.float()
        wb = wb.float()
        # [B, 1, H/2, W/2]
        N, _, H, W = guide.shape
        # hg和wg的形状都是(H, W)
        # 在hg中，每一行的所有元素都是相同的，表示行坐标
        # 在wg中，每一列的所有元素都是相同的，表示列坐标
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        #[B,H,W,1]
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
        slice_dict = []
        # torch.cuda.synchronize()
        # start = time.time()
        for i in range(25):
            slice_dict.append(self.slice(list_coeffs[i], wg, hg, guide)) #[B,1,H,W]
        slice_dict_a = []
        slice_dict_b = []
        for i in range(97):
            inx_a = i//4
            inx_b = inx_a + 1
            inx_b  = min(inx_b,24)
            slice_dict_a.append(slice_dict[inx_a])
            slice_dict_b.append(slice_dict[inx_b])
        
        # 把0-24的视差范围扩展到0-96，总视差范围应该是192，所以这是一半的特征。不过他的扩展方式很奇怪，给我的感觉就是把25个视差维度特征混合了一下，拼成了97维
        final_cost_volume = wa * torch.cat(slice_dict_a,dim = 1) + wb * torch.cat(slice_dict_b,dim = 1)
        slice = self.softmax(final_cost_volume)
        disparity_samples = torch.arange(0, 97, dtype=slice.dtype, device=slice.device).view(1, 97, 1, 1)
        # 把经过代价聚合过程的每个视差等级的feature maps通道压缩成1，每个视差等级只有一个feature map,softmax为每个视差等级都计算一个概率，
        # 每个像素的视差d乘以改层视差等级的概率，累加得出最后该像素的视差值，从而生成视差图
        
        # 最终在[H/2,W/2]的尺寸上预测了0-96的视差概率分布，得到了每个点的预测视差half_disp，然后对2*half_disp进行双线性插值，得到[H,W]尺寸上的每点预测视差
        disparity_samples = disparity_samples.repeat(slice.size()[0],1,slice.size()[2],slice.size()[3])
        half_disp = torch.sum(disparity_samples * slice,dim = 1).unsqueeze(1)
        out2 = F.interpolate(half_disp * 2.0, scale_factor=(2.0, 2.0),
                                      mode='bilinear',align_corners =False).squeeze(1)
                                            
        return out2,out2


