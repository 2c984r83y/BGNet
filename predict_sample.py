# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function, division
import argparse
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import time
from datasets import __datasets__
import gc
import skimage
import skimage.io
import skimage.transform
import numpy as np
from PIL import Image
from datasets.data_io import get_transform
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
import time

model = BGNet_Plus().cuda()

# checkpoint = torch.load('models/kitti_15_BGNet_Plus.pth',map_location=lambda storage, loc: storage)
checkpoint = torch.load('./finetune_30_dsec.pth',map_location=lambda storage, loc: storage)

model.load_state_dict(checkpoint) 
model.eval()
# left_img = Image.open('sample/im0.png').convert('L')
# right_img = Image.open('sample/im1.png').convert('L')

left_img = Image.open('/home/zhaoqinghao/DSEC/output/left/000233.png').convert('L')
right_img = Image.open('/home/zhaoqinghao/DSEC/output/right/000233.png').convert('L')

# left_img = Image.open('/root/KITTI_2015/testing/image_2/000001_10.png').convert('L')
# right_img = Image.open('/root/KITTI_2015/testing/image_3/000001_10.png').convert('L')
w, h = left_img.size
h1 = h % 64
w1 = w % 64
h1 = h  - h1
w1 =  w - w1
h1 = int(h1)
w1 = int(w1)
# left_img = left_img.resize((w1, h1),Image.ANTIALIAS)
# right_img = right_img.resize((w1, h1),Image.ANTIALIAS)
left_img = left_img.resize((w1, h1),Image.Resampling.LANCZOS)   # Resize using Lanczos resampling
right_img = right_img.resize((w1, h1),Image.Resampling.LANCZOS)
# 使用 NumPy 库的 `ascontiguousarray()` 函数将图像数据转换为连续的数组。
# 提高数据处理的效率，因为许多 NumPy 的操作在连续的数组上会更快。
# `dtype=np.float32` 是 `ascontiguousarray()` 函数的一个参数，它定义了新数组的数据类型。
# 将图像数据的数据类型转换为 `float32`，这是一种单精度浮点数类型，它可以有效地减少内存的使用，同时保持足够的精度。
left_img = np.ascontiguousarray(left_img, dtype=np.float32)
right_img = np.ascontiguousarray(right_img, dtype=np.float32)
preprocess = get_transform()    # get_transform()函数返回一个转换列表，它将图像转换为 PyTorch 张量
left_img = preprocess(left_img)
right_img = preprocess(right_img)
# GPU dry run
# create dummy tensor 
imgL = torch.from_numpy(np.zeros_like(left_img))
imgR = torch.from_numpy(np.zeros_like(right_img))
with torch.no_grad():
    pred,_ = model(imgL.unsqueeze(0).cuda(), imgR.unsqueeze(0).cuda()) 
# start timing
time_start=time.time()
# real run, using no_grad() to reduce memory usage and speed up
# unsqueeze(0) 将在第一个维度（索引为 0）上增加一个维度
with torch.no_grad():
    pred,_ = model(left_img.unsqueeze(0).cuda(), right_img.unsqueeze(0).cuda())
# print time cost
print('time cost: ',(time.time()-time_start)*1000,'ms')
print('FPS: ',1/(time.time()-time_start))
pred = pred[0].data.cpu().numpy() * 256   
skimage.io.imsave('sample_disp.png',pred.astype('uint16'))
