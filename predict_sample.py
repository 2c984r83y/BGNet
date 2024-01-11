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
import copy

model = BGNet_Plus().cuda()
checkpoint = torch.load('./pretrained_models/checkpoint_230.pth',map_location=lambda storage, loc: storage)
# checkpoint = torch.load('./pretrained_models/Sceneflow-IRS-BGNet-Plus.pth',map_location=lambda storage, loc: storage)

# checkpoint = torch.load('./finetune_30_dsec.pth',map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint) 
model.eval()
# left_img = Image.open('sample/im0.png').convert('L')
# right_img = Image.open('sample/im1.png').convert('L')

# left_img = Image.open('/home/zhaoqinghao/DSEC/output/left/000233.png').convert('L')
# right_img = Image.open('/home/zhaoqinghao/DSEC/output/right/000233.png').convert('L')
# disp_true = Image.open('/home/zhaoqinghao/DSEC/output/disp/000233.png').convert('L')
left_img = Image.open('/home/zhaoqinghao/dataset/KITTI_2015/training/image_2/000001_10.png').convert('L')
right_img = Image.open('/home/zhaoqinghao/dataset/KITTI_2015/training/image_3/000001_10.png').convert('L')
disp_true = Image.open('/home/zhaoqinghao/dataset/KITTI_2015/training/disp_occ_0/000001_10.png').convert('L')
w, h = left_img.size
print(w, h)
h1 = h % 64
w1 = w % 64
print(w1,h1)
h1 = h - h1
w1 = w - w1
h1 = int(h1)
w1 = int(w1)
# left_img = left_img.resize((w1, h1),Image.ANTIALIAS)
# right_img = right_img.resize((w1, h1),Image.ANTIALIAS)
left_img = left_img.resize((w1, h1),Image.Resampling.LANCZOS)   # Resize using Lanczos resampling
right_img = right_img.resize((w1, h1),Image.Resampling.LANCZOS)
disp_true = disp_true.resize((w1, h1),Image.Resampling.LANCZOS)
left_img = np.ascontiguousarray(left_img, dtype=np.float32)
right_img = np.ascontiguousarray(right_img, dtype=np.float32)
disp_true = np.ascontiguousarray(disp_true, dtype=np.float32)
preprocess = get_transform()    # get_transform()函数返回一个转换列表，它将图像转换为 PyTorch 张量
left_img = preprocess(left_img)
right_img = preprocess(right_img)
disp_true = preprocess(disp_true)
# GPU dry run
# create dummy tensor 
imgL = torch.from_numpy(np.zeros_like(left_img))
imgR = torch.from_numpy(np.zeros_like(right_img))
with torch.no_grad():
    pred,_ = model(imgL.unsqueeze(0).cuda(), imgR.unsqueeze(0).cuda()) 
# start timing
time_start=time.time()
with torch.no_grad():
    pred,_ = model(left_img.unsqueeze(0).cuda(), right_img.unsqueeze(0).cuda())
# print time cost
print('time cost: ',(time.time()-time_start)*1000,'ms')
print('FPS: ',1/(time.time()-time_start))

pred_disp = pred.data.cpu()
# pred_disp = copy.deepcopy(disp_true)
true_disp = copy.deepcopy(disp_true)
index = np.argwhere(true_disp > 0)
disp_true = disp_true.float()  # Convert disp_true to float data type
disp_true[index[0][:], index[1][:], index[2][:]] = torch.abs(true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]] * 0.05)
torch.cuda.empty_cache()
acc = (float(torch.sum(correct))/float(len(index[0])))
print('3-px acc: ',acc*100,'%')
# save disp
pred = pred[0].data.cpu().numpy() * 256
skimage.io.imsave('sample_disp.png',pred.astype('uint16'))