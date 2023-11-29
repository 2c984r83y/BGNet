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

checkpoint = torch.load('models/Sceneflow-IRS-BGNet-Plus.pth',map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint) 
model.eval()
# left_img1 = Image.open('sample/im0.png').convert('L')
# right_img = Image.open('sample/im1.png').convert('L')
left_img = Image.open('/disk2/users/M22_zhaoqinghao/PSMNet/testing/desc_l_1.png').convert('L')
right_img = Image.open('/disk2/users/M22_zhaoqinghao/PSMNet/testing/desc_r_1.png').convert('L')
w, h = left_img.size
h1 = h % 64
w1 = w % 64
h1 = h  - h1
w1 =  w - w1
h1 = int(h1)
w1 = int(w1)
# left_img = left_img.resize((w1, h1),Image.ANTIALIAS)
# right_img = right_img.resize((w1, h1),Image.ANTIALIAS)
left_img = left_img.resize((w1, h1),Image.Resampling.LANCZOS)
right_img = right_img.resize((w1, h1),Image.Resampling.LANCZOS)
left_img = np.ascontiguousarray(left_img, dtype=np.float32)
right_img = np.ascontiguousarray(right_img, dtype=np.float32)
preprocess = get_transform()
left_img = preprocess(left_img)
right_img = preprocess(right_img)
# GPU dry run
imgL = torch.from_numpy(np.zeros_like(left_img))
imgR = torch.from_numpy(np.zeros_like(right_img))
with torch.no_grad():
    pred,_ = model(imgL.unsqueeze(0).cuda(), imgR.unsqueeze(0).cuda()) 
# start timing
time_start=time.time()
# real run, using no_grad() to reduce memory usage and speed up
with torch.no_grad():
    pred,_ = model(left_img.unsqueeze(0).cuda(), right_img.unsqueeze(0).cuda())
# print time cost
print('time cost: ',time.time()-time_start,'s')
pred = pred[0].data.cpu().numpy() * 256   
skimage.io.imsave('sample_disp.png',pred.astype('uint16'))
