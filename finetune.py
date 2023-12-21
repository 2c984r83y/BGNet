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
from PIL import Image
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import math
import copy

import sys
sys.path.append('/root/BGNet/models')
from models import *
from PIL import Image
from datasets.data_io import get_transform

parser = argparse.ArgumentParser(description='BGNet')
parser.add_argument('--model', default='bgnet_plus', help='select a model structure')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/root/KITTI_2015/',help='datapath')
parser.add_argument('--savepath', default='/root/BGNet/output/', help='save path')
parser.add_argument('--testlist', default='/root/BGNet/filenames/kitti15_train.txt', help='testing list')
# parser.add_argument('--resume', default='/root/BGNet/models/kitti_15_BGNet_Plus.pth', help='the directory to save logs and checkpoints')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= '/root/BGNet/models/kitti_15_BGNet_Plus.pth',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

datapath = args.datapath
StereoDataset = __datasets__[args.dataset]
kitti_real_test = args.testlist
kitti_real_test_dataset = StereoDataset(datapath, kitti_real_test, False)
TestImgLoader = DataLoader(kitti_real_test_dataset, batch_size= 1, shuffle=False, num_workers=1, drop_last=False)

if(args.model == 'bgnet'):
    model = BGNet().cuda()
elif(args.model == 'bgnet_plus'):
    model = BGNet_Plus().cuda()
# else:
    # print('wrong model')
    # return -1
sub_name = None    
if(args.dataset == 'kitti_12'):
    sub_name = 'testing/colored_0/'
elif(args.dataset == 'kitti'):
    sub_name = 'testing/image_2/'
# else:
    # print('wrong dataset')
    # return -1
    
if args.cuda:
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    model.load_state_dict(state_dict.get('state_dict', {}), strict=False)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

# checkpoint = torch.load(args.resume,map_location=lambda storage, loc: storage)
# model.load_state_dict(checkpoint) 

def train(imgL,imgR,disp_L):
    model.train()
    
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()


    optimizer.zero_grad()

    output,_ = model(imgL,imgR)

    disp_true = torch.squeeze(disp_true,0)
    disp_true = torch.squeeze(disp_true,0)
    output = torch.squeeze(output,0)
    # print(disp_true.shape)
    # print(output.shape)
        #---------
    mask = (disp_true > 0)
    mask.detach_()
    #----
    
    loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)
    # loss = F.smooth_l1_loss(output, disp_true, size_average=True)

    loss.backward()
    optimizer.step()

    # return loss.data[0]
    return loss.data

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
       lr = 0.001
    else:
       lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    max_acc=0
    max_epo=0
    start_full_time = time.time()

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer,epoch)
        ## training ##
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            imgL, imgR, disp_L= sample['left'], sample['right'], sample['disparity']
            imgL = imgL.cuda()
            imgR = imgR.cuda()
            disp_L = disp_L.cuda()
            loss = train(imgL, imgR, disp_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
        total_train_loss += loss
    print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TestImgLoader)))

    ## Test ##
    #TODO:fix this
#     for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
#         test_loss = test(imgL,imgR, disp_L)
#         print('Iter %d 3-px Accuracy in val = %.3f' %(batch_idx, test_loss*100))
#         total_test_loss += test_loss


#     print('epoch %d total 3-px Accuracy in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
#     if total_test_loss/len(TestImgLoader)*100 > max_acc:
#         max_acc = total_test_loss/len(TestImgLoader)*100
#         max_epo = epoch
#     print('MAX epoch %d total test Accuracy = %.3f' %(max_epo, max_acc))

    # save as pth
    savefilename = args.savemodel+'finetune_'+str(epoch)+'.pth'
    torch.save(model.state_dict(), savefilename)
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    # print(max_epo)
    # print(max_acc)


if __name__ == '__main__':
   main()

