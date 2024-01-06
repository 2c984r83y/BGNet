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
parser.add_argument('--dataset', default='dsec_png', help='dataset name', choices=__datasets__.keys())
# parser.add_argument('--datapath', default='/root/KITTI_2015/',help='datapath')
# parser.add_argument('--savepath', default='/root/BGNet/output/', help='save path')
# parser.add_argument('--trainlist', default='/root/BGNet/filenames/kitti15_train.txt', help='training list')
# parser.add_argument('--testlist', default='/root/BGNet/filenames/KITTI-15-Test.txt', help='testing list')
# parser.add_argument('--loadmodel', default= '/root/BGNet/models/kitti_15_BGNet_Plus.pth',
#                     help='load model')

parser.add_argument('--datapath', default='/home/zhaoqinghao/dataset/DSEC/output',
                    help='datapath')
parser.add_argument('--savepath', default='/disk2/users/M22_zhaoqinghao/BGNet/output/', 
                    help='save path')
parser.add_argument('--trainlist', default='/home/zhaoqinghao/DSEC/train.txt', 
                    help='training list')
parser.add_argument('--testlist', default='/home/zhaoqinghao/DSEC/test.txt', 
                    help='testing list')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
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

dsec_train = args.trainlist
dsec_train_dataset = StereoDataset(datapath, dsec_train, True)
TrainImgLoader = DataLoader(dsec_train_dataset, batch_size= 16, shuffle=True, num_workers=16, drop_last=False)

dsec_test = args.testlist
dsec_test_dataset = StereoDataset(datapath, dsec_test, False)
TestImgLoader = DataLoader(dsec_test_dataset, batch_size= 8, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'bgnet':
    model = BGNet().cuda()
elif args.model == 'bgnet_plus':
    model = BGNet_Plus().cuda()

if args.cuda:
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel, map_location=torch.device('cuda' if args.cuda else 'cpu'))
    model.load_state_dict(state_dict.get('state_dict', {}), strict=False)
    print('load model')
    print(args.loadmodel)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

def train(imgL, imgR, disp_L):
    model.train()
    
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    optimizer.zero_grad()

    output, _ = model(imgL, imgR)

    disp_true = torch.squeeze(disp_true, 0)
    disp_true = torch.squeeze(disp_true, 1)

    mask = (disp_true > 0)
    mask.detach_()
    
    loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data

def test(imgL,imgR,disp_true):
    model.eval()

    with torch.no_grad():
        pred,_ = model(imgL.cuda(), imgR.cuda())
    disp_true = torch.squeeze(disp_true, 1)
    disp_true = torch.squeeze(disp_true, 1)
    pred_disp = pred.data.cpu()
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp > 0)
    disp_true = disp_true.float()  
    # print(pred.shape)
    # print(disp_true.shape)
    disp_true[index[0][:], index[1][:], index[2][:]] = torch.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
                disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
            index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()
    acc = (float(torch.sum(correct))/float(len(index[0])))
    return acc

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
       lr = 0.001
    else:
       lr = 0.0001
    print('learning rate = %.6f' %(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()
    
    for epoch in range(1, args.epochs+1):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)
        print('epoch %d' %(epoch))
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            loss = train(imgL.cuda(), imgR.cuda(), disp_L.cuda())
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
    
        if epoch % 10 == 0:
            savefilename = args.savemodel+'checkpoint_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), savefilename)
    
    total_test_loss = 0
    for batch_idx, sample in enumerate(TestImgLoader):
        imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
        test_loss = test(imgL, imgR, disp_L)
        print('Iter %d 3-px Accuracy in val = %.3f' %(batch_idx, test_loss*100))
        total_test_loss += test_loss

    print('epoch %d total 3-px Accuracy in val = %.3f' %(epoch, total_test_loss/len(TestImgLoader)*100))
    
    if total_test_loss/len(TestImgLoader)*100 > max_acc:
        max_acc = total_test_loss/len(TestImgLoader)*100
        max_epo = epoch
    
    print('MAX epoch %d total test Accuracy = %.3f' %(max_epo, max_acc))


    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)

if __name__ == '__main__':
   main()
