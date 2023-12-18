from __future__ import print_function, division
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from datasets import __datasets__
import gc
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
parser = argparse.ArgumentParser(description='BGNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='bgnet', help='select a model structure')
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/disk2/users/M22_zhaoqinghao/dataset/sceneflow/',
                    help='datapath')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--resume', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
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

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 12, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)
if(args.model == 'bgnet'):
    model = BGNet().cuda()
elif(args.model == 'bgnet_plus'):
    model = BGNet_Plus().cuda()


if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

def train(imgL,imgR, disp_L):
        model.train()

        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        
        output,_ = model(imgL,imgR)
        # output = torch.squeeze(output,1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

        loss.backward()
        optimizer.step()

        return loss.data

def test(imgL,imgR,disp_true):

        model.eval()
  
        if args.cuda:
            imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
        #---------
        mask = disp_true < 192
        #----

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            right_pad = (times+1)*16-imgL.shape[3]
        else:
            right_pad = 0  

        imgL = F.pad(imgL,(0,right_pad, top_pad,0))
        imgR = F.pad(imgR,(0,right_pad, top_pad,0))

        with torch.no_grad():
            output3 = model(imgL,imgR)
            output3 = torch.squeeze(output3)
        
        if top_pad !=0:
            img = output3[:,top_pad:,:]
        else:
            img = output3

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

        return loss.data.cpu()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.0005
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop,imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/(1e-8+len(TrainImgLoader))))

        #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/(1e-8+len(TrainImgLoader)),
        }, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    #------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL,imgR, disp_L)
        print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
        total_test_loss += test_loss

    print('total test loss = %.3f' %(total_test_loss/(len(TestImgLoader)+1e-8)))
    #----------------------------------------------------------------------------------
    #SAVE test information
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
        'test_loss': total_test_loss/(1e-8+len(TestImgLoader)),
    }, savefilename)


if __name__ == '__main__':
   main()
    
