from __future__ import print_function, division
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch.utils.data
import torch.nn.functional as F
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
checkpoint = torch.load('./pretrained_models/checkpoint_230.pth',map_location=lambda storage, loc: storage)
# checkpoint = torch.load('./pretrained_models/Sceneflow-IRS-BGNet-Plus.pth',map_location=lambda storage, loc: storage)
# checkpoint = torch.load('./finetune_30_dsec.pth',map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint) 
model.eval()

left_img = Image.open('/home/zhaoqinghao/DSEC/output/left/000167.png').convert('L')
right_img = Image.open('/home/zhaoqinghao/DSEC/output/right/000167.png').convert('L')
disp_true = Image.open('/home/zhaoqinghao/DSEC/output/disp/000167.png').convert('L')
left_img = left_img.crop((0, 0, 640, 448))
right_img = right_img.crop((0, 0, 640, 448))
disp_true = disp_true.crop((0, 0, 640, 448))
# left_img = Image.open('/home/zhaoqinghao/dataset/KITTI_2015/training/image_2/000001_10.png').convert('L')
# right_img = Image.open('/home/zhaoqinghao/dataset/KITTI_2015/training/image_3/000001_10.png').convert('L')
# disp_true = Image.open('/home/zhaoqinghao/dataset/KITTI_2015/training/disp_occ_0/000001_10.png').convert('L')
# left_img = left_img.crop((0, 0, 1216, 320))
# right_img = right_img.crop((0, 0, 1216, 320))
# disp_true = disp_true.crop((0, 0, 1216, 320))

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

disp_true = torch.squeeze(disp_true, 1)
disp_pred = pred.data.cpu()
mask = disp_true > 0

def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)

# End Point Error (EPE)
print('EPE: ', EPE_metric(disp_pred, disp_true, mask).item())
# percentage of disparity outliers D1, errors greater than max(3px, 0.05d∗)
print('D1: ',D1_metric(disp_pred, disp_true, mask).item() * 100, '%')
# errors larger than 1 pixels (1 pixel error / bad 1.0)
print('1-px err: ', Thres_metric(disp_pred, disp_true, mask, 1).item() * 100, '%')
print('2-px err: ', Thres_metric(disp_pred, disp_true, mask, 2).item() * 100, '%')
print('3-px err: ', Thres_metric(disp_pred, disp_true, mask, 3).item() * 100, '%')
# save disp
pred = pred[0].data.cpu().numpy() * 256
skimage.io.imsave('sample_disp.png',pred.astype('uint16'))

