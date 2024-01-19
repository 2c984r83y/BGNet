from __future__ import print_function, division
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from torch.utils.data import DataLoader
import torch.utils.data
import time
from datasets import __datasets__
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
from utils import *
import torch
import torch.utils.data
import torch.nn.functional as F
import gc

parser = argparse.ArgumentParser(description='BGNet')
parser.add_argument('--model', default='bgnet_plus', help='select a model structure')
parser.add_argument('--dataset', default='dsec_png', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/zhaoqinghao/dataset/DSEC/output',help='datapath')
parser.add_argument('--testlist', default='/home/zhaoqinghao/DSEC/test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./pretrained_models/checkpoint_000300.ckpt', 
                    help='load the weights from a specific checkpoint')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

datapath = args.datapath
StereoDataset = __datasets__[args.dataset]

dsec_test = args.testlist
dsec_test_dataset = StereoDataset(datapath, dsec_test, False)
TestImgLoader = DataLoader(dsec_test_dataset, batch_size= 32, shuffle=False, num_workers=16, drop_last=False)

if args.model == 'bgnet':
    model = BGNet().cuda()
elif args.model == 'bgnet_plus':
    model = BGNet_Plus().cuda()


# load the checkpoint file specified by args.loadckpt
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

def test_sample(imgL,imgR,disp_gt):
    model.eval()
    with torch.no_grad():
        disp_ests,_ = model(imgL.cuda(), imgR.cuda())
    mask = (disp_gt > 0)

    disp_gt = disp_gt.cuda()
    loss = F.smooth_l1_loss(disp_ests[mask], disp_gt[mask], reduction='mean')
    
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    with torch.no_grad():
        image_outputs["errormap"] = disp_error_image_func.apply(disp_ests, disp_gt)
        scalar_outputs["EPE"] = EPE_metric(disp_ests, disp_gt, mask)
        scalar_outputs["D1"] = D1_metric(disp_ests, disp_gt, mask)
        scalar_outputs["Thres1"] = Thres_metric(disp_ests, disp_gt, mask, 1.0)
        scalar_outputs["Thres2"] = Thres_metric(disp_ests, disp_gt, mask, 2.0)
        scalar_outputs["Thres3"] = Thres_metric(disp_ests, disp_gt, mask, 3.0)
    # return loss.data
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def main():
        # TEST
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            start_time = time.time()
            loss, scalar_outputs, image_outputs = test_sample(imgL,imgR,disp_L)
            # save_scalars(logger, 'test', scalar_outputs, batch_idx)
            # save_images(logger, 'test', image_outputs, batch_idx)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx + 1,
                                                                        len(TestImgLoader), loss,
                                                                        time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        
        print("EPE", avg_test_scalars["EPE"])
        print("D1", avg_test_scalars["D1"] * 100, "%")
        print("Thres1", avg_test_scalars["Thres1"] * 100, "%")
        print("Thres2", avg_test_scalars["Thres2"] * 100, "%")
        print("Thres3", avg_test_scalars["Thres3"] * 100, "%")
        
if __name__ == '__main__':
   main()
