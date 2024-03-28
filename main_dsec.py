from __future__ import print_function, division
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from torch.utils.data import DataLoader
import torch.utils.data
import time
from datasets import __datasets__
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
from models.bgnet_plus_png_batch import BGNet_Plus_Batch
from utils import *
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import gc
from tensorboardX import SummaryWriter
from pathlib import Path
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from dataset.provider import DatasetProvider
import hdf5plugin
import random

parser = argparse.ArgumentParser(description='BGNet')
parser.add_argument('--model', default='bgnet_plus_batch', help='select a model structure')

parser.add_argument('--train_dir', help='Path to DSEC dataset directory',default='/disk2/users/M22_zhaoqinghao/Dataset/h5')
parser.add_argument('--test_dir', help='Path to DSEC dataset directory',default='/disk2/users/M22_zhaoqinghao/Dataset/h5')

parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers for dataloader')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--lrepochs',default="50,100,150,200,250:10", type=str,  help='the epochs to decay lr: the downscale rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=10, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--logdir',default='./logs_test/', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
parser.add_argument('--resume', default=False, action='store_true', help='continue training the model')
parser.add_argument('--patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped.')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

train_dir = Path(args.train_dir)
train_dataset_provider = DatasetProvider(train_dir)
train_dataset = train_dataset_provider.get_train_dataset()
batch_size = args.batch_size
num_workers = args.num_workers
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False)

test_dir = Path(args.test_dir)
test_dataset_provider = DatasetProvider(test_dir)
test_dataset = test_dataset_provider.get_train_dataset()
batch_size = args.batch_size
num_workers = args.num_workers
test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False)

if args.model == 'bgnet':
    model = BGNet().cuda()
elif args.model == 'bgnet_plus_batch':
    model = BGNet_Plus_Batch().cuda()

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch']
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))

def train_sample(imgL, imgR, disp_L, compute_metrics=False):
    model.train()
    if args.cuda:
        imgL, imgR, disp_gt = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    # imgL = torch.squeeze(imgL, 0)
    # imgR = torch.squeeze(imgL, 0)
    optimizer.zero_grad()
    disp_ests, _ = model(imgL, imgR)
    mask = disp_gt > 0
    loss = F.smooth_l1_loss(disp_ests[mask], disp_gt[mask], size_average=True)
    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = disp_error_image_func.apply(disp_ests, disp_gt)
            scalar_outputs["EPE"] = EPE_metric(disp_ests, disp_gt, mask)
            scalar_outputs["D1"] = D1_metric(disp_ests, disp_gt, mask)
            scalar_outputs["Thres1"] = Thres_metric(disp_ests, disp_gt, mask, 1.0) 
            scalar_outputs["Thres2"] = Thres_metric(disp_ests, disp_gt, mask, 2.0) 
            scalar_outputs["Thres3"] = Thres_metric(disp_ests, disp_gt, mask, 3.0)
    loss.backward()
    optimizer.step()
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

def test_sample(imgL,imgR,disp_gt):
    model.eval()
    # imgL = torch.squeeze(imgL, 0)
    # imgR = torch.squeeze(imgL, 0)
    with torch.no_grad():
        disp_ests,_ = model(imgL.cuda(), imgR.cuda())
    mask = disp_gt > 0

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

    start_full_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print('This is %d-th epoch' %(epoch))
        adjust_learning_rate(optimizer, epoch, args.lr, args.lrepochs)
        
        # TRAIN
        total_train_loss = 0
        start_time = 0
        for i, data in enumerate(train_loader, start=1):
            global_step = len(train_loader) * epoch + i
            
            
            disp = data['disparity_gt'].squeeze().float() / 256.0
            disp = disp.cuda()
            left_voxel_grid = data['representation']['left']  # [B, C, H, W]
            right_voxel_grid = data['representation']['right']
            left_voxel_grid = left_voxel_grid.cuda()
            right_voxel_grid = right_voxel_grid.cuda()
            _, num_channels, _, _ = left_voxel_grid.shape
            left_channels = [left_voxel_grid[:, i] for i in range(num_channels)]
            right_channels = [right_voxel_grid[:, i] for i in range(num_channels)]
            for j, channel in enumerate(left_channels):
                channel = channel.mul(256 / channel.max()).to(torch.uint8)
                left_voxel_grid[:, j] = channel
            for j, channel in enumerate(right_channels):
                channel = channel.mul(256 / channel.max()).to(torch.uint8)
                right_voxel_grid[:, j] = channel
            _, _ , h, w = left_voxel_grid.shape
            crop_h, crop_w = 256, 320
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)
            left_voxel_grid = left_voxel_grid[:, :, y1:y1 + crop_h, x1:x1 + crop_w]
            right_voxel_grid = right_voxel_grid[:, :, y1:y1 + crop_h, x1:x1 + crop_w]
            disp = disp[:, y1:y1 + crop_h, x1:x1 + crop_w]
            
            loss, scalar_outputs, image_outputs = train_sample(left_voxel_grid, right_voxel_grid, disp, False)
            do_summary = global_step % args.summary_freq == 0
            if do_summary:
                logger.add_scalar('train_loss', loss, global_step)
                # save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch, args.epochs,
                                                                                       i,
                                                                                       len(train_loader) - 1, loss,
                                                                                       time.time() - start_time))
            start_time = time.time()
            total_train_loss += loss
        
        print('Epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(train_loader)))
        logger.add_scalar('total_train_loss', total_train_loss/len(train_loader), epoch)
        # TEST
        avg_test_scalars = AverageMeterDict()
        for i, data in enumerate(test_loader):
            start_time = time.time()
            
            disp = data['disparity_gt'].numpy().squeeze().astype('uint16')/256.0
            disp = torch.from_numpy(disp)
            
            left_voxel_grid = data['representation']['left']  # [B, C, H, W]
            right_voxel_grid = data['representation']['right']
            _, num_channels, _, _ = left_voxel_grid.shape
            left_channels = [left_voxel_grid[:, i] for i in range(num_channels)]
            right_channels = [right_voxel_grid[:, i] for i in range(num_channels)]
            for j, channel in enumerate(left_channels):
                array = channel.numpy()
                img = (array / array.max() * 256).astype('uint8')
                left_voxel_grid[:, j] = torch.from_numpy(img)
            for j, channel in enumerate(right_channels):
                array = channel.numpy()
                img = (array / array.max() * 256).astype('uint8')
                right_voxel_grid[:, j] = torch.from_numpy(img)
            
            left_voxel_grid = left_voxel_grid[:, :, :448, :640]
            right_voxel_grid = right_voxel_grid[:, :, :448, :640]
            disp = disp[:, 0:448, 0:640]
            
            loss, scalar_outputs, image_outputs = test_sample(left_voxel_grid, right_voxel_grid, disp)
            # save_scalars(logger, 'test', scalar_outputs, global_step)
            # save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch, args.epochs,
                                                                                        i,
                                                                                        len(test_loader), loss,
                                                                                        time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(train_loader) * (epoch + 1))
        print("avg_test_scalars", avg_test_scalars)
        
        
        
        # # early stopping
        # if avg_test_loss < best_test_loss:
        #     best_test_loss = avg_test_loss
        #     no_improve_epoch = 0
        # else:
        #     no_improve_epoch += 1

        # if no_improve_epoch > args.patience:
        #     print("Early stopping")
        #     break

        if epoch % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch))
        gc.collect()


    # print('MAX epoch %d total test err = %.3f' %(epoch, best_test_loss))
    print('full train time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()
