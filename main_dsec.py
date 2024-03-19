from __future__ import print_function, division
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from torch.utils.data import DataLoader
import torch.utils.data
import time
from datasets import __datasets__
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
from models.bgnet_plus_attn import BGNet_Plus_Attn
from utils import *
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import gc
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='BGNet')
parser.add_argument('--model', default='BGNet_Plus_Attn', help='select a model structure')
parser.add_argument('--dataset', default='dsec_png', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/zhaoqinghao/dataset/DSEC/output',
                    help='datapath')
parser.add_argument('--trainlist', default='/home/zhaoqinghao/DSEC/train.txt', 
                    help='training list')
parser.add_argument('--testlist', default='/home/zhaoqinghao/DSEC/test.txt', 
                    help='testing list')
# parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--batch_size', type=int, default=28, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--lrepochs',default="200,300,400:10", type=str,  help='the epochs to decay lr: the downscale rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=100, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=2, help='the frequency of saving checkpoint')
parser.add_argument('--logdir',default='./logs/', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
parser.add_argument('--resume', default=True, action='store_true', help='continue training the model')
parser.add_argument('--patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

datapath = args.datapath
StereoDataset = __datasets__[args.dataset]

dsec_train = args.trainlist
dsec_train_dataset = StereoDataset(datapath, dsec_train, True)
TrainImgLoader = DataLoader(dsec_train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=32, drop_last=False)

dsec_test = args.testlist
dsec_test_dataset = StereoDataset(datapath, dsec_test, False)
TestImgLoader = DataLoader(dsec_test_dataset, batch_size= args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

if args.model == 'bgnet':
    model = BGNet().cuda()
elif args.model == 'bgnet_plus':
    model = BGNet_Plus().cuda()

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
    optimizer.zero_grad()
    disp_ests, _ = model(imgL, imgR)
    mask = (disp_gt > 0)
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

def train(imgL, imgR, disp_L):
    model.train()
    if args.cuda:
        imgL, imgR, disp_gt = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    optimizer.zero_grad()
    disp_ests, _ = model(imgL, imgR)
    mask = (disp_gt > 0)
    loss = F.smooth_l1_loss(disp_ests[mask], disp_gt[mask], reduction='mean')
    loss.backward()
    optimizer.step()
    return loss.data

def test(imgL,imgR,disp_gt):
    model.eval()
    with torch.no_grad():
        disp_ests,_ = model(imgL.cuda(), imgR.cuda())
    mask = (disp_gt > 0)

    disp_gt = disp_gt.cuda()
    err_3px = Thres_metric(disp_ests, disp_gt, mask, 3.0)
    return err_3px.data



def main():
    best_test_loss = float("inf") # set to large number
    no_improve_epoch = 0
    start_full_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print('This is %d-th epoch' %(epoch))
        adjust_learning_rate(optimizer, epoch, args.lr, args.lrepochs)
        
        # TRAIN
        total_train_loss = 0
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch + batch_idx
            start_time = time.time()
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            
            # loss = train(imgL.cuda(), imgR.cuda(), disp_L.cuda())
            
            loss, scalar_outputs, image_outputs = train_sample(imgL.cuda(), imgR.cuda(), disp_L.cuda(), False)
            do_summary = global_step % args.summary_freq == 0
            if do_summary:
                logger.add_scalar('train_loss', loss, global_step)
                # save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader) - 1, loss,
                                                                                       time.time() - start_time))
            total_train_loss += loss
        
        print('Epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        logger.add_scalar('total_train_loss', total_train_loss/len(TrainImgLoader), epoch)
        # TEST
        avg_test_scalars = AverageMeterDict()
        total_test_loss = 0
        for batch_idx, sample in enumerate(TestImgLoader):
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            start_time = time.time()
            
            loss, scalar_outputs, image_outputs = test_sample(imgL,imgR,disp_L)
            # save_scalars(logger, 'test', scalar_outputs, global_step)
            # save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            
            test_loss = test(imgL.cuda(), imgR.cuda(), disp_L.cuda())
            total_test_loss += test_loss
            
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch, args.epochs,
                                                                                        batch_idx,
                                                                                        len(TestImgLoader), loss,
                                                                                        time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch + 1))
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
