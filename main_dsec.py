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
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import gc
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='BGNet')
parser.add_argument('--model', default='bgnet_plus', help='select a model structure')
parser.add_argument('--dataset', default='dsec_png', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='/home/zhaoqinghao/dataset/DSEC/output',
                    help='datapath')
parser.add_argument('--savepath', default='/disk2/users/M22_zhaoqinghao/BGNet/output/', 
                    help='save path')
parser.add_argument('--trainlist', default='/home/zhaoqinghao/DSEC/train.txt', 
                    help='training list')
parser.add_argument('--testlist', default='/home/zhaoqinghao/DSEC/test.txt', 
                    help='testing list')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=10, help='the frequency of saving checkpoint')
parser.add_argument('--logdir',default='./log/', help='the directory to save logs and checkpoints')
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
TrainImgLoader = DataLoader(dsec_train_dataset, batch_size= 32, shuffle=True, num_workers=32, drop_last=False)

dsec_test = args.testlist
dsec_test_dataset = StereoDataset(datapath, dsec_test, False)
TestImgLoader = DataLoader(dsec_test_dataset, batch_size= 16, shuffle=False, num_workers=4, drop_last=False)

if args.model == 'bgnet':
    model = BGNet().cuda()
elif args.model == 'bgnet_plus':
    model = BGNet_Plus().cuda()

if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)

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
    disp_pred = pred.data.cpu()
    disp_true = torch.squeeze(disp_true, 1)
    disp_true = torch.squeeze(disp_true, 1)
    mask = disp_true > 0
    disp_pred, disp_true = disp_pred[mask], disp_true[mask]
    E = torch.abs(disp_true - disp_pred)
    err_mask = E > 3
    return torch.mean(err_mask.float()).item()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
       lr = 0.001
    else:
       lr = 0.0001
    print('learning rate = %.6f' %(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    best_test_loss = float("inf") # set to large number
    no_improve_epoch = 0
    start_full_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        print('This is %d-th epoch' %(epoch))
        adjust_learning_rate(optimizer, epoch)
        # TRAIN
        total_train_loss = 0
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch + batch_idx
            start_time = time.time()
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            loss = train(imgL.cuda(), imgR.cuda(), disp_L.cuda())
            do_summary = global_step % args.summary_freq == 0
            if do_summary:
                logger.add_scalar('train_loss', loss, global_step)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        logger.add_scalar('epoch_train_loss', loss, epoch)
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))
        # TEST
        total_test_loss = 0
        for batch_idx, sample in enumerate(TestImgLoader):
            imgL, imgR, disp_L = sample['left'], sample['right'], sample['disparity']
            test_loss = test(imgL, imgR, disp_L)
            print('Iter %d 3-px err in val = %.3f' %(batch_idx, test_loss*100))
            total_test_loss += test_loss
        avg_test_loss = total_test_loss / len(TestImgLoader)
        logger.add_scalar('test loss', test_loss, epoch)
        print('epoch %d avg 3-px err in val = %.3f' %(epoch, avg_test_loss*100))

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


    print('MAX epoch %d total test err = %.3f' %(epoch, best_test_loss))
    print('full train time = %.2f HR' %((time.time() - start_full_time)/3600))

if __name__ == '__main__':
   main()
