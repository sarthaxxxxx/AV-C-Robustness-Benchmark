import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.misc import imsave

# Our libs
from dataset import MUSICDataset
from nets import ModelBuilder

from utils import AverageMeter, makedirs

#new
import argparse
from glob import glob

def norm_tensor(x):
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    for i in range(x.size(0)):
        x[i, :, :, :] = norm(x[i])
    return x

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def inv_norm_tensor(x):
    inv_norm = UnNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    for i in range(x.size(0)):
        x[i, :, :, :] = inv_norm(x[i])

    return x

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_classifier = nets

    def forward(self, frame, audio):
        feat_frame = self.net_frame(frame)
        feat_sound = self.net_sound(audio)
        pred = self.net_classifier(feat_frame, feat_sound)
        return pred, feat_frame, feat_sound

def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    criterion = nn.CrossEntropyLoss()
    torch.set_grad_enabled(False)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    correct = 0

    total = 0
    for i, batch_data in enumerate(loader):
        audios = batch_data['audios']
        frames = batch_data['frames']
        gts = batch_data['labels']

        audio = audios[0].to(args.device).detach()
        frame = frames[0].to(args.device).squeeze(2).detach()
        gt = gts[0].to(args.device)
        # netWrapper.zero_grad()
        # forward pass
        preds, feat_v, feat_a = netWrapper(frame, audio)
        err = criterion(preds, gt) #+ F.cosine_similarity(feat_v, feat_a, 1).mean()

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        loss_meter.update(err.item())
        # print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))
    acc = 100 * correct / total
    print('[Eval Summary] Epoch: {}, Loss: {:.4f}'
          .format(epoch, loss_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['acc'].append(acc)
    print('Accuracy of the audio-visual event recognition network: %.2f %%' % (
            100 * correct / total))

# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    feats_a = []
    feats_v = []
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        audios = batch_data['audios']
        frames = batch_data['frames']
        gts = batch_data['labels']
        audio = audios[0].to(args.device)
        frame = frames[0].to(args.device).squeeze(2)
        gt = gts[0].to(args.device)


        # forward pass
        netWrapper.zero_grad()
        output, feat_v, feat_a = netWrapper.forward(frame, audio)
        feats_v.append(feat_v.detach())
        feats_a.append(feat_a.detach())
        err = criterion(output, gt)

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_classifier: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_classifier,
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())

def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame, net_classifier) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound.state_dict(),
               '{}/sound_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame.state_dict(),
               '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_classifier.state_dict(),
               '{}/classifier_{}'.format(args.ckpt, suffix_latest))

    cur_acc = history['val']['acc'][-1]
    if cur_acc > args.best_acc:
        args.best_acc = cur_acc
        torch.save(net_sound.state_dict(),
                   '{}/sound_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame.state_dict(),
                   '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_classifier.state_dict(),
                   '{}/classifier_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound, net_frame, net_classifier) = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound},
                    {'params': net_classifier.parameters(), 'lr': args.lr_classifier},
                    {'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_classifier *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound = builder.build_sound(
        weights=args.weights_sound)
    net_frame = builder.build_frame(
        pool_type=args.img_pool,
        weights=args.weights_frame)
    net_classifier = builder.build_classifier(
        cls_num=args.cls_num,
        weights=args.weights_classifier)
    nets = (net_sound, net_frame, net_classifier)

    # Dataset and Loader
    dataset_train = MUSICDataset(args.meta_path, args, split='train')
    dataset_val = MUSICDataset(args.meta_path, args, split='val')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    netWrapper = NetWrapper(nets)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    optimizer = create_optimizer(nets, args)

    # History of peroformance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'acc':[], 'cos':[]}}

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)

            checkpoint(nets, history, epoch, args)

        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-name", type=str, default="avsync15")
    parser.add_argument("--batch-size-per-gpu", type=int, default=4)
    parser.add_argument('--seed', default=1234, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='data/ckpt', help='folder to output checkpoints')
    parser.add_argument('--eval_epoch', type=int, default=1, help='frequency to evaluate')
    parser.add_argument('--cls_num', default=15, type=int, help='total class numbers')
    parser.add_argument('--audio_path', default='/mnt/data2/saksham/AV_robust/avsync15/audio')
    parser.add_argument('--frame_path', default='/mnt/data2/saksham/AV_robust/avsync15/frames')
    parser.add_argument('--meta_path', default='/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/train/meta/asva.csv')
    parser.add_argument("--categories", nargs="+", default=
        ['baby_babbling_crying', 'cap_gun_shooting', 'chicken_crowing', 'dog_barking', 'frog_croaking', 'hammering', 'lions_roaring',
         'machine_gun_shooting', 'playing_cello', 'playing_trombone', 'playing_trumpet', 'playing_violin__fiddle', 'sharpen_knife', 'striking_bowling', 'toilet_flushing'])

    # dataloader parameters
    parser.add_argument('--imgSize', default=224, type=int, help='size of input frame')
    parser.add_argument('--frameRate', default=8, type=float, help='video frame sampling rate')
    parser.add_argument('--audLen', default=16_000*6, type=int, help='sound length')
    parser.add_argument('--audRate', default=16_000, type=int, help='sound sampling rate')
    parser.add_argument('--stft_frame', default=1022, type=int, help="stft frame length")
    parser.add_argument('--stft_hop', default=256, type=int, help="stft hop length")
    
    parser.add_argument('--num_epoch', default=30, type=int, help='epochs to train for')
    parser.add_argument('--lr_frame', default=1e-4, type=float, help='LR')
    parser.add_argument('--lr_sound', default=1e-3, type=float, help='LR')
    parser.add_argument('--lr_classifier', default=1e-3, type=float, help='LR')
    parser.add_argument('--lr_steps', nargs='+', type=int, default=[10, 20], help='steps to drop LR in epochs')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')
    parser.add_argument('--disp_iter', type=int, default=20, help='frequency to display')
    parser.add_argument('--num_gpus', default=2, type=int, help='epochs to train for')

    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    
    args = parser.parse_args()
    args.device = torch.device("cuda")
    experiment_index = len(glob(f"{args.results_dir}/*"))
    args.ckpt = os.path.join(args.ckpt, f"experiment_{experiment_index:03d}")

    args.best_err = float("inf")
    args.best_acc = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)