import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

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

class NetWrapper(torch.nn.Module):
    def __init__(self, nets):
        super(NetWrapper, self).__init__()
        self.net_sound, self.net_frame, self.net_classifier = nets

    def forward(self, frame, audio):
        feat_frame = self.net_frame(frame)
        feat_sound = self.net_sound(audio)
        pred = self.net_classifier(feat_frame, feat_sound)
        return pred, feat_frame, feat_sound

def evaluate(netWrapper, loader, args):
    torch.set_grad_enabled(False)
    netWrapper.eval()

    correct = 0
    total = 0
    for i, batch_data in enumerate(loader):
        audios = batch_data['audios']
        frames = batch_data['frames']
        gts = batch_data['labels']

        audio = audios.to(args.device).detach()
        frame = frames.to(args.device).squeeze(2).detach()
        gt = gts.to(args.device)
        preds, feat_v, feat_a = netWrapper(frame, audio)

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

    print('Accuracy of the audio-visual event recognition network: %.2f %%' % (100 * correct / total))

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

    dataset_test = MUSICDataset(args.meta_path, args, split='test')
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)

    netWrapper = NetWrapper(nets)
    netWrapper.to(args.device)

    evaluate(netWrapper, loader_test, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-name", type=str, default="avsync15")
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
    parser.add_argument('--batch-size', type=int, default=20, help='frequency to display')

    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')

    parser.add_argument('--weights_sound', default='', help="weights to finetune net_sound")
    parser.add_argument('--weights_frame', default='', help="weights to finetune net_frame")
    parser.add_argument('--weights_classifier', default='', help="weights to finetune net_classifier")
    
    parser.add_argument('--img_pool', default='maxpool', help="avg or max pool image features")
    parser.add_argument('--log_freq', default=1, type=int, help="log frequency scale")

    args = parser.parse_args()
    args.device = torch.device("cuda")

    args.ckpt = os.path.join(args.ckpt, f"experiment_002_{args.data_name}")

    makedirs(args.ckpt)

    args.best_err = float("inf")
    args.best_acc = 0

    args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
    args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
    args.weights_classifier = os.path.join(args.ckpt, 'classifier_best.pth')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)