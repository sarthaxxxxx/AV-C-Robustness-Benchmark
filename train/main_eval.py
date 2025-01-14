import os
import random
# Numerical libs
import torch
from torchvision import transforms

# Our libs
from dataset import MUSICDataset
from nets import ModelBuilder

#new
import argparse

import warnings
warnings.filterwarnings("ignore")

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
        num_workers=0,
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
    
    parser.add_argument('--batch-size', type=int, default=20, help='frequency to display')

    parser.add_argument('--weights_sound', default='', help="weights to finetune net_sound")
    parser.add_argument('--weights_frame', default='', help="weights to finetune net_frame")
    parser.add_argument('--weights_classifier', default='', help="weights to finetune net_classifier")
    
    parser.add_argument('--img_pool', default='maxpool', help="avg or max pool image features")
    parser.add_argument('--log_freq', default=1, type=int, help="log frequency scale")
    
    parser.add_argument("--add_audio_noise", action="store_true")
    parser.add_argument("--audio_noise_type", type=str, default="gaussian")
    parser.add_argument('--audio_noise_intensity', default=5, type=int)
    
    parser.add_argument("--add_frame_noise", action="store_true")
    parser.add_argument("--frame_noise_type", type=str, default="gaussian")
    parser.add_argument('--frame_noise_intensity', default=5, type=int)

    args = parser.parse_args()
    args.device = torch.device("cuda")

    args.ckpt = os.path.join(args.ckpt, f"experiment_002_{args.data_name}")

    args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')
    args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
    args.weights_classifier = os.path.join(args.ckpt, 'classifier_best.pth')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)