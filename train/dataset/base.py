import random
import os
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import librosa
from PIL import Image
import pandas as pd
import soundfile as sf

from . import video_transforms as vtransforms
from .distortions import noise_function_map_v, noise_function_map_a

TEMP_BASE_DIR = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/train/temp_samples'
SAVE_PROB = 0.05

class BaseDataset(torchdata.Dataset):
    def __init__(self, meta_path, opt, split='train'):
        # params
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.add_audio_noise = opt.add_audio_noise
        self.audio_noise_type = opt.audio_noise_type
        self.audio_noise_intensity = opt.audio_noise_intensity
        
        self.add_frame_noise = opt.add_frame_noise
        self.frame_noise_type = opt.frame_noise_type
        self.frame_noise_intensity = opt.frame_noise_intensity

        if self.add_frame_noise:
            print(f'Adding frame noise of type {self.frame_noise_type} with intensity {self.frame_noise_intensity}')
        
        if self.add_audio_noise:
            print(f'Adding audio noise of type {self.audio_noise_type} with intensity {self.audio_noise_intensity}')

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        df = pd.read_csv(meta_path)
        df = df[df['new_split'] == split].reset_index(drop=True)
         
        self.list_sample = []
        for i, row in df.iterrows():
            label = row['label']
            vid = row['vid']
            frame_count = row['frames_count']
            self.list_sample.append([os.path.join(label,vid), label, frame_count])

        if self.split == 'train':
            random.shuffle(self.list_sample)

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list_1 = []
        transform_list_2 = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list_1.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list_1.append(vtransforms.RandomCrop(self.imgSize))
            transform_list_1.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list_1.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list_1.append(vtransforms.CenterCrop(self.imgSize))

        transform_list_2.append(vtransforms.ToTensor())
        transform_list_2.append(vtransforms.Normalize(mean, std))
        transform_list_2.append(vtransforms.Stack())
        self.vid_transform_1 = transforms.Compose(transform_list_1)
        self.vid_transform_2 = transforms.Compose(transform_list_2)

    def _load_frames(self, path):
        frame = self._load_frame(path)
        frame = self.vid_transform_1([frame])
        if self.add_frame_noise:
            method = noise_function_map_v[self.frame_noise_type]
            frame = [method(frame[0], self.frame_noise_intensity)]
            if random.random() < SAVE_PROB:
                # save the noisy frame
                file_name = path.split('/')[-1].split('.')[0]
                save_path = os.path.join(TEMP_BASE_DIR, f'{file_name}_{self.frame_noise_type}_{self.frame_noise_intensity}.jpg')
                Image.fromarray(frame[0]).save(save_path)

        frame = self.vid_transform_2(frame)
        return frame

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):
        # audio_raw, rate = librosa.load(path, sr=None, mono=True)
        audio_raw, rate = librosa.load(path, sr=self.audRate, mono=True)
        return audio_raw, rate

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # add noise
        if self.add_audio_noise:
            method = noise_function_map_a.get(self.audio_noise_type, noise_function_map_a['weather_noise_a'])
            # method = noise_function_map_a[self.audio_noise_type]
            audio_raw = method(audio_raw, self.audio_noise_intensity)
            if random.random() < SAVE_PROB:
                # save the noisy audio
                file_name = path.split('/')[-1].split('.')[0]
                save_path = os.path.join(TEMP_BASE_DIR, f'{file_name}_{self.audio_noise_type}_{self.audio_noise_intensity}.wav')
                sf.write(save_path, audio_raw, rate)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        audio = torch.from_numpy(audio).unsqueeze(0)
        return audio