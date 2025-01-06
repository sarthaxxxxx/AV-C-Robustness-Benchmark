#####################################
### DAP-CL ENVIRONMENT ####
#####################################

#######################################
# Check script again. Extraction of noise patterns.
#######################################

import numpy as np
from pydub import AudioSegment
import soundfile as sf
import os
import torch.utils.data as data
import torch
import argparse
import collections
import skimage as sk



def gaussian_noise(audio_file, output_path, intensity):
    # load audio
    audio, sr = sf.read(audio_file)
    # calculate std
    noise_std = [.08, .12, 0.18, 0.26, 0.38][intensity - 1]
    # generate white noise
    noise = np.random.normal(0, noise_std, len(audio))
    # add
    audio_with_noise = audio + noise
    sf.write(output_path, audio_with_noise, sr)


def speckle_noise(audio_file, output_path, intensity):
    audio, sr = sf.read(audio_file)
    noise_std = [.15, .2, 0.35, 0.45, 0.6][intensity - 1]
    noise = np.random.normal(0, noise_std, len(audio))
    audio_with_noise = audio + audio * noise
    sf.write(output_path, audio_with_noise, sr)


def shot_noise(audio_file, output_path, intensity):
    audio, sr = sf.read(audio_file)
    c = [60, 25, 12, 5, 3][intensity - 1]
    audio_with_noise = np.random.poisson(lam=c, size=len(audio)) / c
    sf.write(output_path, audio_with_noise, sr)



def impulse_noise(audio_file, output_path, intensity):
    audio, sr = sf.read(audio_file)
    c = [.03, .06, .09, 0.17, 0.27][intensity - 1]
    audio_with_noise = sk.util.random_noise(audio, mode='s&p', amount=c)
    sf.write(output_path, audio_with_noise, sr)


def add_external_noise(audio_path, weather_path, output_path, intensity):
    audio = AudioSegment.from_file(audio_path)
    noise_sound = AudioSegment.from_file(weather_path)

    # adjust the length
    if len(audio) <= len(noise_sound):
        noise_sound = noise_sound[:len(audio)]
    else:
        print(len(audio), len(noise_sound))
        num_repeats = len(audio) // len(noise_sound) + 1
        noise_sound = noise_sound * num_repeats
        noise_sound = noise_sound[:len(audio)]
        print(len(audio), len(noise_sound))

    scale = [1, 2, 4, 6, 8]
    noise_sound = noise_sound.apply_gain(scale[intensity-1])

    output = audio.overlay(noise_sound)
    output.export(output_path, format="wav")


def make_dataset(dir, candi_audios):
    audios = []
    dir = os.path.expanduser(dir)
    # for name in sorted(os.listdir(dir)):
    for name in sorted(candi_audios):
        path = os.path.join(dir, name)
        # item = (path, name)
        audios.append(path)

    return audios

class DistortAudioFolder(data.Dataset):
    def __init__(self, root, candi_audio_names, corruption, noise_path, severity, save_path):
        audios = make_dataset(root, candi_audio_names)
        if len(audios) == 0:
            raise (RuntimeError("Found 0 audios in subfolders of: "))

        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.audio_paths = audios
        self.candi_audio_names = sorted(candi_audio_names)
        self.noise_path = noise_path
        self.weather_path = noise_path
        self.save_path = save_path

    def __getitem__(self, index):
        save_path = os.path.join(self.save_path, self.corruption, 'severity_{}'.format(self.severity))
        
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except:
                print(save_path)
        print(self.candi_audio_names[index])
        
        if self.corruption == 'gaussian_noise':
            gaussian_noise(self.audio_paths[index], os.path.join(save_path, self.candi_audio_names[index]), self.severity)
        elif self.corruption == 'impulse_noise':
            impulse_noise(self.audio_paths[index], os.path.join(save_path, self.candi_audio_names[index]), self.severity)
        elif self.corruption == 'shot_noise':
            shot_noise(self.audio_paths[index], os.path.join(save_path, self.candi_audio_names[index]), self.severity)
        elif self.corruption == 'speckle_noise':
            speckle_noise(self.audio_paths[index], os.path.join(save_path, self.candi_audio_names[index]), self.severity)
        else:
            add_external_noise(self.audio_paths[index], os.path.join(self.weather_path, self.corruption + '.wav'), os.path.join(save_path, self.candi_audio_names[index]), self.severity)

        return 0 

    def __len__(self):
        return len(self.audio_paths)




ROOTDIR = '/people/cs/s/skm200005/UTD/AV-Robustness/'


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--corruption', type=str, default='wind', choices=['all'], help='Type of corruption to apply')
parser.add_argument('--severity', type=int, default=5, choices=[1, 2, 3, 4, 5], help='Severity of corruption to apply')
parser.add_argument('--data_path', type=str, help='Path to test data')
parser.add_argument('--save_path', type=str, help='Path to store corruption data')
parser.add_argument('--noise_path', type=str, default=f'{ROOTDIR}/data/VGGSound/NoisyAudios', help='Path to store corruption data')
args = parser.parse_args()


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

dir = args.data_path
candi_audio_names = os.listdir(dir)
tmp = sorted(candi_audio_names)

noise_path = args.noise_path

d = collections.OrderedDict()
if args.corruption == 'all':
    corruption_list = ['gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise', 'snow', 'frost' , 'spatter', 'wind']
else:
    corruption_list = [args.corruption]

for corruption in corruption_list:
    # for severity in range(1, 6):
    print('Adding the {} corruption (severity={}) to audios'.format(corruption, args.severity))
    distorted_dataset = DistortAudioFolder(
        root=args.data_path,
        candi_audio_names=candi_audio_names,
        corruption=corruption,
        noise_path=args.noise_path,
        severity=args.severity,
        save_path=args.save_path)
    distorted_dataset_loader = torch.utils.data.DataLoader(
        distorted_dataset, batch_size=12, shuffle=False, num_workers=0)
    for _ in distorted_dataset_loader:
        continue