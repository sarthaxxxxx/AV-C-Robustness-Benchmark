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
import torchaudio

from scipy.signal import butter, filtfilt
import torchaudio.functional as F

# def gaussian_noise(audio_file, output_path, intensity):
#     # load audio
#     audio, sr = sf.read(audio_file)
#     # calculate std
#     noise_std = [.08, .12, 0.18, 0.26, 0.38][intensity - 1]
#     # generate white noise
#     noise = np.random.normal(0, noise_std, len(audio))
#     audio_with_noise = audio + noise
#     max_amplitude = max(np.max(np.abs(audio_with_noise)), 1.0) 
#     audio_with_noise = audio_with_noise / max_amplitude
#     sf.write(output_path, audio_with_noise, sr)

# def shot_noise(audio_file, output_path, intensity):
#     audio, sr = sf.read(audio_file)
#     c = [60, 25, 12, 5, 3][intensity - 1]
#     audio_with_noise = np.random.poisson(lam=c, size=len(audio)) / c
#     audio_with_noise = audio_with_noise * audio
#     max_amplitude = max(np.max(np.abs(audio_with_noise)), 1.0)
#     audio_with_noise = audio_with_noise / max_amplitude
#     sf.write(output_path, audio_with_noise, sr)

# def speckle_noise(audio_file, output_path, intensity):
#     audio, sr = sf.read(audio_file)
#     noise_std = [.15, .2, 0.35, 0.45, 0.6][intensity - 1]
#     noise = np.random.normal(0, noise_std, len(audio))
#     audio_with_noise = audio + (audio * noise)
#     sf.write(output_path, audio_with_noise, sr)

# def impulse_noise(audio_file, output_path, intensity):
#     audio, sr = sf.read(audio_file)
#     c = [.03, .06, .09, 0.17, 0.27][intensity - 1]
#     audio_with_noise = sk.util.random_noise(audio, mode='s&p', amount=c)
#     max_amplitude = max(np.max(np.abs(audio_with_noise)), 1.0)
#     audio_with_noise = audio_with_noise / max_amplitude
#     sf.write(output_path, audio_with_noise, sr)

####################################### DIGITAL NOISE #######################################
NOISE_SNRS = [40, 30, 20, 10, 0]
def gaussian_noise(audio_file, output_path, intensity):
    def add_gaussian_noise(waveform, snr_db):
        noise = torch.randn_like(waveform)  # Generate Gaussian noise
        snr_tensor = torch.tensor([snr_db], dtype=waveform.dtype, device=waveform.device).expand(waveform.shape[:-1])  
        return F.add_noise(waveform, noise, snr_tensor)  # Corrected SNR shape
    waveform, sr = torchaudio.load(audio_file)
    waveform_noise = add_gaussian_noise(waveform, NOISE_SNRS[intensity-1])
    torchaudio.save(output_path, waveform_noise, sr)

def speckle_noise(audio_file, output_path, intensity):
    snr_level = NOISE_SNRS[intensity - 1]
    waveform, sr = torchaudio.load(audio_file)  # waveform shape: [channels, samples]
    signal_power = torch.mean(waveform ** 2)
    noise = torch.randn_like(waveform)  # Zero-mean Gaussian noise
    noise = noise * waveform  # Multiply by signal (speckle noise)
    noise_power = torch.mean(noise ** 2) + 1e-8  # Prevent division by zero
    snr_linear = 10 ** (snr_level / 10)  # Standard dB to linear conversion
    noise_scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    noise = noise_scaling_factor * noise
    y_noisy = torch.clamp(waveform + noise, -1.0, 1.0)
    torchaudio.save(output_path, y_noisy, sr)

def shot_noise(audio_file, output_path, intensity):
    waveform, sr = torchaudio.load(audio_file)
    snr_linear = NOISE_SNRS[intensity - 1]
    signal_power = torch.mean(waveform ** 2)
    y_min, y_max = waveform.min(), waveform.max()
    y_norm = (waveform - y_min) / (y_max - y_min)  # Normalize to [0, 1]
    poisson_noise = torch.poisson(y_norm * 50) / 50  # Base Poisson noise
    noise = poisson_noise - y_norm  # Convert to zero-mean noise
    noise_power = torch.mean(noise ** 2)
    snr_linear = 10 ** (snr_linear / 10)  # Convert SNR from dB to linear scale
    noise_scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    noise = noise_scaling_factor * noise
    y_noisy = torch.clamp(waveform + noise, -1.0, 1.0)
    torchaudio.save(output_path, waveform_noise, sr)

def impulse_noise(audio_file, output_path, intensity, impulse_prob=0.05):
    waveform, sr = torchaudio.load(audio_file)  # waveform shape: [channels, samples]
    snr_level = NOISE_SNRS[intensity - 1]
    signal_power = torch.mean(waveform ** 2)
    random_mask = torch.rand_like(waveform)  # Random values in [0,1]
    salt_pepper = torch.where(
        random_mask < (impulse_prob / 2), -1.0,  # Set to -1 (salt)
        torch.where(random_mask > (1 - impulse_prob / 2), 1.0, 0.0)  # Set to 1 (pepper), else 0
    )
    noise_power = torch.mean(salt_pepper ** 2) + 1e-8  # Prevent division by zero
    snr_linear = 10 ** (snr_level / 10)  # Standard dB to linear conversion
    noise_scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    salt_pepper = noise_scaling_factor * salt_pepper
    y_noisy = torch.clamp(waveform + salt_pepper, -1.0, 1.0)
    torchaudio.save(output_path, y_noisy, sr)
###########################################################################################

##################################### Environmental ######################################################
def add_env_noise(audio_file, output_path, intensity, noise_dir=''):
    waveform, sr = torchaudio.load(audio_file)
    assert sr == 16000, "Noise file must have a sample rate of 16 kHz."
    xlen = waveform.shape[-1]
    noise_files = os.listdir(noise_dir)
    if not noise_files:
        raise FileNotFoundError("No noise files found in the directory.")
    noise_file = np.random.choice(noise_files)
    noise_file = os.path.join(noise_dir, noise_file)
    noise_raw, sr = torchaudio.load(noise_file)
    if sr != 16000:
        noise_raw = torchaudio.transforms.Resample(sr, 16000)(noise_raw)
    noise = noise_raw[..., :xlen]
    while noise.shape[-1] < xlen:
        noise = torch.cat([noise, noise], -1)
    noise = noise[..., :xlen] 
    snr_tensor = torch.full_like(waveform[:, 0], NOISE_SNRS[intensity-1])
    waveform_noise = F.add_noise(waveform, noise, snr_tensor)
    torchaudio.save(output_path, waveform_noise, sr)

def snow_noise(audio_file, output_path, intensity):    
    snow_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/data_recipe/src/noise_files/snow'
    add_env_noise(audio_file, output_path, intensity, noise_dir=snow_dir)

def wind_noise(audio_file, output_path, intensity):    
    wind_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/data_recipe/src/noise_files/wind'
    add_env_noise(audio_file, output_path, intensity, noise_dir=wind_dir)

def rain_noise(audio_file, output_path, intensity):    
    rain_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/data_recipe/src/noise_files/rain'
    add_env_noise(audio_file, output_path, intensity, noise_dir=rain_dir)

def frost_noise(audio_file, output_path, intensity):    
    frost_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/data_recipe/src/noise_files/frost'
    add_env_noise(audio_file, output_path, intensity, noise_dir=frost_dir)

def spatter_noise(audio_file, output_path, intensity):    
    spatter_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/data_recipe/src/noise_files/water_drops'
    add_env_noise(audio_file, output_path, intensity, noise_dir=spatter_dir)

###########################################################################################

##################################### Human ######################################################
def underwater_noise(audio_file, output_path, intensity):    
    spatter_dir = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/data_recipe/src/noise_files/underwater'
    add_env_noise(audio_file, output_path, intensity, noise_dir=spatter_dir)

###########################################################################################

# def underwater(audio_file, output_path, intensity):
#     audio, sr = sf.read(audio_file)

#     # Define severity levels for low-pass filtering and muffling
#     cutoff_values = {1: 3000, 2: 2000, 3: 1500, 4: 1000, 5: 500}
#     muffle_factors = {1: 0.85, 2: 0.75, 3: 0.65, 4: 0.55, 5: 0.45}

#     nyquist_rate = sr / 2

#     cutoff = cutoff_values[intensity]
#     muffle = muffle_factors[intensity]

#     # Apply low-pass filter
#     b, a = butter(5, cutoff / nyquist_rate, btype='low')
#     audio_low_pass = filtfilt(b, a, audio) * muffle

#     sf.write(output_path, audio_low_pass, sr)

def add_external_noise(audio_path, weather_path, output_path, intensity):
    
    audio = AudioSegment.from_file(audio_path)
    _, sr = sf.read(audio_path)
    # noise_sound, sr_n = sf.read(weather_path)
    noise_sound = AudioSegment.from_file(weather_path)

    # if noise_sound.frame_rate != sr:
    #     noise_sound = noise_sound.set_frame_rate(sr)

    # adjust the length
    if len(audio) <= len(noise_sound):
        noise_sound = noise_sound[:len(audio)]
    else:
        print(len(audio), len(noise_sound))
        num_repeats = len(audio) // len(noise_sound) + 1
        noise_sound = noise_sound * num_repeats
        noise_sound = noise_sound[:len(audio)]
        print(len(audio), len(noise_sound))

    if noise_sound.frame_rate != sr:
        noise_sound = noise_sound.set_frame_rate(sr)

    scale = [2, 4, 6, 8, 10]
    # noise_sound = noise_sound * scale[intensity-1]
    noise_sound = noise_sound.apply_gain(scale[intensity-1])

    output = audio.overlay(noise_sound)
    # output = audio + noise_sound
    # sf.write(output_path, output, sr)

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
        elif self.corruption == 'underwater':
            underwater(self.audio_paths[index], os.path.join(save_path, self.candi_audio_names[index]), self.severity)
        else:
            add_external_noise(self.audio_paths[index], os.path.join(self.noise_path, self.corruption + '.wav'), os.path.join(save_path, self.candi_audio_names[index]), self.severity)

        return 0 

    def __len__(self):
        return len(self.audio_paths)




ROOTDIR = '/people/cs/s/skm200005/UTD/AV-Robustness/'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--corruption', type=str, default='wind', choices=['all', 'gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise', 'snow', 'frost' , 'spatter', 'wind', 'concert', 'smoke'], help='Type of corruption to apply')
parser.add_argument('--severity', type=int, default=5, choices=[0, 1, 2, 3, 4, 5], help='Severity of corruption to apply')
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
    corruption_list = [
        'gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise', 
        'snow', 'frost' , 'spatter', 'wind', 
        'concert', 'smoke', 'rain', 
        'crowd', 'underwater', 'interference']
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