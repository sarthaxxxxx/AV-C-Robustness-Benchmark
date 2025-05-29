import os
import torch
import torchaudio.functional as F
import torchaudio
import numpy as np
from scipy.signal import butter, lfilter
import random
from PIL import Image
import cv2
import skimage as sk
from io import BytesIO
from wand.api import library as wandlibrary
import ctypes
from wand.image import Image as WandImage
from skimage.filters import gaussian

from scipy.ndimage import gaussian_filter

import warnings
warnings.simplefilter("ignore", UserWarning)

# Helper functions
def clipped_zoom(img, zoom_factor):
    """
    Example placeholder for the clipped_zoom function,
    which zooms in on the center of the image by zoom_factor.
    You can replace this with your actual clipped_zoom implementation.
    """
    # naive example for demonstration:
    # zoom in by resizing + cropping to the original shape
    h, w, c = img.shape
    # new size
    zoom_h, zoom_w = int(h * zoom_factor), int(w * zoom_factor)
    # resize
    resized = cv2.resize(img, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
    # center-crop back
    start_h = (zoom_h - h) // 2
    start_w = (zoom_w - w) // 2
    return resized[start_h:start_h+h, start_w:start_w+w]

wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def add_env_noise(waveform, sample_rate, intensity, noise_dir=''):
    NOISE_SNRS = [20, 15, 10, 5, 0]
    assert sample_rate == 16000, "Noise file must have a sample rate of 16 kHz."
    xlen = waveform.shape[-1]
    noise_files = os.listdir(noise_dir)
    if not noise_files:
        raise FileNotFoundError("No noise files found in the directory.")
    # Pick and load noise
    noise_file = np.random.choice(noise_files)
    noise_path = os.path.join(noise_dir, noise_file)
    noise_raw, sr_noise = torchaudio.load(noise_path)
    # Resample if needed
    if sr_noise != sample_rate:
        noise_raw = torchaudio.transforms.Resample(sr_noise, sample_rate)(noise_raw)
    # Repeat noise if too short
    while noise_raw.shape[-1] < xlen:
        noise_raw = torch.cat([noise_raw, noise_raw], dim=-1)
    noise = noise_raw[..., :xlen]
    snr_db = NOISE_SNRS[intensity - 1]
    snr_tensor = torch.full_like(waveform[:, 0], snr_db)
    waveform_noisy = F.add_noise(waveform, noise, snr_tensor)
    return waveform_noisy

NOISE_SNRS = [40, 30, 20, 10, 0]


# Gaussian
def gaussian_visual(x, severity=5):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    x = Image.fromarray(x.astype(np.uint8))
    return x

def gaussian_audio(waveform, intensity=5):
    noise = torch.randn_like(waveform)
    snr_tensor = torch.tensor([intensity], dtype=waveform.dtype, device=waveform.device).expand(waveform.shape[:-1])
    return F.add_noise(waveform, noise, snr_tensor)

# Impulse
def impulse_visual(x, severity=5):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    x = np.clip(x, 0, 1) * 255
    x = Image.fromarray(x.astype(np.uint8))
    return x

def impulse_audio(waveform, intensity=5, impulse_prob=0.05):
    snr_level = NOISE_SNRS[intensity - 1]
    signal_power = torch.mean(waveform ** 2)
    random_mask = torch.rand_like(waveform)
    salt_pepper = torch.where(
        random_mask < (impulse_prob / 2), -1.0,
        torch.where(random_mask > (1 - impulse_prob / 2), 1.0, 0.0)
    )
    noise_power = torch.mean(salt_pepper ** 2) + 1e-8
    snr_linear = 10 ** (snr_level / 10)
    noise_scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    salt_pepper = noise_scaling_factor * salt_pepper
    y_noisy = torch.clamp(waveform + salt_pepper, -1.0, 1.0)
    return y_noisy

# Shot
def shot_visual(x, severity=5):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255
    x = Image.fromarray(x.astype(np.uint8))
    return x

def shot_audio(waveform, intensity=5):
    snr_level = NOISE_SNRS[intensity - 1]
    signal_power = torch.mean(waveform ** 2)
    y_min, y_max = waveform.min(), waveform.max()
    range_val = y_max - y_min
    if range_val == 0:
        y_norm = torch.full_like(waveform, fill_value=0.5)
    else:
        y_norm = (waveform - y_min) / (y_max - y_min)  # Normalize to [0, 1]
    poisson_noise = torch.poisson(y_norm * 50) / 50  # Poisson base noise
    noise = poisson_noise - y_norm  # Center to zero mean
    noise_power = torch.mean(noise ** 2) + 1e-8
    snr_linear = 10 ** (snr_level / 10)
    noise_scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    noise = noise_scaling_factor * noise
    y_noisy = torch.clamp(waveform + noise, -1.0, 1.0)
    return y_noisy

# Speckle
def speckle_visual(x, severity=5):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    x = Image.fromarray(x.astype(np.uint8))
    return x

def speckle_audio(waveform, intensity=5):
    snr_level = NOISE_SNRS[intensity - 1]
    signal_power = torch.mean(waveform ** 2)
    noise = torch.randn_like(waveform)
    noise = noise * waveform  # Speckle noise (multiplicative)
    noise_power = torch.mean(noise ** 2) + 1e-8
    snr_linear = 10 ** (snr_level / 10)
    noise_scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))
    noise = noise_scaling_factor * noise
    y_noisy = torch.clamp(waveform + noise, -1.0, 1.0)
    return y_noisy

# Compression
def compression_visual(x, severity=5):
    if isinstance(x, np.ndarray):
        x = Image.fromarray(x)
    c = [25, 18, 15, 10, 7][severity - 1]
    buffer = BytesIO()
    x.save(buffer, format='JPEG', quality=c)
    buffer.seek(0)
    compressed = Image.open(buffer).copy()
    buffer.close()

    return compressed

def compression_audio(waveform, sample_rate=16000, intensity=5):
    cutoff_freqs = [8000, 6000, 4000, 2000, 1000]  # Lower = more degradation
    cutoff = cutoff_freqs[intensity - 1]
    def butter_lowpass(cutoff, sr, order=6):
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    b, a = butter_lowpass(cutoff, sample_rate)
    filtered_waveform = torch.tensor([lfilter(b, a, ch.numpy()) for ch in waveform], dtype=waveform.dtype)
    return filtered_waveform

# Snow
def snow_visual(x, severity=5):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
    x = np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    x = Image.fromarray(x.astype(np.uint8))
    return x

def snow_audio(waveform, intensity=5):    
    snow_dir = './corruptions/audio_corruptions/snow'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=snow_dir)

# Frost
def frost_visual(x, severity=5):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    frost_dir = './corruptions/visual_corruptions'
    filename = [f"{frost_dir}/frost{1}.png", f"{frost_dir}/frost{2}.png", f"{frost_dir}/frost{3}.png", f"{frost_dir}/frost{4}.jpg", f"{frost_dir}/frost{5}.jpg", f"{frost_dir}/frost{6}.jpg"][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]
    x = np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)
    x = Image.fromarray(x.astype(np.uint8))
    return x

def frost_audio(waveform, intensity=5):
    frost_dir = './corruptions/audio_corruptions/frost'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=frost_dir)

# Spatter
def spatter_visual(x, severity=5):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        x = Image.fromarray(x.astype(np.uint8))

        return x
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        x = np.clip(x + color, 0, 1) * 255
        x = Image.fromarray(x.astype(np.uint8))
        return x

def spatter_audio(waveform, intensity=5):
    spatter_dir = './corruptions/audio_corruptions/water_drops'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=spatter_dir)

# Wind
def wind_visual(x, severity=5):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    x = np.array(x) / 255.

    x = Image.fromarray(x)

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
                        cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        x = np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        x = Image.fromarray(x.astype(np.uint8))
        return x
    else:  # greyscale to RGB
        x = np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
        x = Image.fromarray(x.astype(np.uint8))
        return x

def wind_audio(waveform, intensity=5):
    wind_dir = './corruptions/audio_corruptions/wind'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=wind_dir)

# Rain
def rain_visual(x, severity=5):
    c = [
        (0.05, 0.2, 3,   0.3,  10, 4,  0.8),
        (0.05, 0.2, 3.5, 0.3,  10, 5,  0.7),
        (0.1,  0.2, 4,   0.25, 12, 6,  0.7),
        (0.1,  0.2, 4.5, 0.25, 15, 8,  0.65),
        (0.1,  0.25,5,   0.2,  18,10, 0.6),
    ][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    rain_layer = np.random.normal(loc=c[0], scale=c[1], size=x.shape[:2])
    rain_layer = clipped_zoom(rain_layer[..., np.newaxis], c[2])
    rain_layer[rain_layer < c[3]] = 0.0
    rain_layer = np.clip(rain_layer, 0, 1)
    pil_rain = Image.fromarray((rain_layer.squeeze() * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    pil_rain.save(output, format='PNG')
    wand_rain = MotionImage(blob=output.getvalue())
    angle = random.uniform(-120, -60)
    wand_rain.motion_blur(radius=c[4], sigma=c[5], angle=angle)

    rain_np = cv2.imdecode(np.frombuffer(wand_rain.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    if rain_np is None:
        rain_np = (rain_layer[...,0] * 255).astype(np.uint8)
    else:
        if len(rain_np.shape) == 3 and rain_np.shape[2] > 1:
            rain_np = rain_np[..., 0]  # take first channel
    rain_np = rain_np.astype(np.float32) / 255.0
    rain_color = np.stack([rain_np * 0.5, rain_np * 0.9, rain_np * 1.2], axis=-1)
    rain_color = np.clip(rain_color, 0, 1)
    alpha = c[6]
    gray_x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1)
    x = alpha * x + (1 - alpha) * np.maximum(x, gray_x * 1.3 + 0.3)
    final = np.clip(x + rain_color + np.rot90(rain_color, k=2), 0, 1) * 255.0
    x = Image.fromarray(final.astype(np.uint8))
    return x

def rain_audio(waveform, intensity=5):
    rain_dir = './corruptions/audio_corruptions/rain'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=rain_dir)

# Underwater
def underwater_visual(x, severity=5):
    x = np.array(x)
    img = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    blur_levels = {1: 3, 2: 5, 3: 7, 4: 10, 5: 15}  # Blur kernel size
    red_reduction = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3, 5: 0.1}  # % of red channel kept
    contrast_factors = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}  # Contrast reduction
    haze_factors = {1: 20, 2: 40, 3: 60, 4: 80, 5: 100}  # White overlay intensity

    blur_k = blur_levels[severity]
    red_factor = red_reduction[severity]
    contrast = contrast_factors[severity]
    haze_intensity = haze_factors[severity]

    img_underwater = img.astype(np.float32)
    img_underwater[:, :, 0] *= red_factor  # Reduce red
    img_underwater = np.clip(img_underwater, 0, 255).astype(np.uint8)
    img_blurred = cv2.GaussianBlur(img_underwater, (blur_k, blur_k), 0)
    img_low_contrast = cv2.convertScaleAbs(img_blurred, alpha=contrast, beta=0)
    haze = np.full_like(img_low_contrast, (haze_intensity, haze_intensity, haze_intensity), dtype=np.uint8)
    img_hazy = cv2.addWeighted(img_low_contrast, 0.85, haze, 0.15, 0)
    x = Image.fromarray(img_hazy.astype(np.uint8))
    return  x

def underwater_audio(waveform, intensity=5):
    underwater_dir = './corruptions/audio_corruptions/underwater'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=underwater_dir)

# Concert
def concert_visual(x, severity=5):
    '''Simulate concert effect on the image - brightness.'''
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    x = np.clip(x, 0, 1) * 255
    x = Image.fromarray(x.astype(np.uint8))
    return x

def concert_audio(waveform, intensity=5):
    concert_dir = './corruptions/audio_corruptions/concert'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=concert_dir)

# Smoke
def smoke_visual(x, severity=5):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    x = np.array(x) / 255.0
    max_val = x.max()
    noise = np.random.rand(224, 224)
    smoke_pattern = gaussian_filter(noise, sigma=c[1])
    smoke_pattern = np.expand_dims(smoke_pattern, axis=-1)  # Add channel dimension
    smoke_pattern = np.repeat(smoke_pattern, 3, axis=-1)    # Match RGB channels
    x += c[0] * smoke_pattern
    x = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
    x = Image.fromarray(x.astype(np.uint8))
    return x

def smoke_audio(waveform, intensity=5):
    smoke_dir = './corruptions/audio_corruptions/crackling_fire_and_siren'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=smoke_dir)


# Crowd
def crowd_visual(img_np, severity=5):
    img_np = np.array(img_np)
    occlusion_dir = './corruptions/visual_corruptions/crowd_image'
    if not isinstance(img_np, np.ndarray):
        raise TypeError("Expected input image as NumPy array")
    h, w, _ = img_np.shape
    occlusion_sizes = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
    occlusion_scale = occlusion_sizes[severity]
    occlusion_files = [f for f in os.listdir(occlusion_dir) if f.endswith('.png')]
    if not occlusion_files:
        raise FileNotFoundError("No occlusion images found in the specified directory.")
    occlusion_image_path = os.path.join(occlusion_dir, random.choice(occlusion_files))
    occlusion_image = cv2.imread(occlusion_image_path, cv2.IMREAD_UNCHANGED)
    occlusion_h, occlusion_w = int(h * occlusion_scale), int(w * occlusion_scale)
    occlusion_image = cv2.resize(occlusion_image, (occlusion_w, occlusion_h))
    x_offset = random.randint(0, w - occlusion_w)
    y_offset = random.randint(0, h - occlusion_h)
    overlay = img_np.copy()
    if occlusion_image.shape[2] == 4:  # RGBA (with alpha)
        alpha_mask = occlusion_image[:, :, 3] / 255.0
        for c in range(3):
            overlay[y_offset:y_offset+occlusion_h, x_offset:x_offset+occlusion_w, c] = (
                (1 - alpha_mask) * overlay[y_offset:y_offset+occlusion_h, x_offset:x_offset+occlusion_w, c] +
                alpha_mask * occlusion_image[:, :, c]
            )
    else:
        overlay[y_offset:y_offset+occlusion_h, x_offset:x_offset+occlusion_w] = occlusion_image[:, :, :3]

    x = Image.fromarray(overlay.astype(np.uint8))
    return x

def crowd_audio(waveform, intensity=5):
    crowd_dir = './corruptions/audio_corruptions/crowd_speech'
    return add_env_noise(waveform, sample_rate=16000, intensity=intensity, noise_dir=crowd_dir)

# Interference
def interference_visual(x, severity=5):
    x = np.array(x)
    x = Image.fromarray(x)
    deg = random.randint(-severity*6-5, severity*6+5)
    x = x.rotate(deg)
    return x

def interference_audio(waveform, sr=16000, intensity=5):
    '''
    Add silences to the audio
    '''
    severity_levels = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}  # Fraction of audio to be muted
    mask_fraction = severity_levels[intensity]
    total_samples = waveform.shape[1]
    num_mask_samples = int(total_samples * mask_fraction)
    num_chunks = intensity * 2  # Increase the number of chunks with severity
    chunk_size = num_mask_samples // num_chunks  # Each silent chunk size
    chunk_size = max(chunk_size, sr // 10)  # At least 0.1 sec per chunk
    mask_positions = random.sample(range(total_samples - chunk_size), num_chunks)
    for pos in mask_positions:
        waveform[:, pos:pos + chunk_size] = 0  # Zero out chunk
    return waveform

corruption_dict = {
    'gaussian': (gaussian_visual, gaussian_audio),
    'impulse': (impulse_visual, impulse_audio),
    'shot': (shot_visual, shot_audio),
    'speckle': (speckle_visual, speckle_audio),
    'compression': (compression_visual, compression_audio),
    'snow': (snow_visual, snow_audio),
    'frost': (frost_visual, frost_audio),
    'spatter': (spatter_visual, spatter_audio),
    'wind': (wind_visual, wind_audio),
    'rain': (rain_visual, rain_audio),
    'underwater': (underwater_visual, underwater_audio),
    'concert': (concert_visual, concert_audio),
    'smoke': (smoke_visual, smoke_audio),
    'crowd': (crowd_visual, crowd_audio),
    'interference': (interference_visual, interference_audio )
}
