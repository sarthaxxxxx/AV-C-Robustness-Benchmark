import numpy as np
import skimage as sk
from scipy.ndimage import zoom as scizoom
from .weather_distortions import spatter_noise_v, spatter_noise_a, snow_noise_a, frost_noise_a, frost_noise_v, snow_noise_v

# visual corruptions

def gaussian_noise_v(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.  # Normalize to [0, 1]
    noisy_x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_x.astype(np.uint8)

def shot_noise_v(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    noisy_x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255
    return noisy_x.astype(np.uint8)

def impulse_noise_v(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    noisy_x = np.clip(x, 0, 1) * 255
    return noisy_x.astype(np.uint8)

def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    noisy_x = np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
    return noisy_x.astype(np.uint8)

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]

############## audio corruptions  ####################################
def gaussian_noise_a(audio, intensity):
    noise_std = [.08, .12, 0.18, 0.26, 0.38][intensity - 1]
    noise = np.random.normal(0, noise_std, len(audio))
    audio_with_noise = audio + noise
    max_amplitude = max(np.max(np.abs(audio_with_noise)), 1.0)  # Avoid divide-by-zero
    audio_with_noise = audio_with_noise / max_amplitude
    return audio_with_noise

def shot_noise_a(audio, intensity):
    c = [60, 25, 12, 5, 3][intensity - 1]
    noise = np.random.poisson(lam=c, size=len(audio)) / c
    audio_with_noise = audio + noise
    max_amplitude = max(np.max(np.abs(audio_with_noise)), 1.0)  # Avoid divide-by-zero
    audio_with_noise = audio_with_noise / max_amplitude
    return audio_with_noise

def impulse_noise_a(audio, intensity):
    c = [.03, .06, .09, 0.17, 0.27][intensity - 1]
    audio_with_noise = sk.util.random_noise(audio, mode='s&p', amount=c)
    max_amplitude = max(np.max(np.abs(audio_with_noise)), 1.0)  # Avoid divide-by-zero
    audio_with_noise = audio_with_noise / max_amplitude
    return audio_with_noise

def speckle_noise_a(audio, intensity):
    noise_std = [.15, .2, 0.35, 0.45, 0.6][intensity - 1]
    noise = np.random.normal(0, noise_std, len(audio))
    audio_with_noise = audio + audio * noise
    max_amplitude = max(np.max(np.abs(audio_with_noise)), 1.0)  # Avoid divide-by-zero
    audio_with_noise = audio_with_noise / max_amplitude
    return audio_with_noise

noise_function_map_v = {
    'gaussian': gaussian_noise_v,
    'shot': shot_noise_v,
    'impulse': impulse_noise_v,
    'speckle': speckle_noise,
    'spatter': spatter_noise_v,
    'snow': snow_noise_v,
    'frost': frost_noise_v,
}

noise_function_map_a = {
    'gaussian': gaussian_noise_a,
    'shot': shot_noise_a,
    'impulse': impulse_noise_a,
    'speckle': speckle_noise_a,
    'spatter': spatter_noise_a,
    'snow': snow_noise_a,
    'frost': frost_noise_a,
}