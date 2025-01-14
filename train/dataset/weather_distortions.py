import numpy as np
import cv2
from skimage.filters import gaussian
from pydub import AudioSegment
import soundfile as sf
from scipy.ndimage import zoom as scizoom
# from wand.image import Image as WandImage
# from wand.api import library as wandlibrary
import os
# import ctypes
# from PIL import Image as PILImage
# from io import BytesIO

# wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
#                                               ctypes.c_double,  # radius
#                                               ctypes.c_double,  # sigma
#                                               ctypes.c_double)  # angle

def spatter_noise_v(x, severity=1):
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

        noise_img = cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        return noise_img.astype(np.uint8)
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        noisy_x = np.clip(x + color, 0, 1) * 255
        return noisy_x.astype(np.uint8)

def frost_noise_v(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    frost_dir = './dataset/external_image'
    filename = [f"{frost_dir}/frost{1}.png", f"{frost_dir}/frost{2}.png", f"{frost_dir}/frost{3}.png", f"{frost_dir}/frost{4}.jpg", f"{frost_dir}/frost{5}.jpg", f"{frost_dir}/frost{6}.jpg"][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255).astype(np.uint8)    

# class MotionImage(WandImage):
#     def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
#         wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

def snow_noise_v(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
    noise_image = np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    return noise_image.astype(np.uint8)

############## audio corruptions  ####################################

def weather_noise_a(audio_raw, weather_audio_path, intensity):
    temp_audio_path = '/mnt/user/saksham/AV_robust/AV-C-Robustness-Benchmark/train/temp_samples/temp.wav'
    sf.write(temp_audio_path, audio_raw, 16_000)
    audio = AudioSegment.from_file(temp_audio_path, frame_rate=16_000, channels=1)
    os.remove(temp_audio_path)
    rain_sound = AudioSegment.from_file(weather_audio_path, frame_rate=16_000, channels=1)

    if len(audio) <= len(rain_sound):
        rain_sound = rain_sound[:len(audio)]
    else:
        num_repeats = len(audio) // len(rain_sound) + 1
        rain_sound = rain_sound * num_repeats
        rain_sound = rain_sound[:len(audio)]

    scale = [1, 2, 4, 6, 8]
    rain_sound = rain_sound.apply_gain(scale[intensity - 1])
    output = audio.overlay(rain_sound)
    
    output_samples = np.array(output.get_array_of_samples()).astype(np.float32)[:len(audio_raw)]
    max_amplitude = max(np.max(np.abs(output_samples)), 1.0)
    output_samples = output_samples / max_amplitude

    return output_samples

def spatter_noise_a(audio, intensity):
    spatter_path = './dataset/external_audio/spatter.wav'
    return weather_noise_a(audio, spatter_path, intensity)

def snow_noise_a(audio, intensity):
    spatter_path = './dataset/external_audio/snow.wav'
    return weather_noise_a(audio, spatter_path, intensity)

def frost_noise_a(audio, intensity):
    spatter_path = './dataset/external_audio/frost.wav'
    return weather_noise_a(audio, spatter_path, intensity)