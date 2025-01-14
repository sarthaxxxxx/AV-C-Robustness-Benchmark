import numpy as np
import cv2
from skimage.filters import gaussian
from pydub import AudioSegment

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
    
############## audio corruptions  ####################################

def weather_noise_a(audio_raw, weather_audio_path, intensity):
    audio = AudioSegment(
        audio_raw.tobytes(),
        frame_rate=16_000,
        sample_width=audio_raw.dtype.itemsize,
        channels=1
    )
    rain_sound = AudioSegment.from_file(weather_audio_path)

    if len(audio) <= len(rain_sound):
        rain_sound = rain_sound[:len(audio)]
    else:
        num_repeats = len(audio) // len(rain_sound) + 1
        rain_sound = rain_sound * num_repeats
        rain_sound = rain_sound[:len(audio)]

    scale = [1, 2, 4, 6, 8]
    rain_sound = rain_sound.apply_gain(scale[intensity - 1])
    output = audio.overlay(rain_sound)
    output_samples = np.array(output.get_array_of_samples()).astype(np.float32)
    max_amplitude = max(np.max(np.abs(output_samples)), 1.0)
    output_samples = output_samples / max_amplitude

    return output_samples

def spatter_noise_a(audio, intensity):
    spatter_path = '.external_audio/spatter.wav'
    return weather_noise_a(audio, spatter_path, intensity)