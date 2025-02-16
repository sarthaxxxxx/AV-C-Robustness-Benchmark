import os
import cv2
import os.path
import random
import numpy as np

import torch
import torchvision.transforms as trn
import torch.utils.data as data

from PIL import Image

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
# from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import map_coordinates
import warnings

warnings.simplefilter("ignore", UserWarning)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# /////////////// Some notes ///////////////
# Digital noise:
#   - Gaussian noise: random noise added to the image
#   - Shot noise: random noise that looks like dots
#   - Impulse noise: random noise that looks like salt and pepper
#   - Speckle noise: random noise that looks like grains
#   - [TODO] Compression noise: random noise due to compression artifacts
# Environmental:
#   - Snow: random snowflakes
#   - Frost: random frost
#   - Spatter: random splatters
#   - Wind: random wind sounds (audio) and motion blur (image)
#   - Rain: random raindrops
#   - Underwater - muffled audio and blue-green tint (image)
#   - [TODO] Occulusion: occlusion (image) + noise from the occuluded object (audio)
# Human-related:
#   - Concert: brightness (image) and concert sound (audio)
#   - Smoke: random smoke + smoke alarm (audio)
#   - Crowd: random crowd noise (audio) + occlusion (image)
#   - Interference: random rotation/zoom/shake (image) + interference/human speaking (audio)
# /////////////// End Display Results ///////////////

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, candi_images):
    images = []
    dir = os.path.expanduser(dir)
    # for name in sorted(os.listdir(dir)):
    for root, _, files in os.walk(dir):
        print("files: ", files)
        for name in sorted(files):
            if name in candi_images:
                path = os.path.join(dir, name)
                item = (path, name)
                images.append(item)
    print("images: ", len(images))

    return images


def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except FileNotFoundError:
        print(f"Warning: File not found: {path}")
        return None    


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DistortImageFolder(data.Dataset):
    def __init__(self, root, save_path, candi_images, method, severity, transform=None, loader=default_loader):
        self.root = root
        self.save_path = save_path
        self.candi_images = candi_images
        self.method = method
        self.severity = severity
        self.transform = transform

    def __getitem__(self, index):
        # Get the frame file name
        frame_name = self.candi_images[index]
        frame_path = os.path.join(self.root, frame_name)  # Construct the full path

        # Load the image
        img = pil_loader(frame_path)
        if img is None:
            return None  # Skip if the file is not found

        # Apply the distortion method
        distorted_img = self.method(img, self.severity)

        if isinstance(distorted_img, np.ndarray):
            distorted_img = PILImage.fromarray(np.uint8(distorted_img))

        # Apply transformations (if any)
        if self.transform is not None:
            distorted_img = self.transform(distorted_img)

        # Save the distorted image
        save_dir = os.path.join(self.save_path, '')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, frame_name)
        os.makedirs(save_path.split('frame_')[0], exist_ok=True)
        print("save_path: ", save_path, "index: ", index, "len: ", len(self.candi_images))
        distorted_img.save(save_path, quality=85, optimize=True)

        return 0  # Return a dummy value (since we're not training)

    def __len__(self):
        return len(self.candi_images)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def crowd(x, severity = 1):
    """Simulate occlusions/shadow in an image."""
    x = np.array(x) / 255.
    h, w, _ = x.shape
    c = [20, 40, 80, 100, 120][severity - 1]
    x_bb, y_bb = random.randint(0, w - c), random.randint(0, h - c)
    x_bb_end, y_bb_end = x_bb + c, y_bb + c
    x[y_bb:y_bb_end, x_bb:x_bb_end] = 0.3
    return np.clip(x, 0, 1) * 255


def interference(x, severity):
    """
    Random rotation of the image.
    """
    deg = random.randint(-severity*6-5, severity*6+5)
    return x.rotate(deg)

####################################### DIGITAL NOISE #######################################
def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255
##############################################################################


####################################### Environmental #######################################
def snow(x, severity=1):
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
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

def wind(x, severity=1):
    '''Simulate wind effect on the image - motion blur.'''
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
                        cv2.IMREAD_UNCHANGED)
    # x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
    #                  cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


##############################################################################################

def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


def underwater(x, severity = 1):
    """Simulate underwater effect on the image - blue-green tint."""
    x = np.array(x) / 255.
    tint_values = {
        1: np.array([1.0, 0.9, 0.8]),
        2: np.array([1.0, 0.85, 0.7]),
        3: np.array([1.0, 0.8, 0.6]),
        4: np.array([0.9, 0.7, 0.5]),
        5: np.array([0.8, 0.6, 0.4])
    }
    blue_tint = tint_values.get(severity, tint_values[3])
    x = np.clip(x * blue_tint, 0, 1)

    # light_scatter = {1: 3, 2: 5, 3: 9, 4: 13, 5: 17}
    # light_scatter = light_scatter.get(severity, light_scatter[3])
    # x = cv2.GaussianBlur(x, (light_scatter, light_scatter), light_scatter)
    return np.clip(x, 0, 1) * 255  

def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255

# def wind(x, severity=1):
#     '''Simulate wind effect on the image - motion blur.'''
#     c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

#     output = BytesIO()
#     x.save(output, format='PNG')
#     x = MotionImage(blob=output.getvalue())

#     x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

#     x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
#                         cv2.IMREAD_UNCHANGED)
#     # x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#     #                  cv2.IMREAD_UNCHANGED)

#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def smoke(x, severity=1):
    '''Simulate smoke effect on the image - grayish appearance.'''
    import numpy as np
    from scipy.ndimage import gaussian_filter

    # Smoke parameters based on severity
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    # Normalize the image
    x = np.array(x) / 255.0
    max_val = x.max()

    # Generate smoke-like noise (Gaussian blobs)
    noise = np.random.rand(224, 224)
    smoke_pattern = gaussian_filter(noise, sigma=c[1])

    # Add color tint for smoke (grayish appearance)
    smoke_pattern = np.expand_dims(smoke_pattern, axis=-1)  # Add channel dimension
    smoke_pattern = np.repeat(smoke_pattern, 3, axis=-1)    # Match RGB channels

    # Blend smoke with the image
    x += c[0] * smoke_pattern

    # Normalize and clip
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    frost_dir = '/people/cs/s/skm200005/UTD/AV-Robustness/data/VGGSound'
    filename = [f"{frost_dir}/frost{1}.png", f"{frost_dir}/frost{2}.png", f"{frost_dir}/frost{3}.png", f"{frost_dir}/frost{4}.jpg", f"{frost_dir}/frost{5}.jpg", f"{frost_dir}/frost{6}.jpg"][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)




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

def rain(x, severity=1):
    """
    Similar to your snow() function but adds watery, bluish raindrops.
    x: input image as a NumPy array (H, W, 3), e.g., (224, 224, 3). dtype=uint8
    severity: integer 1-5 controlling how intense the effect is.
    """

    # Each tuple defines:
    # (loc, scale, zoom_factor, threshold, motion_blur_radius, motion_blur_sigma, alpha_blend)
    # Tweak these to your liking.
    c = [
        (0.05, 0.2, 3,   0.3,  10, 4,  0.8),
        (0.05, 0.2, 3.5, 0.3,  10, 5,  0.7),
        (0.1,  0.2, 4,   0.25, 12, 6,  0.7),
        (0.1,  0.2, 4.5, 0.25, 15, 8,  0.65),
        (0.1,  0.25,5,   0.2,  18,10, 0.6),
    ][severity - 1]

    # Convert x to float [0..1]
    x = np.array(x, dtype=np.float32) / 255.0

    # 1) Create random "raindrop" noise (monochrome)
    rain_layer = np.random.normal(loc=c[0], scale=c[1], size=x.shape[:2])

    # 2) Zoom in/out to cluster/spread droplets
    rain_layer = clipped_zoom(rain_layer[..., np.newaxis], c[2])

    # 3) Threshold out smaller values to form discrete drops
    rain_layer[rain_layer < c[3]] = 0.0
    rain_layer = np.clip(rain_layer, 0, 1)

    # 4) Convert to a PIL image -> use Wand for motion blur
    pil_rain = PILImage.fromarray((rain_layer.squeeze() * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    pil_rain.save(output, format='PNG')
    wand_rain = MotionImage(blob=output.getvalue())

    # Random angle for slanted raindrops
    angle = random.uniform(-120, -60)
    wand_rain.motion_blur(radius=c[4], sigma=c[5], angle=angle)

    # 5) Convert back to NumPy [0..1], single-channel
    # Some versions of wand might raise a warning about fromstring
    rain_np = cv2.imdecode(np.frombuffer(wand_rain.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    if rain_np is None:
        # fallback in case decode fails
        rain_np = (rain_layer[...,0] * 255).astype(np.uint8)
    else:
        # might contain alpha or multiple channels
        if len(rain_np.shape) == 3 and rain_np.shape[2] > 1:
            rain_np = rain_np[..., 0]  # take first channel
    rain_np = rain_np.astype(np.float32) / 255.0

    # 6) Add blue tint: expand single channel to 3 channels with a mild (R, G, B) scale
    rain_color = np.stack([rain_np * 0.5, rain_np * 0.9, rain_np * 1.2], axis=-1)
    rain_color = np.clip(rain_color, 0, 1)

    # 7) Blend with original
    alpha = c[6]
    gray_x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1)
    # Slight brightness correction on darker zones
    x = alpha * x + (1 - alpha) * np.maximum(x, gray_x * 1.3 + 0.3)

    # 8) Add the streaks in forward orientation and a 180° rotation
    #    to add more scattered droplets
    final = np.clip(x + rain_color + np.rot90(rain_color, k=2), 0, 1) * 255.0
    return final.astype(np.uint8)


def spatter(x, severity=1):
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

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
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

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255

def concert(x, severity=1):
    '''Simulate concert effect on the image - brightness.'''
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    '''Simulate saturation effect on the image.'''
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

def save_distorted_for_image(method, candi_image_names, severity, data_path, save_path):
    print(f"Applying {method.__name__} with severity {severity}")

    # Create the distorted dataset
    distorted_dataset = DistortImageFolder(
        root=data_path,  # Use the base data path directly
        save_path=save_path,
        candi_images=candi_image_names,  # List of frame file names (e.g., "frame_0000076312.jpg")
        method=method,
        severity=severity,
        transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224)])
    )
    print("distorted_dataset: ", distorted_dataset)

    # Create the DataLoader
    distorted_dataset_loader = torch.utils.data.DataLoader(
        distorted_dataset, batch_size=128, shuffle=False, num_workers=4
    )

    print("distorted_dataset_loader: ", distorted_dataset_loader)

    # Process the frames
    for i, _ in enumerate(distorted_dataset_loader):
        print(i)
        continue

import collections
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--corruption', type=str, default='gaussian_noise', choices=['gaussian_noise'], help='Type of corruption to apply')
parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5, 0], help='Severity of corruption to apply, 0: all')
parser.add_argument('--data_path', type=str, help='Path to test data')
parser.add_argument('--save_path', type=str, help='Path to store corruption data')
args = parser.parse_args()


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

d = collections.OrderedDict()
if args.corruption == 'all':
    d['Gaussian Noise'] = gaussian_noise
    d['Shot Noise'] = shot_noise
    d['Impulse Noise'] = impulse_noise
    d['Speckle Noise'] = speckle_noise
    d['Snow'] = snow
    d['Frost'] = frost
    d['Spatter'] = spatter
    d['Wind'] = wind
    d['Concert'] = concert
    d['Smoke'] = smoke
    d['Crowd'] = crowd
    d['Interference'] = interference
    d['Underwater'] = underwater
    d['Rain'] = rain
else:
    d[args.corruption] = eval(args.corruption.lower())

# dir = os.path.join(args.data_path, '')
# candi_image_names = os.listdir(dir)
def get_image_files(base_dir):
    image_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.jpg'):  # Only include .jpg files
                image_files.append(os.path.relpath(os.path.join(root, file), base_dir))
    return image_files

# Example usage
base_dir = "/home/jovyan/EPIC-KITCHENS"
candi_image_names = get_image_files(base_dir)
print(f"Found {len(candi_image_names)} image files.")

for method_name in d.keys():
    save_distorted_for_image(d[method_name], candi_image_names, args.severity, args.data_path, args.save_path)

