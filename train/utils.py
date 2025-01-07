import os
import shutil
import librosa
import cv2
import subprocess as sp
from threading import Timer
import copy
import numpy as np
import torch
from torch.autograd import Variable

def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()