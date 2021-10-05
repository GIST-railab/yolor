import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from glob import glob

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


frame_root = '/HDD/accident_anticipation/Data/DoTA_P650N700/Frames'
save_to = '/HDD/accident_anticipation/Data/DoTA_P650N700/BBox'

tr_vids = glob(os.path.join(frame_root, 'train/*/'))
val_vids = glob(os.path.join(frame_root, 'val/*/'))

tr_vids.sort()
val_vids.sort()

all_vids = []
all_vids.extend(tr_vids)
all_vids.extend(val_vids)

weights = 'yolor_p6.pt'
cfg = 'cfg/yolor_p6.cfg'
device = '0'
img_size = 1280
conf = 0.5

device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

model = Darknet(cfg, img_size).cuda()
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()
if half:
    model.half()  # to FP16


names = 'data/coco.names'
names = load_classes(names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


def inferenceVideo(video, model, names):
    #frames = glob(os.path.join(tr_vids, '*.jpg'))
    #frames.sort()
    frames_vid = os.path.join(video, '*.jpg')
    result_txt = []

    dataset = LoadImages(frames_vid, img_size=img_size, auto_size=64)

    img = torch.zeros((1, 3, img_size, img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

