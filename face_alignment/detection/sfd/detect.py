import torch
import torch.nn.functional as F

import os
import sys
import cv2
import random
import datetime
import math
import argparse
import numpy as np

import scipy.io as sio
import zipfile
from .net_s3fd import s3fd
from .bbox import *


def detect(net, img, device):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    # Creates a batch of 1
    img = img.reshape((1,) + img.shape)

    img = torch.from_numpy(img).float().to(device)

    return batch_detect(net, img, device)


def batch_detect(net, img_batch, device):
    """
    Inputs:
        - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
    """

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    BB, CC, HH, WW = img_batch.size()

    with torch.no_grad():
        olist = net(img_batch.float())  # patched uint8_t overflow error

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    bboxlists = []

    olist = [oelem.data.cpu() for oelem in olist]

    for j in range(BB):
        bboxlist = []
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[j, 1, hindex, windex]
                loc = oreg[j, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])

        bboxlists.append(bboxlist)

    bboxlists = np.array(bboxlists)
    return bboxlists


def flip_detect(net, img, device):
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
