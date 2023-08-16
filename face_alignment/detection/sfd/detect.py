import torch
import torch.nn.functional as F

import cv2
import numpy as np

from .bbox import *


def detect(net, img, device):
    img = img.transpose(2, 0, 1)
    # Creates a batch of 1
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img.copy()).to(device, dtype=torch.float32)

    return batch_detect(net, img, device)


def batch_detect(net, img_batch, device):
    """
    Inputs:
        - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
    """

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    batch_size = img_batch.size(0)
    img_batch = img_batch.to(device, dtype=torch.float32)

    img_batch = img_batch.flip(-3)  # RGB to BGR
    img_batch = img_batch - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        olist = net(img_batch)  # patched uint8_t overflow error

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    olist = [oelem.data.cpu().numpy() for oelem in olist]

    bboxlists = get_predictions(olist, batch_size)
    return bboxlists


def get_predictions(olist, batch_size):
    bboxlists = []
    variances = [0.1, 0.2]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        stride = 2**(i + 2)    # 4,8,16,32,64,128
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            score = ocls[:, 1, hindex, windex][:,None]
            loc = oreg[:, :, hindex, windex].copy()
            boxes = decode(loc, priors, variances)
            bboxlists.append(np.concatenate((boxes, score), axis=1))
    
    if len(bboxlists) == 0: # No candidates within given threshold
        bboxlists = np.array([[] for _ in range(batch_size)])
    else:
        bboxlists = np.stack(bboxlists, axis=1)
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
