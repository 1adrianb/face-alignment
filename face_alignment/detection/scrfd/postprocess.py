"""SCRFD postprocessing utilities.

Adapted from https://github.com/deepinsight/insightface (MIT License).
"""
import numpy as np


def distance2bbox(points, distance):
    """Decode distance predictions to bounding boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def generate_anchor_centers(height, width, stride, num_anchors=2):
    """Generate anchor center points for a feature map."""
    anchor_centers = np.stack(
        np.mgrid[:height, :width][::-1], axis=-1
    ).astype(np.float32)
    anchor_centers = (anchor_centers * stride).reshape((-1, 2))
    if num_anchors > 1:
        anchor_centers = np.stack(
            [anchor_centers] * num_anchors, axis=1
        ).reshape((-1, 2))
    return anchor_centers


def nms(dets, threshold):
    """Standard NMS on (N, 5) array of [x1, y1, x2, y2, score]."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep
