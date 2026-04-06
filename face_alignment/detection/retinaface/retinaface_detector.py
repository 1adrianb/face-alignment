import torch
import numpy as np
import cv2

from ..core import FaceDetector
from ...utils import load_file_from_url
from .net_retinaface import RetinaFace, cfg_mnet
from .prior_box import PriorBox
from .box_utils import decode, py_cpu_nms

# Weights from https://github.com/biubug6/Pytorch_Retinaface
# Original Google Drive: https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1
models_urls = {
    'retinaface_mobilenet0.25':
        'https://www.adrianbulat.com/downloads/python-fan/retinaface_mobilenet0.25.pth',
}


class RetinaFaceDetector(FaceDetector):
    """RetinaFace detector with MobileNet0.25 backbone.

    Pure PyTorch implementation. Supports CPU, CUDA, and MPS.
    Adapted from https://github.com/biubug6/Pytorch_Retinaface (MIT License).
    """

    def __init__(self, device, path_to_detector=None, verbose=False,
                 confidence_threshold=0.5, nms_threshold=0.4):
        super(RetinaFaceDetector, self).__init__(device, verbose)

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.cfg = cfg_mnet
        self.variance = cfg_mnet['variance']

        net = RetinaFace(cfg=cfg_mnet)

        if path_to_detector is None:
            state_dict = torch.load(
                load_file_from_url(models_urls['retinaface_mobilenet0.25']),
                map_location=device, weights_only=True)
        else:
            state_dict = torch.load(path_to_detector, map_location=device, weights_only=True)

        # Handle state dicts with 'module.' prefix from DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()
        self.face_detector = net

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        img = np.float32(image)
        img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)

        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            loc, conf, _ = self.face_detector(img)

        im_height, im_width = image.shape[:2]
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)

        boxes = decode(loc.squeeze(0), priors, self.variance)
        boxes = boxes * scale
        scores = conf.squeeze(0)[:, 1]

        # Filter by confidence
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if boxes.shape[0] == 0:
            return []

        # NMS
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        dets = np.hstack((boxes_np, scores_np[:, np.newaxis])).astype(np.float32)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep]

        return dets.tolist()

    @property
    def reference_scale(self):
        return 165

    def detect_from_batch(self, tensor):
        bboxlists = []
        for i in range(tensor.shape[0]):
            image = tensor[i].cpu().numpy().transpose(1, 2, 0)
            bboxlists.append(self.detect_from_image(image))
        return bboxlists
