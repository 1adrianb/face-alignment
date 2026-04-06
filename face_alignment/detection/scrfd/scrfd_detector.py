import warnings
import cv2
import numpy as np

from ..core import FaceDetector
from ...utils import load_file_from_url
from .postprocess import distance2bbox, generate_anchor_centers, nms

model_url = (
    'https://github.com/yakhyo/facial-analysis/releases/'
    'download/v0.0.1/det_2.5g.onnx'
)


class SCRFDDetector(FaceDetector):
    """SCRFD face detector using ONNX Runtime.

    Requires the ``onnxruntime`` package (install with ``pip install onnxruntime``
    or ``pip install onnxruntime-gpu`` for GPU support).

    Adapted from https://github.com/deepinsight/insightface (MIT License).
    """

    def __init__(self, device, path_to_detector=None, verbose=False,
                 confidence_threshold=0.5, nms_threshold=0.4):
        super(SCRFDDetector, self).__init__(device, verbose)

        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "SCRFD detector requires 'onnxruntime'. "
                "Install it with: pip install onnxruntime "
                "(or onnxruntime-gpu for GPU support)"
            )

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (640, 640)
        self.input_mean = 127.5
        self.input_std = 128.0
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self._center_cache = {}

        if path_to_detector is None:
            path_to_detector = load_file_from_url(
                model_url, file_name='scrfd_2.5g_bnkps.onnx')

        providers = []
        if 'cuda' in device:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        if 'mps' in device:
            warnings.warn(
                'SCRFD detector does not support MPS. Falling back to CPU.')

        self.session = onnxruntime.InferenceSession(
            path_to_detector, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.use_kps = len(outputs) == 9

    def _preprocess(self, image):
        """Resize with letterbox and normalize."""
        img_h, img_w = image.shape[:2]
        input_w, input_h = self.input_size
        scale = min(input_w / img_w, input_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((input_h, input_w, 3), self.input_mean, dtype=np.float32)
        padded[:new_h, :new_w, :] = resized

        blob = (padded - self.input_mean) / self.input_std
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return blob, scale

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        blob, det_scale = self._preprocess(image)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        fmc = len(self.strides)
        scores_list = []
        bboxes_list = []

        for idx, stride in enumerate(self.strides):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride

            height = self.input_size[1] // stride
            width = self.input_size[0] // stride
            key = (height, width, stride)

            if key not in self._center_cache:
                self._center_cache[key] = generate_anchor_centers(
                    height, width, stride, self.num_anchors)
            anchor_centers = self._center_cache[key]

            pos_inds = np.where(scores.flatten() >= self.confidence_threshold)[0]
            if len(pos_inds) == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds.reshape(-1, 4))
            pos_scores = scores.flatten()[pos_inds]
            pos_bboxes = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

        if len(scores_list) == 0:
            return []

        scores_all = np.concatenate(scores_list)
        bboxes_all = np.concatenate(bboxes_list)

        # Scale back to original image coordinates
        bboxes_all /= det_scale

        dets = np.hstack((bboxes_all, scores_all[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, self.nms_threshold)
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
