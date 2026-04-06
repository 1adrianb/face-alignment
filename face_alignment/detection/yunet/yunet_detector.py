import warnings
import cv2

from ..core import FaceDetector
from ...utils import load_file_from_url

model_url = (
    'https://github.com/opencv/opencv_zoo/raw/main/models/'
    'face_detection_yunet/face_detection_yunet_2023mar.onnx'
)


class YuNetDetector(FaceDetector):
    """YuNet face detector using OpenCV's FaceDetectorYN.

    Requires opencv-python >= 4.5.4. Runs on CPU via OpenCV's DNN backend.
    """

    def __init__(self, device, path_to_detector=None, verbose=False,
                 score_threshold=0.5, nms_threshold=0.3):
        super(YuNetDetector, self).__init__(device, verbose)

        if 'cpu' not in device:
            warnings.warn(
                'YuNet detector runs on CPU via OpenCV DNN. '
                'The device argument will be ignored.'
            )

        if path_to_detector is None:
            path_to_detector = load_file_from_url(
                model_url, file_name='face_detection_yunet_2023mar.onnx')

        self.face_detector = cv2.FaceDetectorYN.create(
            path_to_detector, '', (0, 0), score_threshold, nms_threshold)

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]
        self.face_detector.setInputSize((w, h))

        _, faces = self.face_detector.detect(image)

        if faces is None:
            return []

        bboxlist = []
        for face in faces:
            x, y, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
            confidence = float(face[14])
            bboxlist.append([x, y, x + fw, y + fh, confidence])

        return bboxlist

    @property
    def reference_scale(self):
        return 165

    def detect_from_batch(self, tensor):
        bboxlists = []
        for i in range(tensor.shape[0]):
            image = tensor[i].cpu().numpy().transpose(1, 2, 0)
            bboxlists.append(self.detect_from_image(image))
        return bboxlists
