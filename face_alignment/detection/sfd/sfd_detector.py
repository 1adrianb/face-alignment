import os
import cv2

try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from ..core import FaceDetector
from ...utils import appdata_dir

from .net_s3fd import s3fd
from .bbox import *
from .detect import *


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        # Initialise the face detector
        if path_to_detector is None:
            path_to_detector = os.path.join(
                base_path, "s3fd_convert.pth")

            if not os.path.isfile(path_to_detector):
                print("Downloading the face detection CNN. Please wait...")

                path_to_temp_detector = os.path.join(
                    base_path, "s3fd_convert.pth.download")

                if os.path.isfile(path_to_temp_detector):
                    os.remove(os.path.join(path_to_temp_detector))

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/python-fan/s3fd_convert.pth",
                    os.path.join(path_to_temp_detector))

                os.rename(os.path.join(path_to_temp_detector), os.path.join(path_to_detector))

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(torch.load(path_to_detector))
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)
        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep, :]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
