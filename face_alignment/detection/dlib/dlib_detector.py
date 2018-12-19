import os
import cv2
import dlib

try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from ..core import FaceDetector
from ...utils import appdata_dir


class DlibDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super().__init__(device, verbose)

        print('Warning: this detector is deprecated. Please use a different one, i.e.: S3FD.')
        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        # Initialise the face detector
        if 'cuda' in device:
            if path_to_detector is None:
                path_to_detector = os.path.join(
                    base_path, "mmod_human_face_detector.dat")

                if not os.path.isfile(path_to_detector):
                    print("Downloading the face detection CNN. Please wait...")

                    path_to_temp_detector = os.path.join(
                        base_path, "mmod_human_face_detector.dat.download")

                    if os.path.isfile(path_to_temp_detector):
                        os.remove(os.path.join(path_to_temp_detector))

                    request_file.urlretrieve(
                        "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat",
                        os.path.join(path_to_temp_detector))

                    os.rename(os.path.join(path_to_temp_detector), os.path.join(path_to_detector))

            self.face_detector = dlib.cnn_face_detection_model_v1(path_to_detector)
        else:
            self.face_detector = dlib.get_frontal_face_detector()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path, rgb=False)

        detected_faces = self.face_detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        if 'cuda' not in self.device:
            detected_faces = [[d.left(), d.top(), d.right(), d.bottom()] for d in detected_faces]
        else:
            detected_faces = [[d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()] for d in detected_faces]

        return detected_faces

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0
