import sys
import errno
import warnings
import os
import cv2
import dlib

from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir, HASH_REGEX

from ..core import FaceDetector

# Pytorch load supports only pytorch models
def load_file_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file

class DlibDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super().__init__(device, verbose)

        warnings.warn('Warning: this detector is deprecated. Please use a different one, i.e.: S3FD.')

        # Initialise the face detector
        if 'cuda' in device:
            if path_to_detector is None:      
                path_to_detector = load_file_from_url("https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat")

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
