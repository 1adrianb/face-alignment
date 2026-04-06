import os
import logging
import torch
import warnings
from enum import IntEnum
import numpy as np
from tqdm import tqdm

from .utils import *
from .models import FAN, ResNetDepth
from .folder_data import FolderData

logger = logging.getLogger(__name__)

COMPILE_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'face_alignment', 'compile')


def _compile_cache_path(landmarks_type, network_size, device, dtype):
    """Return the path to the cached compile artifacts."""
    device_str = device if isinstance(device, str) else str(device)
    device_str = device_str.split(':')[0]
    dtype_str = str(dtype).replace('torch.', '')
    lm_str = '3d' if landmarks_type == 3 else '2d'
    key = f'{lm_str}_fan{network_size}_{device_str}_{dtype_str}'
    return os.path.join(COMPILE_CACHE_DIR, f'{key}.bin')


def _load_compile_cache(cache_path):
    """Load cached compile artifacts if available."""
    if not os.path.exists(cache_path):
        return False
    try:
        with open(cache_path, 'rb') as f:
            artifact_bytes = f.read()
        torch.compiler.load_cache_artifacts(artifact_bytes)
        logger.info('Loaded compile cache from %s', cache_path)
        return True
    except Exception as e:
        logger.warning('Failed to load compile cache (%s), recompiling', e)
        return False


def _save_compile_cache(cache_path):
    """Save compile artifacts to disk."""
    try:
        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is not None:
            artifact_bytes, _ = artifacts
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(artifact_bytes)
            logger.info('Saved compile cache to %s', cache_path)
    except Exception as e:
        logger.warning('Failed to save compile cache: %s', e)


class LandmarksType(IntEnum):
    """Enum class defining the type of landmarks to detect.

    ``TWO_D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``TWO_HALF_D`` - this points represent the projection of the 3D points into 3D
    ``THREE_D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    TWO_D = 1
    TWO_HALF_D = 2
    THREE_D = 3


class NetworkSize(IntEnum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4


models_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', dtype=torch.float32, flip_input=True, face_detector='sfd',
                 face_detector_kwargs=None, verbose=False, compile=True, max_batch_size=1):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose
        self.dtype = dtype
        self.max_batch_size = max_batch_size

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        face_detector_kwargs = face_detector_kwargs or {}
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose, **face_detector_kwargs)

        # Initialise the face alignment networks
        if landmarks_type == LandmarksType.TWO_D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)

        self.face_alignment_net = self._load_native(network_name, network_size, device, dtype)

        # Initialise the depth prediction network
        if landmarks_type == LandmarksType.THREE_D:
            self.depth_prediciton_net = self._load_native_depth(device, dtype)

        # Apply torch.compile for faster inference
        if compile:
            try:
                cache_path = _compile_cache_path(landmarks_type, network_size, device, dtype)
                cache_hit = _load_compile_cache(cache_path)

                self.face_alignment_net = torch.compile(self.face_alignment_net)
                if landmarks_type == LandmarksType.THREE_D:
                    self.depth_prediciton_net = torch.compile(self.depth_prediciton_net)

                if not cache_hit:
                    # Warm up to trigger compilation, then save cache
                    warnings.warn(
                        'Compiling face alignment model (one-time cost). '
                        'Subsequent runs will be faster.')
                    sample = torch.randn(1, 3, CROP_RESOLUTION, CROP_RESOLUTION,
                                         device=device, dtype=dtype)
                    self.face_alignment_net(sample)
                    if landmarks_type == LandmarksType.THREE_D:
                        sample_depth = torch.randn(1, 3 + NUM_LANDMARKS,
                                                   CROP_RESOLUTION, CROP_RESOLUTION,
                                                   device=device, dtype=dtype)
                        self.depth_prediciton_net(sample_depth)
                    _save_compile_cache(cache_path)
            except Exception as e:
                logger.warning('torch.compile failed (%s), using eager mode', e)

    def _load_native(self, network_name, network_size, device, dtype):
        """Load FAN model with native PyTorch (no compilation)."""
        net = FAN(network_size)
        fan_weights = torch.load(
            load_file_from_url(models_urls[network_name]),
            map_location=device, weights_only=True)
        net.load_state_dict(fan_weights)
        net.to(device, dtype=dtype)
        net.eval()
        return net

    def _load_native_depth(self, device, dtype):
        """Load depth model with native PyTorch (no compilation)."""
        net = ResNetDepth()
        depth_weights = torch.load(
            load_file_from_url(models_urls['depth']),
            map_location=device, weights_only=False)
        depth_dict = {k.replace('module.', ''): v
                      for k, v in depth_weights['state_dict'].items()}
        net.load_state_dict(depth_dict)
        net.to(device, dtype=dtype)
        net.eval()
        return net

    def get_landmarks(self, image_or_path, detected_faces=None, return_bboxes=False, return_landmark_score=False):
        """Deprecated, please use get_landmarks_from_image

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
        """
        return self.get_landmarks_from_image(image_or_path, detected_faces, return_bboxes, return_landmark_score)

    @torch.no_grad()
    def get_landmarks_from_image(self, image_or_path, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmark
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """
        image = get_image(image_or_path)

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(image.copy())

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        # Precompute centers, scales, and crops for all faces
        centers = []
        scales = []
        crops = []
        for d in detected_faces:
            center = np.array(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * CENTER_Y_OFFSET
            scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale
            centers.append(center)
            scales.append(scale)
            crops.append(crop(image, center, scale).transpose((2, 0, 1)).astype(np.float32) / 255.0)

        # Batch FAN forward pass (chunked by max_batch_size)
        all_crops = torch.from_numpy(np.stack(crops))
        out_chunks = []
        for start in range(0, len(crops), self.max_batch_size):
            inp_chunk = all_crops[start:start + self.max_batch_size].to(self.device, dtype=self.dtype)
            out_chunk = self.face_alignment_net(inp_chunk)
            if isinstance(out_chunk, list):
                out_chunk = out_chunk[-1]
            out_chunk = out_chunk.detach()
            if self.flip_input:
                flip_out = self.face_alignment_net(flip(inp_chunk))
                if isinstance(flip_out, list):
                    flip_out = flip_out[-1]
                out_chunk += flip(flip_out.detach(), is_label=True)
            out_chunks.append(out_chunk)

        out_batch = torch.cat(out_chunks, dim=0).to(device='cpu', dtype=torch.float32).numpy()
        inp_batch = all_crops.to(self.device, dtype=self.dtype)

        # Per-face postprocessing (each face has its own center/scale)
        landmarks = []
        landmarks_scores = []
        for i in range(len(detected_faces)):
            center, scale = centers[i], scales[i]
            out = out_batch[i:i+1]

            pts, pts_img, scores = get_preds_fromhm(out, center, scale)
            pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
            pts, pts_img = pts.view(NUM_LANDMARKS, 2) * 4, pts_img.view(NUM_LANDMARKS, 2)
            scores = scores.squeeze(0)

            if self.landmarks_type == LandmarksType.THREE_D:
                heatmaps = np.zeros((NUM_LANDMARKS, CROP_RESOLUTION, CROP_RESOLUTION), dtype=np.float32)
                for j in range(NUM_LANDMARKS):
                    if pts[j, 0] > 0 and pts[j, 1] > 0:
                        heatmaps[j] = draw_gaussian(
                            heatmaps[j], pts[j], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.device, dtype=self.dtype)
                depth_inp = torch.cat((inp_batch[i:i+1], heatmaps), 1)
                depth_pred = self.depth_prediciton_net(
                    depth_inp).data.cpu().view(NUM_LANDMARKS, 1).to(dtype=torch.float32)
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (CROP_RESOLUTION / (SCALE_FACTOR * scale)))), 1)

            landmarks.append(pts_img.numpy())
            landmarks_scores.append(scores)

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores, detected_faces
        else:
            return landmarks

    @torch.no_grad()
    def get_landmarks_from_batch(self, image_batch, detected_faces=None, return_bboxes=False,
                                 return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_batch {torch.tensor} -- The input images batch

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmarks
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_batch(image_batch)

        if len(detected_faces) == 0:
            warnings.warn("No faces were detected.")
            if return_bboxes or return_landmark_score:
                return None, None, None
            else:
                return None

        landmarks = []
        landmarks_scores_list = []
        # A batch for each frame
        for i, faces in enumerate(detected_faces):
            res = self.get_landmarks_from_image(
                image_batch[i].cpu().numpy().transpose(1, 2, 0),
                detected_faces=faces,
                return_landmark_score=return_landmark_score,
            )
            if return_landmark_score:
                landmark_set, landmarks_scores, _ = res
                landmarks_scores_list.append(landmarks_scores)
            else:
                landmark_set = res
            # Bacward compatibility
            if landmark_set is not None:
                landmark_set = np.concatenate(landmark_set, axis=0)
            else:
                landmark_set = []
            landmarks.append(landmark_set)

        if not return_bboxes:
            detected_faces = None
        if not return_landmark_score:
            landmarks_scores_list = None
        if return_bboxes or return_landmark_score:
            return landmarks, landmarks_scores_list, detected_faces
        else:
            return landmarks

    def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True,
                                     return_bboxes=False, return_landmark_score=False):
        """Scan a directory for images with a given extension type(s) and predict the landmarks for each
            face present in the images found.

         Arguments:
            path {str} -- path to the target directory containing the images

        Keyword Arguments:
            extensions {list of str} -- list containing the image extensions considered (default: ['.jpg', '.png'])
            recursive {boolean} -- If True, scans for images recursively (default: True)
            show_progress_bar {boolean} -- If True displays a progress bar (default: True)
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
        """
        dataset = FolderData(path, self.face_detector.tensor_or_path_to_ndarray, extensions, recursive, self.verbose)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=4)
      
        predictions = {}
        for (image_path, image) in tqdm(dataloader, disable=not show_progress_bar):
            image_path, image = image_path[0], image[0]
            bounding_boxes = self.face_detector.detect_from_image(image)
            if return_bboxes or return_landmark_score:
                preds, bbox, score = self.get_landmarks_from_image(
                    image, bounding_boxes, return_bboxes=return_bboxes, return_landmark_score=return_landmark_score)
                predictions[image_path] = (preds, bbox, score)
            else:
                preds = self.get_landmarks_from_image(image, bounding_boxes)
                predictions[image_path] = preds

        return predictions
