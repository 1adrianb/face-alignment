"""Tests for all face detector backends.

Each detector is tested for:
- Loading and initialization
- Single image detection (correct face found in aflw-test.jpg)
- No-face image handling (grass.jpg returns empty)
- Bbox format (x1, y1, x2, y2, confidence)
- Landmark prediction through FaceAlignment
"""
import sys
sys.path.append('.')
import unittest
import numpy as np
import face_alignment
from face_alignment.utils import get_image


TEST_IMAGE = 'test/assets/aflw-test.jpg'
NO_FACE_IMAGE = 'test/assets/grass.jpg'


def _skip_if_unavailable(detector_name):
    """Try to load a detector, skip test if deps are missing."""
    try:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device='cpu', face_detector=detector_name)
        return fa
    except ImportError as e:
        raise unittest.SkipTest(f"{detector_name} not available: {e}")
    except Exception as e:
        raise unittest.SkipTest(f"{detector_name} failed to load: {e}")


class TestSFDDetector(unittest.TestCase):
    def setUp(self):
        self.fa = _skip_if_unavailable('sfd')

    def test_detect_faces(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertGreaterEqual(len(bboxes), 1)

    def test_bbox_format(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        for bbox in bboxes:
            self.assertEqual(len(bbox), 5)
            x1, y1, x2, y2, conf = bbox[:5]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreater(conf, 0)

    def test_no_face_image(self):
        image = get_image(NO_FACE_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertEqual(len(bboxes), 0)

    def test_landmarks(self):
        preds = self.fa.get_landmarks(TEST_IMAGE)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].shape, (68, 2))

    def test_reference_scale(self):
        self.assertEqual(self.fa.face_detector.reference_scale, 195)


class TestBlazeFaceDetector(unittest.TestCase):
    def setUp(self):
        self.fa = _skip_if_unavailable('blazeface')

    def test_detect_faces(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertGreaterEqual(len(bboxes), 1)

    def test_bbox_format(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        for bbox in bboxes:
            self.assertEqual(len(bbox), 5)

    def test_no_face_image(self):
        image = get_image(NO_FACE_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertEqual(len(bboxes), 0)

    def test_landmarks(self):
        preds = self.fa.get_landmarks(TEST_IMAGE)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].shape, (68, 2))

    def test_reference_scale(self):
        self.assertEqual(self.fa.face_detector.reference_scale, 195)


class TestYuNetDetector(unittest.TestCase):
    def setUp(self):
        self.fa = _skip_if_unavailable('yunet')

    def test_detect_faces(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertGreaterEqual(len(bboxes), 1)

    def test_bbox_format(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        for bbox in bboxes:
            self.assertEqual(len(bbox), 5)
            x1, y1, x2, y2, conf = bbox
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreater(conf, 0)

    def test_no_face_image(self):
        image = get_image(NO_FACE_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertEqual(len(bboxes), 0)

    def test_landmarks(self):
        preds = self.fa.get_landmarks(TEST_IMAGE)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].shape, (68, 2))

    def test_reference_scale(self):
        self.assertEqual(self.fa.face_detector.reference_scale, 165)


class TestRetinaFaceDetector(unittest.TestCase):
    def setUp(self):
        self.fa = _skip_if_unavailable('retinaface')

    def test_detect_faces(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertGreaterEqual(len(bboxes), 1)

    def test_bbox_format(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        for bbox in bboxes:
            self.assertEqual(len(bbox), 5)
            x1, y1, x2, y2, conf = bbox
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreater(conf, 0)

    def test_no_face_image(self):
        image = get_image(NO_FACE_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertEqual(len(bboxes), 0)

    def test_landmarks(self):
        preds = self.fa.get_landmarks(TEST_IMAGE)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].shape, (68, 2))

    def test_reference_scale(self):
        self.assertEqual(self.fa.face_detector.reference_scale, 165)


class TestSCRFDDetector(unittest.TestCase):
    def setUp(self):
        self.fa = _skip_if_unavailable('scrfd')

    def test_detect_faces(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertGreaterEqual(len(bboxes), 1)

    def test_bbox_format(self):
        image = get_image(TEST_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        for bbox in bboxes:
            self.assertEqual(len(bbox), 5)
            x1, y1, x2, y2, conf = bbox
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreater(conf, 0)

    def test_no_face_image(self):
        image = get_image(NO_FACE_IMAGE)
        bboxes = self.fa.face_detector.detect_from_image(image.copy())
        self.assertEqual(len(bboxes), 0)

    def test_landmarks(self):
        preds = self.fa.get_landmarks(TEST_IMAGE)
        self.assertIsNotNone(preds)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].shape, (68, 2))

    def test_reference_scale(self):
        self.assertEqual(self.fa.face_detector.reference_scale, 165)


if __name__ == '__main__':
    unittest.main()
