import unittest
from face_alignment import *

class Tester(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self._fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=False)