import unittest
import face_alignment


class Tester(unittest.TestCase):
    def test_predict_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu')
        fa.get_landmarks('test/assets/aflw-test.jpg')

if __name__ == '__main__':
    unittest.main()
