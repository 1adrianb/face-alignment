import unittest
import numpy as np
import face_alignment
import sys
import torch
sys.path.append('.')
from face_alignment.utils import get_image


class Tester(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_data = [np.array([[140., 240., -85.66882],
                                         [143., 267., -80.8413],
                                         [146., 288., -75.863716],
                                         [149., 309., -68.59805],
                                         [155., 330., -53.45323],
                                         [164., 342., -30.021921],
                                         [173., 348., -3.172535],
                                         [188., 354., 22.986391],
                                         [212., 357., 38.229393],
                                         [242., 357., 31.675377],
                                         [266., 354., 12.673657],
                                         [284., 351., -9.144677],
                                         [302., 336., -28.259748],
                                         [314., 318., -40.44641],
                                         [323., 300., -45.717842],
                                         [329., 279., -49.108025],
                                         [335., 252., -52.744144],
                                         [152., 207., -8.623771],
                                         [164., 201., 4.978135],
                                         [176., 198., 15.687496],
                                         [188., 198., 23.345095],
                                         [200., 201., 27.896727],
                                         [245., 204., 36.703583],
                                         [257., 201., 36.338104],
                                         [269., 201., 33.219273],
                                         [284., 204., 27.73946],
                                         [299., 216., 17.927979],
                                         [221., 225., 36.493015],
                                         [218., 237., 46.664886],
                                         [218., 249., 58.6025],
                                         [215., 261., 61.38716],
                                         [203., 273., 38.602806],
                                         [209., 276., 43.43367],
                                         [218., 276., 47.01009],
                                         [227., 276., 46.390728],
                                         [233., 276., 43.90469],
                                         [170., 228., 6.148446],
                                         [179., 222., 16.064037],
                                         [188., 222., 18.701725],
                                         [200., 228., 18.116598],
                                         [191., 231., 19.561537],
                                         [179., 231., 15.033958],
                                         [248., 231., 27.855675],
                                         [257., 225., 32.28227],
                                         [269., 225., 33.661648],
                                         [278., 231., 26.486513],
                                         [269., 234., 32.230045],
                                         [257., 234., 32.62314],
                                         [185., 306., 27.934061],
                                         [194., 297., 40.528717],
                                         [209., 291., 48.805637],
                                         [215., 291., 51.26784],
                                         [224., 291., 51.571762],
                                         [239., 300., 47.687435],
                                         [248., 309., 38.29609],
                                         [236., 312., 47.75453],
                                         [224., 315., 51.587326],
                                         [212., 315., 50.961407],
                                         [203., 312., 47.905617],
                                         [194., 309., 40.80765],
                                         [188., 303., 28.68554],
                                         [206., 300., 44.749367],
                                         [215., 300., 48.15791],
                                         [224., 300., 48.014362],
                                         [248., 306., 38.097015],
                                         [224., 303., 48.84072],
                                         [215., 303., 48.249004],
                                         [206., 303., 45.470398]], dtype=np.float32)]

    def test_predict_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu')
        preds = fa.get_landmarks('test/assets/aflw-test.jpg')
        self.assertEqual(len(preds), len(self.reference_data))
        for pred, reference in zip(preds, self.reference_data):
            self.assertTrue(np.allclose(pred, reference, atol=0.07))

    def test_predict_batch_points(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu')

        reference_data = self.reference_data + self.reference_data
        reference_data.append([])
        image = get_image('test/assets/aflw-test.jpg')
        batch = np.stack([image, image, np.zeros_like(image)])
        batch = torch.Tensor(batch.transpose(0, 3, 1, 2))

        preds = fa.get_landmarks_from_batch(batch)

        self.assertEqual(len(preds), len(reference_data))
        for pred, reference in zip(preds, reference_data):
            self.assertTrue(np.allclose(pred, reference, atol=0.07))

    def test_predict_points_from_dir(self):
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cpu')

        reference_data = {
            'test/assets/grass.jpg': None,
            'test/assets/aflw-test.jpg': self.reference_data}

        preds = fa.get_landmarks_from_directory('test/assets/',
                                                extensions=['.jpg', '.png'])

        for k, points in preds.items():
            if k not in reference_data:
                continue
            if isinstance(points, list):
                for p, p_reference in zip(points, reference_data[k]):
                    self.assertTrue(np.allclose(p, p_reference, atol=0.07))
            else:
                self.assertEqual(points, reference_data[k])


if __name__ == '__main__':
    unittest.main()
