import unittest
from face_alignment.utils import *
import numpy as np
import torch


class Tester(unittest.TestCase):
    def test_flip_is_label(self):
        # Generate the points
        heatmaps = torch.from_numpy(np.random.randint(1, high=250, size=(68, 64, 64)).astype('float32'))

        flipped_heatmaps = flip(flip(heatmaps.clone(), is_label=True), is_label=True)

        assert np.allclose(heatmaps.numpy(), flipped_heatmaps.numpy())

    def test_flip_is_image(self):
        fake_image = torch.torch.rand(3, 256, 256)
        fliped_fake_image = flip(flip(fake_image.clone()))

        assert np.allclose(fake_image.numpy(), fliped_fake_image.numpy())

    def test_getpreds(self):
        pts = torch.from_numpy(np.random.randint(1, high=63, size=(68, 2)).astype('float32'))

        heatmaps = np.zeros((68, 256, 256))
        for i in range(68):
            if pts[i, 0] > 0:
                heatmaps[i] = draw_gaussian(heatmaps[i], pts[i], 2)
        heatmaps = torch.from_numpy(np.expand_dims(heatmaps, axis=0))

        preds, _ = get_preds_fromhm(heatmaps)

        assert np.allclose(pts.numpy(), preds.numpy(), atol=5)

if __name__ == '__main__':
    unittest.main()
