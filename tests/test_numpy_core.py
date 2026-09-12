"""Framework-free unit tests (no PyTorch needed):  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from esrgan_tug.data.degradation import degrade, imresize_matlab  # noqa: E402
from esrgan_tug.data.simulator import generate_patch, make_region_style  # noqa: E402
from esrgan_tug.metrics.classification import metrics_from_cm  # noqa: E402
from esrgan_tug.metrics.image_quality import psnr, ssim  # noqa: E402
from esrgan_tug.metrics.stats import mcnemar, paired_t  # noqa: E402
from esrgan_tug.metrics.tug import tug, tug_with_ci  # noqa: E402

# Table VI of the paper (proposed ESR-GAN output) and Fig. 8(a) (LR input)
CM_SR = [[482, 4, 3, 6, 5], [3, 464, 5, 6, 22], [2, 4, 471, 18, 5], [5, 7, 20, 456, 12], [3, 31, 6, 16, 444]]
CM_LR = [[431, 12, 15, 24, 18], [9, 362, 22, 31, 76], [7, 21, 388, 63, 21], [14, 29, 68, 349, 40], [11, 92, 27, 46, 324]]


class TestClassificationMetrics(unittest.TestCase):
    def test_table_vii(self):
        m = metrics_from_cm(CM_SR)
        self.assertAlmostEqual(m["oa"], 0.9268, places=4)
        self.assertAlmostEqual(m["kappa"], 0.9085, places=4)
        np.testing.assert_allclose(m["precision"], [0.9737, 0.9098, 0.9327, 0.9084, 0.9098], atol=1e-4)
        np.testing.assert_allclose(m["iou"], [0.9396, 0.8498, 0.8820, 0.8352, 0.8162], atol=1e-4)
        self.assertAlmostEqual(m["miou"], 0.8646, places=4)
        self.assertAlmostEqual(m["macro_f1"], 0.9268, places=4)

    def test_fig8a(self):
        m = metrics_from_cm(CM_LR)
        self.assertAlmostEqual(m["oa"], 0.7416, places=4)
        self.assertAlmostEqual(m["kappa"], 0.6770, places=4)

    def test_tug_paper_numbers(self):
        # paper: "recovers 88 percent of the accuracy that the x4 degradation destroys"
        self.assertAlmostEqual(1 - tug(92.68, 74.16, 95.24), 0.8786, places=4)


class TestImageMetrics(unittest.TestCase):
    def test_psnr_ssim_identity_and_noise(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        self.assertEqual(psnr(a, a), float("inf"))
        self.assertAlmostEqual(ssim(a, a), 1.0, places=6)
        b = np.clip(a.astype(int) + rng.normal(0, 5, a.shape), 0, 255).astype(np.uint8)
        self.assertTrue(25 < psnr(b, a) < 45)

    def test_imresize_roundtrip(self):
        x = np.tile(np.linspace(0, 255, 64), (64, 1)).astype(np.uint8)[..., None].repeat(3, 2)
        up = imresize_matlab(imresize_matlab(x, 0.25), 4)
        self.assertEqual(up.shape, x.shape)
        self.assertGreater(psnr(up, x, crop=8), 35)


class TestDataAndStats(unittest.TestCase):
    def test_simulator_and_degradation(self):
        st = make_region_style(3, 0)
        for c in range(5):
            img, lab, info = generate_patch(c, st, np.random.SeedSequence([0, c]), 96)
            self.assertEqual(img.shape, (96, 96, 3))
            self.assertEqual(np.bincount(lab.ravel(), minlength=5).argmax(), c)
            lr, p = degrade(img, 4, rng=np.random.default_rng(c))
            self.assertEqual(lr.shape, (24, 24, 3))
            lr2, _ = degrade(img, 4, params=p)
            np.testing.assert_array_equal(lr, lr2)  # stored params reproduce the LR exactly

    def test_determinism(self):
        st = make_region_style(1, 0)
        a = generate_patch(2, st, np.random.SeedSequence([5]), 64)[0]
        b = generate_patch(2, st, np.random.SeedSequence([5]), 64)[0]
        np.testing.assert_array_equal(a, b)

    def test_stats(self):
        rng = np.random.default_rng(1)
        x = rng.normal(30, 1, 500)
        r = paired_t(x + 0.2 + rng.normal(0, 0.1, 500), x)
        self.assertLess(r["p"], 1e-10)
        y = rng.integers(0, 5, 1000)
        pa = np.where(rng.random(1000) < 0.9, y, (y + 1) % 5)
        pb = np.where(rng.random(1000) < 0.8, y, (y + 1) % 5)
        self.assertLess(mcnemar(y, pa, pb)["p"], 1e-3)
        ci = tug_with_ci(y, pa, pb, y, n_boot=200)
        self.assertTrue(ci["ci_low"] <= ci["tug"] <= ci["ci_high"])


if __name__ == "__main__":
    unittest.main()
