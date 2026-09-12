"""Reconstruction metrics, Eqs. (14)-(16).

PSNR / MSE / SSIM follow the SR-benchmark convention: luminance (Y of YCbCr,
ITU-R BT.601, range 16-235) with `crop` border pixels removed; SSIM uses an
11x11 Gaussian window, sigma 1.5 (Wang et al. 2004). LPIPS (AlexNet) needs
PyTorch and the `lpips` package, see `lpips_batch`.
"""
from __future__ import annotations

import cv2
import numpy as np


def rgb_to_y(img_u8):
    x = img_u8.astype(np.float64) / 255.0
    return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]


def _prep(sr, hr, crop, y_only):
    if sr.shape != hr.shape:
        raise ValueError(f"shape mismatch {sr.shape} vs {hr.shape}")
    a = rgb_to_y(sr) if (y_only and sr.ndim == 3) else sr.astype(np.float64)
    b = rgb_to_y(hr) if (y_only and hr.ndim == 3) else hr.astype(np.float64)
    if crop:
        a, b = a[crop:-crop, crop:-crop], b[crop:-crop, crop:-crop]
    return a, b


def mse(sr, hr, crop=4, y_only=True):
    a, b = _prep(sr, hr, crop, y_only)
    return float(np.mean((a - b) ** 2))


def psnr(sr, hr, crop=4, y_only=True, peak=255.0):
    m = mse(sr, hr, crop, y_only)
    return float("inf") if m == 0 else float(10 * np.log10(peak ** 2 / m))


def _ssim_single(a, b):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)  # separable 11x11 Gaussian window (identical to filter2D with outer(k, k))
    f = lambda x: cv2.sepFilter2D(x, -1, k, k)[5:-5, 5:-5]  # noqa: E731
    mu1, mu2 = f(a), f(b)
    s1 = f(a ** 2) - mu1 ** 2
    s2 = f(b ** 2) - mu2 ** 2
    s12 = f(a * b) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return float(m.mean())


def ssim(sr, hr, crop=4, y_only=True):
    a, b = _prep(sr, hr, crop, y_only)
    if a.ndim == 2:
        return _ssim_single(a, b)
    return float(np.mean([_ssim_single(a[..., c], b[..., c]) for c in range(a.shape[2])]))


def lpips_batch(pairs, net="alex", device=None):
    """LPIPS for a list of (sr_u8, hr_u8) pairs. Requires torch + lpips."""
    import lpips
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    out = []
    with torch.no_grad():
        for sr, hr in pairs:
            t = [torch.from_numpy(x.transpose(2, 0, 1)).float().div(127.5).sub(1).unsqueeze(0).to(device) for x in (sr, hr)]
            out.append(float(fn(t[0], t[1]).item()))
    return out
