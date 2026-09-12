"""Forward degradation model, Eq. (1):  I_LR = [ (I_HR * k_sigma) ↓s + n ]_JPEG_q

* k_sigma : anisotropic Gaussian, sigma_x, sigma_y ~ U[0.6, 2.4], random rotation
* ↓s      : decimation by s (centre-aligned sampling, matches bicubic upsampling grid)
* n       : additive Gaussian noise, std ~ U[0, 12]/255
* JPEG_q  : quality ~ U[45, 95]
Every parameter is resampled per patch. Also provides a Real-ESRGAN style
second-order degradation (used only to train the Real-ESRGAN baseline) and
MATLAB-style bicubic downsampling for the natural-image benchmarks.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_CFG = {"sigma": [0.6, 2.4], "noise": [0.0, 12.0], "jpeg": [45, 95], "kernel_size": 21}


def aniso_gaussian_kernel(ksize, sx, sy, theta):
    r = ksize // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    c, s = np.cos(theta), np.sin(theta)
    xr, yr = c * x + s * y, -s * x + c * y
    k = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return (k / k.sum()).astype(np.float32)


def decimate(img, s):
    """Centre-aligned decimation: LR pixel i samples HR position s*i + (s-1)/2."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // s, h // s), interpolation=cv2.INTER_LINEAR)


def jpeg(img_u8, q):
    ok, buf = cv2.imencode(".jpg", img_u8[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def sample_params(rng, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    return {
        "sigma_x": float(rng.uniform(*cfg["sigma"])),
        "sigma_y": float(rng.uniform(*cfg["sigma"])),
        "theta": float(rng.uniform(0, np.pi)),
        "noise": float(rng.uniform(*cfg["noise"])),
        "jpeg_q": int(rng.integers(cfg["jpeg"][0], cfg["jpeg"][1] + 1)),
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
    }


def degrade(hr_u8, scale=4, params=None, rng=None, cfg=None, ksize=None):
    """Apply Eq. (1) to an RGB uint8 image. Returns (lr_u8, params)."""
    if params is None:
        params = sample_params(rng if rng is not None else np.random.default_rng(), cfg)
    ksize = ksize or (cfg or DEFAULT_CFG).get("kernel_size", 21)
    h, w = hr_u8.shape[:2]
    hr = hr_u8[: h - h % scale, : w - w % scale].astype(np.float32) / 255.0
    k = aniso_gaussian_kernel(ksize, params["sigma_x"], params["sigma_y"], params["theta"])
    blurred = cv2.filter2D(hr, -1, k, borderType=cv2.BORDER_REFLECT)
    lr = decimate(blurred, scale)
    nrng = np.random.default_rng(params["noise_seed"])
    lr = lr + nrng.normal(0, params["noise"] / 255.0, lr.shape).astype(np.float32)
    lr = np.clip(np.round(lr * 255), 0, 255).astype(np.uint8)
    return jpeg(lr, params["jpeg_q"]), params


def degrade_high_order(hr_u8, scale=4, rng=None):
    """Simplified Real-ESRGAN second-order degradation (baseline training only)."""
    rng = rng if rng is not None else np.random.default_rng()
    img = hr_u8.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    for stage, out_scale in ((1, rng.uniform(0.3, 1.2)), (2, None)):
        sx, sy = rng.uniform(0.2, 3.0 if stage == 1 else 1.5, 2)
        k = aniso_gaussian_kernel(21, sx, sy, rng.uniform(0, np.pi))
        img = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
        if out_scale is None:
            target = (w // scale, h // scale)
        else:
            target = (max(8, int(img.shape[1] * out_scale)), max(8, int(img.shape[0] * out_scale)))
        interp = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC][rng.integers(3)]
        img = cv2.resize(img, target, interpolation=interp)
        sig = rng.uniform(1, 30 if stage == 1 else 25) / 255.0
        if rng.random() < 0.4:
            img = img + rng.normal(0, sig, img.shape[:2])[..., None].astype(np.float32)
        else:
            img = img + rng.normal(0, sig, img.shape).astype(np.float32)
        u8 = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
        img = jpeg(u8, rng.integers(30, 95)).astype(np.float32) / 255.0
    if img.shape[:2] != (h // scale, w // scale):
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# MATLAB-compatible bicubic imresize (antialiased when downscaling)
# ----------------------------------------------------------------------------
def _cubic(x):
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return ((1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1) +
            (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def _contributions(in_len, out_len, scale, kernel_width=4.0):
    antialias = scale < 1
    if antialias:
        kernel_width = kernel_width / scale
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2
    ind = left[:, None] + np.arange(p)[None]
    dist = u[:, None] - ind
    wts = scale * _cubic(dist * scale) if antialias else _cubic(dist)
    wts = wts / wts.sum(1, keepdims=True)
    # symmetric padding
    aux = np.concatenate([np.arange(in_len), np.arange(in_len - 1, -1, -1)])
    ind = aux[np.mod(ind - 1, aux.size).astype(int)]
    keep = np.any(wts != 0, axis=0)
    return wts[:, keep], ind[:, keep]


def imresize_matlab(img, scale):
    """Bicubic resize equivalent to MATLAB imresize(img, scale, 'bicubic')."""
    arr = img.astype(np.float64)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[..., None]
    h, w = arr.shape[:2]
    oh, ow = int(np.ceil(h * scale)), int(np.ceil(w * scale))
    wy, iy = _contributions(h, oh, scale)
    wx, ix = _contributions(w, ow, scale)
    tmp = (arr[iy] * wy[:, :, None, None]).sum(1)
    out = (tmp[:, ix] * wx[None, :, :, None]).sum(2)
    if img.dtype == np.uint8:
        out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out[..., 0] if squeeze else out
