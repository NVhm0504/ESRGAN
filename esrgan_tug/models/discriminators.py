"""Discriminators: VGG-style strided trunk of Fig. 2(b) and Real-ESRGAN's U-Net with spectral norm."""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class VGGDiscriminator(nn.Module):
    """k3n64s1 -> k3n64s2 -> k3n128s1 -> k3n128s2 -> k3n256s1 -> k3n256s2 -> k3n512s1 -> k3n512s2
    -> Dense 1024 -> LReLU -> Dense 1. Returns the non-transformed logit C(x) used by Eq. (6).

    pool_size > 0 inserts adaptive average pooling before the dense head so the
    head size does not depend on the patch size (6 reproduces SRGAN's 96-px head);
    pool_size = 0 flattens the full 512 x (input/16)^2 map.
    """

    def __init__(self, in_ch=3, nf=64, input_size=192, pool_size=6, fc=1024, **_):
        super().__init__()
        spec = [(nf, 1, False), (nf, 2, True), (2 * nf, 1, True), (2 * nf, 2, True),
                (4 * nf, 1, True), (4 * nf, 2, True), (8 * nf, 1, True), (8 * nf, 2, True)]
        layers, c = [], in_ch
        for out, stride, bn in spec:
            layers.append(nn.Conv2d(c, out, 3, stride, 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            c = out
        self.features = nn.Sequential(*layers)
        side = pool_size if pool_size else input_size // 16
        self.pool = nn.AdaptiveAvgPool2d(pool_size) if pool_size else nn.Identity()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * side * side, fc), nn.LeakyReLU(0.2, inplace=True), nn.Linear(fc, 1))

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class UNetDiscriminatorSN(nn.Module):
    """Real-ESRGAN U-Net discriminator with spectral normalisation (per-pixel logits)."""

    def __init__(self, in_ch=3, nf=64, skip=True, **_):
        super().__init__()
        self.skip = skip
        sn = spectral_norm
        self.conv0 = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False))
        self.conv7 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(self, x):
        lr = lambda t: F.leaky_relu(t, 0.2)  # noqa: E731
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        x0 = lr(self.conv0(x))
        x1 = lr(self.conv1(x0))
        x2 = lr(self.conv2(x1))
        x3 = lr(self.conv3(x2))
        x4 = lr(self.conv4(up(x3)))
        if self.skip:
            x4 = x4 + x2
        x5 = lr(self.conv5(up(x4)))
        if self.skip:
            x5 = x5 + x1
        x6 = lr(self.conv6(up(x5)))
        if self.skip:
            x6 = x6 + x0
        return self.conv9(lr(self.conv8(lr(self.conv7(x6)))))
