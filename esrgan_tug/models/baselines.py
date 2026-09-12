"""Competing generators: Bicubic, SRCNN (9-5-5), VDSR (20 layers), SRResNet (16 blocks, BN)."""
import torch.nn as nn
import torch.nn.functional as F


class Bicubic(nn.Module):
    def __init__(self, scale=4, **_):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


class SRCNN(nn.Module):
    def __init__(self, scale=4, in_ch=3, f1=64, f2=32, **_):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, f1, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(f2, in_ch, 5, padding=2))

    def forward(self, x):
        return self.body(F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False))


class VDSR(nn.Module):
    def __init__(self, scale=4, in_ch=3, nf=64, depth=20, **_):
        super().__init__()
        self.scale = scale
        layers = [nn.Conv2d(in_ch, nf, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(nf, nf, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(nf, in_ch, 3, padding=1)]
        self.body = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return up + self.body(up)


class _ResBlockBN(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1), nn.BatchNorm2d(nf), nn.PReLU(),
                                  nn.Conv2d(nf, nf, 3, 1, 1), nn.BatchNorm2d(nf))

    def forward(self, x):
        return x + self.body(x)


class SRResNet(nn.Module):
    """Ledig et al. (2017) generator; also the SRGAN generator."""

    def __init__(self, scale=4, in_ch=3, out_ch=3, nf=64, n_blocks=16, **_):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, nf, 9, 1, 4), nn.PReLU())
        self.body = nn.Sequential(*[_ResBlockBN(nf) for _ in range(n_blocks)])
        self.body_end = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1), nn.BatchNorm2d(nf))
        up = []
        for _ in range({2: 1, 4: 2, 8: 3}[scale]):
            up += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.PReLU()]
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(nf, out_ch, 9, 1, 4)

    def forward(self, x):
        h = self.head(x)
        return self.tail(self.up(h + self.body_end(self.body(h))))
