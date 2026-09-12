"""RRDB generator with the dual channel-spatial attention gate (RRDB-DA), Eqs. (3)-(5).

`RRDBNet(n_blocks=8, attention=True, upsampler="pixelshuffle")` is the proposed
ESR-GAN generator of Fig. 2(a). `RRDBNet(n_blocks=23, attention=False,
upsampler="nearest")` reproduces the original ESRGAN / Real-ESRGAN generator
(16.70 M parameters).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_init_weights(module, scale=1.0):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            m.weight.data.mul_(scale)
            if m.bias is not None:
                m.bias.data.zero_()


class ChannelAttention(nn.Module):
    """Eq. (3): M_c(F) = sigma(W2 delta(W1 GAP(F))),  F' = M_c(F) * F."""

    def __init__(self, nf=64, reduction=16):
        super().__init__()
        mid = max(1, nf // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(nf, mid, 1), nn.ReLU(inplace=True), nn.Conv2d(mid, nf, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Eq. (4): M_s(F') = sigma(f7x7([AvgPool_c(F'); MaxPool_c(F')])),  F'' = M_s(F') * F'."""

    def __init__(self, kernel=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(self, x):
        desc = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(desc))


class DenseBlock(nn.Module):
    """Residual dense block of ESRGAN (5 convs, growth gc, local residual scaling)."""

    def __init__(self, nf=64, gc=32, res_scale=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.res_scale = res_scale
        default_init_weights(self, 0.1)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x + self.res_scale * x5


class RRDB(nn.Module):
    """Eq. (5): F_out = F_in + beta * A(R3(R2(R1(F_in)))), A = M_s o M_c (identity if attention=False)."""

    def __init__(self, nf=64, gc=32, res_scale=0.2, attention=False, reduction=16, sa_kernel=7):
        super().__init__()
        self.rdb1 = DenseBlock(nf, gc, res_scale)
        self.rdb2 = DenseBlock(nf, gc, res_scale)
        self.rdb3 = DenseBlock(nf, gc, res_scale)
        self.att = nn.Sequential(ChannelAttention(nf, reduction), SpatialAttention(sa_kernel)) if attention else nn.Identity()
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.att(self.rdb3(self.rdb2(self.rdb1(x))))


class RRDBNet(nn.Module):
    def __init__(self, scale=4, in_ch=3, out_ch=3, nf=64, gc=32, n_blocks=8, res_scale=0.2,
                 attention=True, reduction=16, sa_kernel=7, upsampler="pixelshuffle"):
        super().__init__()
        if scale not in (2, 4, 8):
            raise ValueError("scale must be 2, 4 or 8")
        self.scale, self.upsampler = scale, upsampler
        n_up = int(math.log2(scale))
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc, res_scale, attention, reduction, sa_kernel) for _ in range(n_blocks)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if upsampler == "pixelshuffle":      # Fig. 2(a): two sub-pixel x2 stages, then Conv 3x3 n3
            layers = []
            for _ in range(n_up):
                layers += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, inplace=True)]
            self.upsample = nn.Sequential(*layers)
            self.conv_hr = nn.Identity()
        elif upsampler == "nearest":         # original ESRGAN: nearest + conv, then HR conv
            self.upconvs = nn.ModuleList([nn.Conv2d(nf, nf, 3, 1, 1) for _ in range(n_up)])
            self.conv_hr = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True))
        else:
            raise ValueError(upsampler)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))       # global residual skip
        if self.upsampler == "pixelshuffle":
            feat = self.upsample(feat)
        else:
            for conv in self.upconvs:
                feat = self.act(conv(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.conv_hr(feat))
