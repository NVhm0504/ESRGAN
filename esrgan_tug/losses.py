"""Composite generator objective, Eqs. (6)-(13)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

VGG19_LAYERS = {"conv1_2": 2, "conv2_2": 7, "conv3_4": 16, "conv4_4": 25, "conv5_4": 34}


def luminance(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


class VGGFeatures(nn.Module):
    """Frozen VGG-19 (ImageNet) returning pre-activation features of the requested layers."""

    def __init__(self, layers=("conv5_4",), pretrained=True):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1" if pretrained else None).features
        last = max(VGG19_LAYERS[k] for k in layers)
        self.body = vgg[: last + 1]
        for m in self.body:
            if isinstance(m, nn.ReLU):
                m.inplace = False  # keep stored pre-activation maps intact
        self.idx = {VGG19_LAYERS[k]: k for k in layers}
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):  # always frozen / eval
        return super().train(False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = {}
        for i, m in enumerate(self.body):
            x = m(x)
            if i in self.idx:
                out[self.idx[i]] = x
        return out


class PerceptualLoss(nn.Module):
    """Eq. (10): mean squared distance of VGG-19 conv5_4 (pre-activation) features."""

    def __init__(self, layer_weights=None, criterion="mse", pretrained=True):
        super().__init__()
        self.layer_weights = layer_weights or {"conv5_4": 1.0}
        self.vgg = VGGFeatures(tuple(self.layer_weights), pretrained)
        self.crit = F.mse_loss if criterion == "mse" else F.l1_loss

    def forward(self, sr, hr):
        fs = self.vgg(sr)
        with torch.no_grad():
            fh = self.vgg(hr)
        return sum(w * self.crit(fs[k], fh[k]) for k, w in self.layer_weights.items())


def ragan_d_loss(real_logit, fake_logit):
    """Eq. (7)."""
    return (F.binary_cross_entropy_with_logits(real_logit - fake_logit.mean(), torch.ones_like(real_logit)) +
            F.binary_cross_entropy_with_logits(fake_logit - real_logit.mean(), torch.zeros_like(fake_logit)))


def ragan_g_loss(real_logit, fake_logit):
    """Eq. (8)."""
    return (F.binary_cross_entropy_with_logits(real_logit - fake_logit.mean(), torch.zeros_like(real_logit)) +
            F.binary_cross_entropy_with_logits(fake_logit - real_logit.mean(), torch.ones_like(fake_logit)))


def vanilla_d_loss(real_logit, fake_logit):
    return (F.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit)) +
            F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit)))


def vanilla_g_loss(real_logit, fake_logit):
    return F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))


D_LOSSES = {"ragan": ragan_d_loss, "vanilla": vanilla_d_loss}
G_LOSSES = {"ragan": ragan_g_loss, "vanilla": vanilla_g_loss}


class FrequencyLoss(nn.Module):
    """Eq. (11): L1 between Fourier magnitude spectra of the luminance channel."""

    def forward(self, sr, hr):
        fs = torch.fft.fft2(luminance(sr.float()), norm="ortho").abs()
        fh = torch.fft.fft2(luminance(hr.float()), norm="ortho").abs()
        return F.l1_loss(fs, fh)


class EdgeLoss(nn.Module):
    """Eq. (12): L1 between Sobel gradients of the luminance + lambda_tv * TV(I_SR)."""

    def __init__(self, tv_weight=1e-4):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", kx.transpose(2, 3).contiguous())
        self.tv_weight = tv_weight

    def sobel(self, y):
        y = F.pad(y, (1, 1, 1, 1), mode="replicate")
        return torch.cat([F.conv2d(y, self.kx), F.conv2d(y, self.ky)], 1)

    def forward(self, sr, hr):
        sr, hr = sr.float(), hr.float()
        grad = F.l1_loss(self.sobel(luminance(sr)), self.sobel(luminance(hr)))
        tv = (sr[:, :, 1:, :] - sr[:, :, :-1, :]).abs().mean() + (sr[:, :, :, 1:] - sr[:, :, :, :-1]).abs().mean()
        return grad + self.tv_weight * tv


class GeneratorLoss(nn.Module):
    """Eq. (13): L_G = l_pix L_pix + l_per L_per + l_adv L_adv + l_f L_freq + l_e L_edge.

    Terms with zero weight are not built, so the same class serves the
    ablation configurations A-D and the baselines.
    """

    def __init__(self, weights, gan_type="ragan", pixel="l1", perceptual_layers=None,
                 perceptual_criterion="mse", tv_weight=1e-4, vgg_pretrained=True):
        super().__init__()
        self.w = {k: float(v) for k, v in weights.items() if float(v) > 0}
        self.gan_type = gan_type
        self.pix = nn.L1Loss() if pixel == "l1" else nn.MSELoss()
        self.per = PerceptualLoss(perceptual_layers, perceptual_criterion, vgg_pretrained) if "per" in self.w else None
        self.freq = FrequencyLoss() if "freq" in self.w else None
        self.edge = EdgeLoss(tv_weight) if "edge" in self.w else None

    def forward(self, sr, hr, real_logit=None, fake_logit=None):
        sr, hr = sr.float(), hr.float()
        t = {}
        if "pix" in self.w:
            t["pix"] = self.pix(sr, hr)
        if self.per is not None:
            t["per"] = self.per(sr, hr)
        if "adv" in self.w and fake_logit is not None:
            t["adv"] = G_LOSSES[self.gan_type](real_logit.float() if real_logit is not None else None, fake_logit.float())
        if self.freq is not None:
            t["freq"] = self.freq(sr, hr)
        if self.edge is not None:
            t["edge"] = self.edge(sr, hr)
        total = sum(self.w[k] * v for k, v in t.items())
        return total, {k: float(v.detach()) for k, v in t.items()}
