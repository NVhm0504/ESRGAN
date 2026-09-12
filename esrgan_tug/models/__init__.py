"""Model registry."""
import torch
import torch.nn as nn

from .baselines import SRCNN, VDSR, Bicubic, SRResNet
from .discriminators import UNetDiscriminatorSN, VGGDiscriminator
from .rrdbnet import RRDBNet

GENERATORS = {"rrdbnet": RRDBNet, "srresnet": SRResNet, "srcnn": SRCNN, "vdsr": VDSR, "bicubic": Bicubic}
DISCRIMINATORS = {"vgg": VGGDiscriminator, "unet_sn": UNetDiscriminatorSN}


def build_generator(spec, scale=4):
    spec = dict(spec)
    return GENERATORS[spec.pop("arch")](scale=scale, **spec)


def build_discriminator(spec, input_size=192):
    spec = dict(spec)
    arch = spec.pop("arch")
    if arch == "vgg":
        spec.setdefault("input_size", input_size)
    return DISCRIMINATORS[arch](**spec)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizedClassifier(nn.Module):
    """Wraps a torchvision classifier so it consumes RGB tensors in [0, 1]."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def build_classifier(n_classes=5, arch="resnet50", pretrained=True):
    import torchvision
    ctor = getattr(torchvision.models, arch)
    net = None
    if pretrained:
        try:
            net = ctor(weights="DEFAULT")
        except Exception as e:  # offline machine
            print(f"[classifier] ImageNet weights unavailable ({e}); training {arch} from scratch")
    if net is None:
        net = ctor(weights=None)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return NormalizedClassifier(net)
