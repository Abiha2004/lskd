"""CIFAR-style ResNets (ResNet-8/14/20/32/44/56/110).

ATTRIBUTION: ADAPTED. The layer layout and parameter names follow the CIFAR
ResNet used in the CRD / mdistiller code bases
(https://github.com/megvii-research/mdistiller, mdistiller/models/cifar/resnet.py)
so that the official pretrained ResNet-56 teacher checkpoint can be loaded with
`load_state_dict`. We re-wrote the module ourselves: unlike the original, our
forward() returns only the logits (no feature list), and we added
`build_model` / `load_model_weights` helpers.

Architecture (He et al., 2016, Sec. 4.2): a 3x3 conv stem with 16 filters,
then three stages of `n` BasicBlocks with 16, 32, 64 filters, where
depth = 6n + 2. Stages 2 and 3 halve the spatial size (32 -> 16 -> 8).
Global average pooling over the 8x8 map, then a linear classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 conv layers with a residual (skip) connection: out = relu(F(x) + x)."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample  # 1x1 conv + BN when the shape changes
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return F.relu(out)


class CifarResNet(nn.Module):
    def __init__(self, depth, num_classes=100, num_filters=(16, 16, 32, 64)):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2 (e.g. 20, 32, 56, 110)"
        n = (depth - 2) // 6
        self.depth = depth
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(num_filters[1], n, stride=1)
        self.layer2 = self._make_layer(num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 32x32x16
        x = self.layer1(x)                      # 32x32x16
        x = self.layer2(x)                      # 16x16x32
        x = self.layer3(x)                      # 8x8x64
        x = self.avgpool(x).flatten(1)          # 64
        return self.fc(x)                       # num_classes logits


ARCHS = {"resnet8": 8, "resnet14": 14, "resnet20": 20, "resnet32": 32,
         "resnet44": 44, "resnet56": 56, "resnet110": 110}


def build_model(arch, num_classes):
    if arch not in ARCHS:
        raise ValueError("Unknown arch %s; choose from %s" % (arch, list(ARCHS)))
    return CifarResNet(ARCHS[arch], num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def extract_state_dict(ckpt):
    """Accept our checkpoints, mdistiller checkpoints ({'model': ...}) or raw state dicts."""
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    # strip DataParallel / wrapper prefixes
    cleaned = {}
    for k, v in ckpt.items():
        for prefix in ("module.", "student.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v
    return cleaned


def load_model_weights(model, path, map_location="cpu"):
    """Load weights strictly and fail loudly if the checkpoint does not match."""
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older torch without weights_only
        ckpt = torch.load(path, map_location=map_location)
    state = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError("Checkpoint %s is missing %d keys, e.g. %s" % (path, len(missing), missing[:5]))
    if unexpected:
        print("[load_model_weights] ignoring %d unexpected keys, e.g. %s" % (len(unexpected), unexpected[:5]))
    return ckpt
