"""Data pipeline for CIFAR-10 / CIFAR-100.

ATTRIBUTION: IMPLEMENTED BY OUR TEAM. Dataset download/parsing uses
torchvision.datasets (REUSED library code). The augmentation is the standard
CIFAR recipe (random 32x32 crop from a 4-pixel zero-padded image + random
horizontal flip, as in He et al. 2016 and mdistiller), but we implement it
directly on GPU tensors so the GPU is not starved by a slow CPU data loader
on Colab/Kaggle (a feasibility adjustment, not a change to the method).

Preprocessing order (identical for training, evaluation and the web app):
uint8 [0,255] -> float [0,1] -> (augment: train only) -> per-channel
normalization (x - mean_c) / std_c.
"""
import numpy as np
import torch
import torch.nn.functional as F

STATS = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

NUM_CLASSES = {"cifar100": 100, "cifar10": 10}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

CLASS_NAMES = {"cifar100": CIFAR100_CLASSES, "cifar10": CIFAR10_CLASSES}


def load_arrays(dataset, root="./data", train=True, download=True):
    """Return (images uint8 [N,32,32,3], labels int64 [N]) from torchvision."""
    from torchvision import datasets
    cls = {"cifar100": datasets.CIFAR100, "cifar10": datasets.CIFAR10}[dataset]
    ds = cls(root=root, train=train, download=download)
    assert list(ds.classes) == CLASS_NAMES[dataset], "class-name order mismatch"
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def normalize(x01, dataset):
    """x01: float tensor (B,3,H,W) in [0,1] -> channel-normalized tensor."""
    mean, std = STATS[dataset]
    mean = torch.tensor(mean, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def augment(x01, generator=None, pad=4):
    """Random crop (zero padding of `pad` px) + random horizontal flip, batched on-device."""
    b, c, h, w = x01.shape
    dev = x01.device
    padded = F.pad(x01, (pad, pad, pad, pad), value=0.0)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=generator)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=dev, generator=generator)
    rows = (oy[:, None] + torch.arange(h, device=dev)[None, :])  # (b,h)
    cols = (ox[:, None] + torch.arange(w, device=dev)[None, :])  # (b,w)
    bi = torch.arange(b, device=dev)[:, None, None, None]
    ci = torch.arange(c, device=dev)[None, :, None, None]
    out = padded[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]  # (b,c,h,w)
    flip = torch.rand(b, device=dev, generator=generator) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out


class TensorLoader:
    """Keeps the whole split on `device` as uint8 and yields normalized batches."""

    def __init__(self, images, labels, dataset, batch_size, train, device, seed=0, limit=None):
        if limit is not None:
            images, labels = images[:limit], labels[:limit]
        self.x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).contiguous()  # uint8 N,3,32,32
        self.y = torch.from_numpy(labels).to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def __len__(self):
        n = self.x.shape[0]
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.x.shape[0]
        if self.train:
            order = torch.randperm(n, device=self.device, generator=self.gen)
        else:
            order = torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            xb = self.x[idx].float().div_(255.0)
            if self.train:
                xb = augment(xb, generator=self.gen)
            yield normalize(xb, self.dataset), self.y[idx]
