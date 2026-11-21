"""Inference pipeline shared by infer.py (CLI) and app.py (Streamlit):

    Input image -> Preprocessing -> Trained model -> Prediction -> Output

Preprocessing for an arbitrary uploaded image: convert to RGB, centre-crop to
a square, resize to 32x32 (the CIFAR resolution the network was trained on),
scale to [0,1], then per-channel normalization with the CIFAR training-set
mean/std stored in the checkpoint. This is exactly the test-time transform
used in training/evaluation, so the deployed model sees the same input
distribution (apart from the domain shift of real photos vs CIFAR).

ATTRIBUTION: IMPLEMENTED BY OUR TEAM.
"""
import os
import time

import numpy as np
import torch
from PIL import Image

from lskd.data import CLASS_NAMES, STATS
from lskd.losses import zscore
from lskd.models import build_model, load_model_weights
from lskd.utils import sha256_of, torch_load


class LoadedModel:
    def __init__(self, path, device="cpu"):
        ck = torch_load(path)
        meta = ck if isinstance(ck, dict) else {}
        self.path = path
        self.arch = meta.get("arch", "resnet56")
        self.dataset = meta.get("dataset", "cifar100")
        self.num_classes = meta.get("num_classes", 100)
        self.method = meta.get("method", "teacher")
        self.class_names = meta.get("class_names", CLASS_NAMES[self.dataset])
        self.mean = tuple(meta.get("norm_mean", STATS[self.dataset][0]))
        self.std = tuple(meta.get("norm_std", STATS[self.dataset][1]))
        self.epoch = meta.get("epoch") if "arch" in meta else None  # external ckpts: unknown
        self.test_acc = meta.get("test_acc")
        self.sha256 = sha256_of(path)
        self.model = build_model(self.arch, self.num_classes)
        load_model_weights(self.model, path)
        self.model.eval().to(device)
        self.device = device


def preprocess(img, mean, std):
    """PIL image -> (1,3,32,32) normalized tensor, plus the 32x32 PIL image actually fed to the model."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    small = img.resize((32, 32), Image.BICUBIC) if s != 32 else img
    x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    m = torch.tensor(mean).view(1, 3, 1, 1)
    sd = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / sd, small


@torch.no_grad()
def predict(lm, img, topk=5):
    x, small = preprocess(img, lm.mean, lm.std)
    t0 = time.perf_counter()
    logits = lm.model(x.to(lm.device)).float()[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    probs = logits.softmax(0)
    p, idx = probs.topk(topk)
    return dict(
        top=[(lm.class_names[i], float(pp)) for pp, i in zip(p.tolist(), idx.tolist())],
        pred=lm.class_names[int(idx[0])], confidence=float(p[0]),
        logits=logits.cpu().numpy(), zlogits=zscore(logits[None])[0].cpu().numpy(),
        logit_mean=float(logits.mean()), logit_std=float(logits.std()),
        latency_ms=latency_ms, model_input=small,
    )


def discover_checkpoints(root="checkpoints"):
    found = []
    for dp, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".pth"):
                found.append(os.path.join(dp, f))
    return sorted(found)
