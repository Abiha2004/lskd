"""Small helpers. ATTRIBUTION: IMPLEMENTED BY OUR TEAM."""
import hashlib
import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def topk_correct(logits, targets, ks=(1, 5)):
    maxk = max(ks)
    pred = logits.topk(maxk, dim=1).indices
    hit = pred.eq(targets[:, None])
    return [hit[:, :k].any(dim=1).sum().item() for k in ks]


def sha256_of(path, n_chars=12):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_chars]


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
