"""Save a few CIFAR-100 TEST images (never used for training) as PNG files for the
demo app, named <index>_<true_label>.png.

  python scripts/export_demo_images.py --n 60
ATTRIBUTION: IMPLEMENTED BY OUR TEAM.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lskd.data import CLASS_NAMES, load_arrays  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="demo_images")
    args = p.parse_args()
    x, y = load_arrays(args.dataset, train=False)
    idx = np.random.RandomState(args.seed).choice(len(x), args.n, replace=False)
    os.makedirs(args.out, exist_ok=True)
    for i in idx:
        Image.fromarray(x[i]).save(os.path.join(args.out, "%05d_%s.png" % (i, CLASS_NAMES[args.dataset][y[i]])))
    print("Saved %d test images to %s/" % (args.n, args.out))


if __name__ == "__main__":
    main()
