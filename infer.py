"""Command-line inference.

  python infer.py --ckpt checkpoints/student_kd_ls.pth --image demo_images/0001_apple.png
  python infer.py --ckpt checkpoints/student_kd.pth checkpoints/student_kd_ls.pth --image my_photo.jpg

Prints the top-5 classes with probabilities for each checkpoint.
ATTRIBUTION: IMPLEMENTED BY OUR TEAM.
"""
import argparse

from PIL import Image

from lskd.inference import LoadedModel, predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()
    img = Image.open(args.image)
    for path in args.ckpt:
        lm = LoadedModel(path)
        r = predict(lm, img)
        print("\n%s  [%s, %s, sha256 %s]" % (path, lm.arch, lm.method, lm.sha256))
        print("  logits: mean %.2f  std %.2f   latency %.1f ms" % (r["logit_mean"], r["logit_std"], r["latency_ms"]))
        for name, prob in r["top"]:
            print("  %-18s %6.2f%%" % (name, 100 * prob))


if __name__ == "__main__":
    main()
