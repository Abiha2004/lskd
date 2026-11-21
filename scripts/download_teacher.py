"""Download the official pretrained CIFAR-100 ResNet-56 teacher (72.34% top-1).

ATTRIBUTION: the teacher WEIGHTS are REUSED from the mdistiller release
(https://github.com/megvii-research/mdistiller/releases/tag/checkpoints),
which is the same teacher used by Sun et al. (2024) and by DKD/CTKD/MLKD.
This download script is ours.

If the download fails, train the teacher yourself instead:
    python train.py --preset c100_teacher_resnet56
    cp runs/c100_teacher_resnet56/final.pth checkpoints/teachers/resnet56_cifar100.pth
"""
import os
import shutil
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lskd.models import build_model, load_model_weights  # noqa: E402

URL = "https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar"
DEST = "checkpoints/teachers/resnet56_cifar100.pth"


def main():
    os.makedirs("checkpoints/teachers", exist_ok=True)
    tar_path = "checkpoints/cifar_teachers.tar"
    if not os.path.exists(tar_path):
        print("Downloading", URL)
        urllib.request.urlretrieve(URL, tar_path)
    extract_dir = "checkpoints/_cifar_teachers"
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if "resnet56" in m.name and m.name.endswith(".pth")]
        if not members:
            raise SystemExit("resnet56 checkpoint not found in the archive: %s" % [m.name for m in tf.getmembers()][:20])
        tf.extractall(extract_dir, members=members)
    src = os.path.join(extract_dir, members[0].name)
    shutil.copy(src, DEST)
    model = build_model("resnet56", 100)
    load_model_weights(model, DEST)  # raises if the layer names do not match
    print("Saved and verified teacher weights ->", DEST)
    print("(train.py prints its test accuracy; it should be ~72.34%)")


if __name__ == "__main__":
    main()
