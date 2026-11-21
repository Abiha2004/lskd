# Logit Standardization in Knowledge Distillation — reproduction, experiments & deployment

CSE-602 Machine Learning project (Fall 2026) — Abiha Jawaid (36395), Bismah Imran (36222), Zainab Asif (36232)

Paper: S. Sun, W. Ren, J. Li, R. Wang, X. Cao, *Logit Standardization in Knowledge Distillation*, CVPR 2024.

**Links (fill in before 25 Nov 23:55):**
- Code: `<GitHub URL>`
- Trained weights: `<Google Drive / Hugging Face URL>` (public, no permission request needed)

## What this repository does

A large **teacher** (ResNet-56, 72.34% on CIFAR-100) teaches a small **student** (ResNet-20, 3x fewer parameters).
We train the student two ways and compare them:

| method | what is compared in the distillation loss |
|---|---|
| Vanilla KD (Hinton et al., 2015) | `softmax(z_student / T)` vs `softmax(z_teacher / T)` with a shared temperature T=4 |
| KD + Logit Standardization (Sun et al., 2024) | `softmax(Zscore(z_student) / τ)` vs `softmax(Zscore(z_teacher) / τ)`, τ=2, where `Zscore(z) = (z - mean(z)) / std(z)` per image |

The only code difference is one `if standardize:` in `lskd/losses.py`.

## Repository layout

```
lskd/models.py        CIFAR ResNet-8…110 (layer names compatible with the official teacher checkpoint)
lskd/losses.py        Z-score, vanilla KD loss, KD+LS loss, total loss (paper Algorithm 1 & 2)
lskd/data.py          CIFAR loading, GPU-side augmentation, normalization, class names
lskd/experiments.py   every run in the report as a named preset + who runs what
lskd/inference.py     preprocessing + prediction used by the CLI and the web app
train.py              training (CE / KD / KD+LS), checkpoints every epoch, --resume, --smoke
evaluate.py           accuracy, ECE, logit statistics, agreement & rank-correlation with the teacher
infer.py              command-line inference on an image
app.py                Streamlit web app (deployment)
scripts/download_teacher.py      fetch official pretrained ResNet-56 teacher
scripts/export_demo_images.py    save unseen CIFAR-100 test images for the demo
scripts/make_report_assets.py    build report tables/figures from runs/ and results/
scripts/toy_example.py           re-computes the paper's Fig. 2 toy example (NumPy only)
tests/test_core.py               unit tests of the method's mathematical properties
run_experiments.ipynb            one-click driver for Colab / Kaggle GPUs
report/                          LaTeX source of the final report
```

## Code attribution (required by the course)

| Component | Status | Source |
|---|---|---|
| `lskd/losses.py` (Z-score, KD, KD+LS, total loss) | **Implemented by our team** from the paper's equations; checked against the official repo's `normalize()` | [sunshangquan/logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD) |
| `lskd/models.py` (CIFAR ResNet) | **Adapted** — rewritten by us, but parameter names/layout follow mdistiller so the official teacher loads | [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller) |
| Teacher weights ResNet-56 (CIFAR-100) | **Reused** (pretrained checkpoint) | mdistiller release `cifar_teachers.tar` |
| Training hyper-parameters | **Reused values** from the paper / mdistiller configs | Sun et al. 2024, Sec. 5 |
| `lskd/data.py`, `train.py`, `evaluate.py`, `infer.py`, `app.py`, `scripts/*`, `tests/*` | **Implemented by our team** | — |
| CIFAR download/parsing | **Reused library** | `torchvision.datasets` |
| CIFAR-10 teacher, all students | **Trained by our team** | — |

AI coding assistants were used while writing parts of this code; every component was run, tested and checked by the team.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                              # 7 tests of the method's properties
python scripts/toy_example.py                          # reproduces the paper's Fig. 2 numbers
```

## Training & experiments (GPU recommended: Colab / Kaggle T4)

Open `run_experiments.ipynb` on Colab or Kaggle, or from a terminal:

```bash
python scripts/download_teacher.py                     # official ResNet-56 teacher (72.34%)
python train.py --preset c100_kd_ls --smoke            # 1-minute pipeline check
python train.py --preset c100_kd                       # reproduction: vanilla KD
python train.py --preset c100_kd_ls                    # reproduction: KD + logit standardization
```

All presets (`lskd/experiments.py`):

| preset | purpose |
|---|---|
| `c100_kd`, `c100_kd_ls`, `c100_student_ce` | reproduction of paper Table 2 (ResNet56→ResNet20) |
| `abl_raw_w0.9`, `abl_raw_w9`, `abl_z_w0.9` (+ `c100_kd_ls`) | **Experiment A** — 2×2 ablation: is the gain from the Z-score or from the 10× larger KD weight? |
| `c10_teacher_resnet56`, `c10_kd`, `c10_kd_ls` | **Experiment B** — dataset variation: does LS still help with only 10 classes? |
| `tau1_kd_ls`, `tau4_kd_ls` | **Experiment C** (optional) — base temperature τ ∈ {1, 2, 4} |

Shared settings: SGD momentum 0.9, weight decay 5e-4, batch 64, lr 0.05, 240 epochs, lr ×0.1 at epochs 150/180/210, seed 0.
Each run writes `runs/<preset>/{history.csv, summary.json, final.pth, best.pth, last.pth}`. We report **final-epoch** accuracy (and best-epoch for comparison with the paper's convention).

## Evaluation

```bash
python evaluate.py --dataset cifar100 --teacher checkpoints/teachers/resnet56_cifar100.pth \
    --models "Vanilla KD=runs/c100_kd/final.pth" "KD+LS=runs/c100_kd_ls/final.pth"
python scripts/make_report_assets.py                   # -> report/generated/*.tex, report/figures/*
```

## Trained models

Place the files in `checkpoints/` (download link above):

| file | model | test top-1 |
|---|---|---|
| `checkpoints/student_kd.pth` | ResNet-20, vanilla KD (copy of `runs/c100_kd/final.pth`) | see report |
| `checkpoints/student_kd_ls.pth` | ResNet-20, KD + LS (copy of `runs/c100_kd_ls/final.pth`) | see report |
| `checkpoints/teachers/resnet56_cifar100.pth` | ResNet-56 teacher (official) | 72.34% |

Each of our checkpoints stores its architecture, dataset, method, normalization constants, class names and test accuracy, so inference needs nothing else.

## Deployment (web app)

```bash
python scripts/export_demo_images.py      # once: 60 unseen CIFAR-100 test images -> demo_images/
streamlit run app.py                      # opens http://localhost:8501
```

- **Input:** any PNG/JPG image (upload), or one of the unseen CIFAR-100 test images.
- **Preprocessing:** RGB → centre-crop to square → resize to 32×32 → scale to [0,1] → per-channel normalization with the CIFAR-100 mean (0.5071, 0.4867, 0.4408) and std (0.2675, 0.2565, 0.2761) → tensor (1,3,32,32).
- **Model:** ResNet-20 student(s) loaded from `checkpoints/` (CPU is enough: a few ms per image).
- **Output:** predicted class, top-5 classes with probabilities for each model, whether the students agree, raw logit mean/std, a plot of raw vs Z-scored logits, inference latency, and each checkpoint's SHA-256 so it is visible that the predictions come from the trained files.

Command-line alternative:

```bash
python infer.py --ckpt checkpoints/student_kd.pth checkpoints/student_kd_ls.pth --image demo_images/<file>.png
```

## Reproducibility notes

- Mixed precision (fp16 autocast) is used on GPU for speed; the loss is computed in fp32. Disable with `--no-amp`.
- Data augmentation runs on the GPU (same random-crop + flip recipe as the paper's code) to avoid the CPU bottleneck of free cloud GPUs.
- We use one seed per configuration (the paper averages 4), because of compute limits.
