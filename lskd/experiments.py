"""Registry of every training run in the report, so each run is reproducible
with one command:  python train.py --preset <name>

Defaults follow the paper / mdistiller CIFAR-100 recipe: SGD (momentum 0.9,
weight decay 5e-4), batch 64, lr 0.05, 240 epochs, lr x0.1 at epochs
150/180/210. Vanilla KD uses T=4, lambda_CE=0.1, lambda_KD=0.9 (mdistiller
default); KD+LS uses base temperature tau=2, lambda_CE=0.1, lambda_KD=9
(official logit-standardization-KD recipe; Sun et al. 2024, Sec. 5 & Tab. 4).

ATTRIBUTION: IMPLEMENTED BY OUR TEAM (hyper-parameter values taken from the paper).
"""

T100 = "checkpoints/teachers/resnet56_cifar100.pth"   # official pretrained teacher (72.34%)
T10 = "runs/c10_teacher_resnet56/final.pth"            # teacher we train ourselves on CIFAR-10

_C100 = dict(dataset="cifar100", arch="resnet20", teacher_arch="resnet56", teacher_ckpt=T100)
_C10 = dict(dataset="cifar10", arch="resnet20", teacher_arch="resnet56", teacher_ckpt=T10)

PRESETS = {
    # ---------------- Reproduction (paper Table 2, ResNet56 -> ResNet20) ----------------
    "c100_kd":        dict(_C100, method="kd",    temperature=4.0, ce_weight=0.1, kd_weight=0.9),
    "c100_kd_ls":     dict(_C100, method="kd_ls", temperature=2.0, ce_weight=0.1, kd_weight=9.0),
    # student trained alone (no teacher) = "Student" row of Table 2 (69.06%)
    "c100_student_ce": dict(dataset="cifar100", arch="resnet20", method="ce"),
    # fallback: train our own CIFAR-100 teacher if the official checkpoint cannot be downloaded
    "c100_teacher_resnet56": dict(dataset="cifar100", arch="resnet56", method="ce"),

    # ---------------- Additional experiment A: ablation (is it the Z-score or the big KD weight?) ----
    # 2x2 design at a common temperature of 2 (same as paper Table 4):
    #   {raw logits, Z-scored logits} x {lambda_KD = 0.9, lambda_KD = 9}
    "abl_raw_w0.9":   dict(_C100, method="kd",    temperature=2.0, ce_weight=0.1, kd_weight=0.9),
    "abl_raw_w9":     dict(_C100, method="kd",    temperature=2.0, ce_weight=0.1, kd_weight=9.0),
    "abl_z_w0.9":     dict(_C100, method="kd_ls", temperature=2.0, ce_weight=0.1, kd_weight=0.9),
    # (Z-score, w=9) cell is exactly c100_kd_ls -> no extra run needed

    # ---------------- Additional experiment B: dataset variation (CIFAR-10, K=10) ----------------
    "c10_teacher_resnet56": dict(dataset="cifar10", arch="resnet56", method="ce"),
    "c10_kd":         dict(_C10, method="kd",    temperature=4.0, ce_weight=0.1, kd_weight=0.9),
    "c10_kd_ls":      dict(_C10, method="kd_ls", temperature=2.0, ce_weight=0.1, kd_weight=9.0),

    # ---------------- Additional experiment C (optional): base temperature tau ----------------
    "tau1_kd_ls":     dict(_C100, method="kd_ls", temperature=1.0, ce_weight=0.1, kd_weight=9.0),
    "tau4_kd_ls":     dict(_C100, method="kd_ls", temperature=4.0, ce_weight=0.1, kd_weight=9.0),
}

# Suggested split of runs across the three team members' free GPU accounts
# (Kaggle gives 30 GPU-hours/week per account; Colab free is less predictable).
# Order inside each list = priority.
ASSIGNMENT = {
    "member1": ["c100_kd", "c100_kd_ls", "c100_student_ce"],
    "member2": ["abl_raw_w0.9", "abl_raw_w9", "abl_z_w0.9"],
    "member3": ["c10_teacher_resnet56", "c10_kd", "c10_kd_ls", "tau1_kd_ls", "tau4_kd_ls"],
}
