"""Distillation losses: vanilla KD (Hinton et al., 2015) and KD with Z-score
logit standardization (Sun et al., CVPR 2024).

ATTRIBUTION: IMPLEMENTED BY OUR TEAM from the equations in the papers
(Sun et al. 2024, Algorithm 1 and Algorithm 2). We cross-checked our
`zscore` against the official repository
(https://github.com/sunshangquan/logit-standardization-KD), whose `normalize`
uses torch.std (unbiased, divides by K-1) plus a 1e-7 epsilon. We default to
the same choice so our numbers are comparable; `unbiased=False` gives the 1/K
version written in Algorithm 1 of the paper (difference is a factor
sqrt(K/(K-1)): 1.005 for CIFAR-100, 1.054 for CIFAR-10).
"""
import torch
import torch.nn.functional as F


def zscore(logits, unbiased=True, eps=1e-7):
    """Per-sample Z-score of a (batch, K) logit matrix: (z - mean(z)) / std(z).

    After this, every row has mean 0 and standard deviation 1, whatever the
    original scale of the model's logits was. The ordering (rank) of the
    classes is unchanged because this is a positive linear map.
    """
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True, unbiased=unbiased)
    return (logits - mean) / (std + eps)


def kd_kl(student_logits, teacher_logits, temperature):
    """KL(p_teacher || p_student) with a SHARED temperature, scaled by T^2.

    T^2 keeps the gradient magnitude comparable across temperatures
    (Hinton et al., 2015). Returns the batch mean.
    """
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1).mean()
    return kl * temperature ** 2


def distillation_loss(student_logits, teacher_logits, temperature, standardize, unbiased=True):
    """The only difference between vanilla KD and KD+LS is the `standardize` switch.

    standardize=False: softmax(z / T)              -> vanilla KD
    standardize=True : softmax(Zscore(z) / tau)     -> logit standardization
    (`temperature` plays the role of the base temperature tau in the second case.)
    """
    s = student_logits.float()
    t = teacher_logits.float()
    if standardize:
        s = zscore(s, unbiased=unbiased)
        t = zscore(t, unbiased=unbiased)
    return kd_kl(s, t, temperature)


def total_loss(student_logits, targets, teacher_logits, method, temperature,
               ce_weight, kd_weight, unbiased=True):
    """Algorithm 2, line 6: lambda_CE * CE(y, softmax(z)) + lambda_KD * tau^2 * KL(...).

    Returns (loss, ce_term, kd_term) where the terms are already weighted.
    Note: the CE term always uses the RAW student logits (no standardization),
    exactly as in the paper.
    """
    ce = F.cross_entropy(student_logits.float(), targets)
    if method == "ce":
        return ce, ce, torch.zeros_like(ce)
    standardize = method == "kd_ls"
    kd = distillation_loss(student_logits, teacher_logits, temperature, standardize, unbiased)
    ce_term = ce_weight * ce
    kd_term = kd_weight * kd
    return ce_term + kd_term, ce_term, kd_term
