"""Streamlit demo: compare a student distilled with vanilla KD against a student
distilled with KD + logit standardization (and optionally the teacher).

    streamlit run app.py

Pipeline shown on screen: Input -> Preprocessing -> Trained model(s) -> Prediction -> Output
ATTRIBUTION: IMPLEMENTED BY OUR TEAM.
"""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from lskd.inference import LoadedModel, discover_checkpoints, predict  # noqa: E402

st.set_page_config(page_title="Logit Standardization KD - demo", layout="wide")


@st.cache_resource
def load(path):
    return LoadedModel(path)


def default_pick(paths, key):
    for p in paths:
        if key in os.path.basename(p):
            return p
    return None


st.title("Knowledge Distillation with Logit Standardization")
st.caption("CSE-602 project - reproduction of Sun et al., CVPR 2024. "
           "Teacher ResNet-56 -> student ResNet-20 on CIFAR-100. All predictions are computed live by "
           "the trained checkpoints listed below.")

# ---------------- sidebar: models ----------------
ckpts = discover_checkpoints("checkpoints")
if not ckpts:
    st.error("No .pth files found in ./checkpoints. See README section 'Trained models'.")
    st.stop()
with st.sidebar:
    st.header("Models")
    m1 = st.selectbox("Model A (vanilla KD)", ckpts,
                      index=ckpts.index(default_pick(ckpts, "student_kd.pth") or ckpts[0]))
    ls_default = default_pick(ckpts, "student_kd_ls") or ckpts[-1]
    m2 = st.selectbox("Model B (KD + logit standardization)", ckpts, index=ckpts.index(ls_default))
    show_teacher = st.checkbox("Also show the teacher", value=True)
    teacher_path = None
    if show_teacher:
        t_def = default_pick(ckpts, "resnet56") or ckpts[0]
        teacher_path = st.selectbox("Teacher", ckpts, index=ckpts.index(t_def))

models = [("A: vanilla KD", load(m1)), ("B: KD + LS", load(m2))]
if teacher_path:
    models.append(("Teacher", load(teacher_path)))

with st.sidebar:
    st.header("Provenance")
    for label, lm in models:
        acc = "%.2f%%" % lm.test_acc if lm.test_acc is not None else "n/a"
        st.markdown("**%s**  \n`%s`  \n%s, %s, %s classes  \nepoch %s, test acc %s  \nsha256 `%s`"
                    % (label, lm.path, lm.arch, lm.method, lm.num_classes,
                       (lm.epoch + 1) if lm.epoch is not None else "n/a", acc, lm.sha256))

# ---------------- input ----------------
st.subheader("1. Input")
source = st.radio("Image source", ["Upload an image", "Unseen CIFAR test image"], horizontal=True)
img, true_label = None, None
if source == "Upload an image":
    up = st.file_uploader("PNG / JPG, any size (it will be centre-cropped and resized to 32x32)",
                          type=["png", "jpg", "jpeg", "bmp", "webp"])
    if up is not None:
        img = Image.open(up)
else:
    files = sorted(f for f in os.listdir("demo_images") if f.endswith(".png")) if os.path.isdir("demo_images") else []
    if not files:
        st.warning("demo_images/ is empty - run `python scripts/export_demo_images.py` once.")
    else:
        if "demo_idx" not in st.session_state:
            st.session_state.demo_idx = 0
        c1, c2 = st.columns([1, 3])
        if c1.button("Random test image"):
            st.session_state.demo_idx = random.randrange(len(files))
        choice = c2.selectbox("or pick one", files, index=st.session_state.demo_idx)
        img = Image.open(os.path.join("demo_images", choice))
        true_label = choice.rsplit(".", 1)[0].split("_", 1)[1]

if img is None:
    st.info("Provide an image to run inference.")
    st.stop()

results = [(label, lm, predict(lm, img)) for label, lm in models]

st.subheader("2. Preprocessing")
c1, c2, c3 = st.columns(3)
c1.image(img.convert("RGB"), caption="Original input (%dx%d)" % img.size, width=220)
c2.image(results[0][2]["model_input"].resize((220, 220), Image.NEAREST),
         caption="What the network sees: 32x32 (shown enlarged)", width=220)
c3.markdown("- RGB, centre-crop to square\n- resize to **32x32** (bicubic)\n- scale to [0, 1]\n"
            "- normalize per channel: (x - mean) / std\n  - mean = %s\n  - std = %s\n- tensor shape (1, 3, 32, 32)"
            % (str(models[0][1].mean), str(models[0][1].std)))
if true_label:
    st.markdown("**Ground-truth label:** `%s`" % true_label)

st.subheader("3-4. Model predictions")
cols = st.columns(len(results))
for col, (label, lm, r) in zip(cols, results):
    ok = ""
    if true_label:
        ok = " (correct)" if r["pred"] == true_label else " (wrong)"
    col.markdown("### %s\n**%s** - %.1f%%%s" % (label, r["pred"], 100 * r["confidence"], ok))
    fig, ax = plt.subplots(figsize=(4, 2.4))
    names = [n for n, _ in r["top"]][::-1]
    probs = [100 * p for _, p in r["top"]][::-1]
    ax.barh(names, probs, color="#4C72B0")
    ax.set_xlim(0, 100)
    ax.set_xlabel("probability (%)")
    ax.set_title("Top-5")
    fig.tight_layout()
    col.pyplot(fig)
    plt.close(fig)
    col.caption("raw logits: mean %.2f, std %.2f | inference %.1f ms on CPU"
                % (r["logit_mean"], r["logit_std"], r["latency_ms"]))

if len(set(r["pred"] for _, _, r in results[:2])) == 1:
    st.success("Both students agree: %s" % results[0][2]["pred"])
else:
    st.warning("The two students disagree: A says %s, B says %s" % (results[0][2]["pred"], results[1][2]["pred"]))

st.subheader("5. Why logit standardization matters (this image)")
st.markdown("Left: the **raw logits** of each model, sorted by the teacher's ranking if available. "
            "Right: the same logits after the **Z-score** (mean 0, std 1). The paper's claim is that the KD+LS "
            "student is free to use its own logit scale (left) while matching the teacher's *relative* pattern (right).")
ref = results[-1][2]["logits"] if teacher_path else results[0][2]["logits"]
order = np.argsort(-ref)[:20]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for label, _, r in results:
    axes[0].plot(range(20), r["logits"][order], marker="o", ms=3, label=label)
    axes[1].plot(range(20), r["zlogits"][order], marker="o", ms=3, label=label)
axes[0].set_title("Raw logits (top-20 classes)")
axes[1].set_title("Z-scored logits")
for a in axes:
    a.set_xlabel("class rank (by reference model)")
    a.legend(fontsize=8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
