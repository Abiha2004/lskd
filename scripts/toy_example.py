"""Re-compute the toy case of Sun et al. (2024), Fig. 2, with NumPy only.

Classes: (cat, dog, bird, frog). Temperature 1.
  Teacher     logits (1.0, 4.0, 3.0, 2.0)  -> predicts dog
  Student S1  logits (1.0, 2.8, 3.0, 2.0)  -> close to the teacher in magnitude, predicts bird (wrong)
  Student S2  logits (0.1, 0.4, 0.3, 0.2)  -> teacher / 10: same ranking, predicts dog (right)

Vanilla KL says S1 is the better student (smaller loss) although it is wrong.
After Z-score standardization, S2 has zero loss (it carries exactly the
teacher's relations) and S1 has a positive loss - the loss now agrees with the
predictions. ATTRIBUTION: IMPLEMENTED BY OUR TEAM (logit values read off Fig. 2).
"""
import numpy as np

CLASSES = ["cat", "dog", "bird", "frog"]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def zscore(z, ddof=1):
    return (z - z.mean()) / z.std(ddof=ddof)


def run(verbose=True):
    T = np.array([1.0, 4.0, 3.0, 2.0])
    S1 = np.array([1.0, 2.8, 3.0, 2.0])
    S2 = np.array([0.1, 0.4, 0.3, 0.2])
    out = {}
    for name, s in (("S1", S1), ("S2", S2)):
        out[name] = dict(
            pred=CLASSES[int(s.argmax())],
            kl_raw=kl(softmax(T), softmax(s)),
            kl_z=kl(softmax(zscore(T)), softmax(zscore(s))),
            z=np.round(zscore(s), 3).tolist(),
        )
    if verbose:
        print("Teacher prediction:", CLASSES[int(T.argmax())], " Z-scored teacher:", np.round(zscore(T), 3).tolist())
        for k, v in out.items():
            print("%s predicts %-4s | vanilla KL = %.4f | KL after Z-score = %.4f | Z-scored logits %s"
                  % (k, v["pred"], v["kl_raw"], v["kl_z"], v["z"]))
    return out


if __name__ == "__main__":
    run()
