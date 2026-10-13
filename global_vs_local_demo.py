import numpy as np, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = stratified_train_test_split(X, y, test_size=0.3, random_state=0)
m = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
print("acc", round(m.score(Xte, yte), 3))

sv = shap.TreeExplainer(m).shap_values(Xte)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)

# Global: mean |SHAP|
g = np.abs(sv).mean(0)
order = np.argsort(-g)
gt = order[0]
name = X.columns[gt]

# Local: rank of the global-top feature inside each patient's own explanation
rank = (np.argsort(-np.abs(sv), axis=1) == gt).argmax(1) + 1
n = len(Xte)
print("global top feature:", name)
print("NOT top-1 locally:", (rank > 1).sum(), "of", n)
print("not in top-3:", (rank > 3).sum(), "of", n)

# ---- Figure ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)
fig.patch.set_facecolor("white")
BLUE, GREY, RED = "#2F5D8A", "#B8C0CC", "#C8553D"

# Panel 1: the "city map" (global)
top = order[:10][::-1]
colors = [BLUE if i == gt else GREY for i in top]
ax[0].barh([X.columns[i] for i in top], g[top], color=colors)
ax[0].set_title("Global view (the map)\nmean |SHAP|, top 10 features", fontsize=11, loc="left")
ax[0].set_xlabel("mean |SHAP value|")

# Panel 2: the "turn-by-turn" (local)
ranks = np.arange(1, 8)
counts = [(rank == r).sum() for r in ranks]
counts.append((rank > 7).sum())
labels = [str(r) for r in ranks] + ["8+"]
bar_colors = [BLUE] + [RED] * 7
ax[1].bar(labels, counts, color=bar_colors)
for i, c in enumerate(counts):
    ax[1].text(i, c + 1, str(c), ha="center", fontsize=9)
ax[1].set_title(f"Local view (the directions)\nrank of '{name}' in each patient's own explanation",
                fontsize=11, loc="left")
ax[1].set_xlabel("rank within one patient's explanation (1 = top driver)")
ax[1].set_ylabel(f"patients (n={n})")
ax[1].text(0.97, 0.93, f"Top driver for {n - (rank > 1).sum()} of {n}\nNot top driver for {(rank > 1).sum()} of {n}",
           transform=ax[1].transAxes, ha="right", va="top", fontsize=10)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Wisconsin breast cancer data, random forest, SHAP", fontsize=12, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("global_vs_local.png", bbox_inches="tight")
print("saved global_vs_local.png")