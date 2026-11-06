"""Figures for the RealPDE Track-1 report. Every number is either analytic or
copied from the experiment record cited in the caption."""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def gauss_opt_k(rho):
    """k* with (sigma/s) phi(k) = erf(k/sqrt 2), i.e. the reward-optimal h/s."""
    lo, hi = 1e-9, 40.0
    for _ in range(100):
        k = 0.5 * (lo + hi)
        f = rho * math.exp(-0.5 * k * k) / math.sqrt(2 * math.pi) - math.erf(k / math.sqrt(2))
        lo, hi = (k, hi) if f > 0 else (lo, k)
    return 0.5 * (lo + hi)


def reward(k, s_over_sigma):
    return math.exp(-2 * k * s_over_sigma) * math.erf(k / math.sqrt(2))


def fig_sps():
    x = np.logspace(-2, 1, 300)                      # s / sigma
    k = np.array([gauss_opt_k(1 / v) for v in x])
    cov = np.array([math.erf(v / math.sqrt(2)) for v in k])
    r_opt = np.array([reward(a, b) for a, b in zip(k, x)])
    r90 = np.array([reward(1.6449, b) for b in x])
    r50 = np.array([reward(0.6745, b) for b in x])
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
    ax[0].semilogx(x, cov, color=C1)
    ax[0].set_xlabel(r"residual scale $s/\sigma$")
    ax[0].set_ylabel("optimal coverage")
    ax[0].set_ylim(0, 1)
    ax[1].semilogx(x, r90 / r_opt, color=C2, label="fixed 90% interval")
    ax[1].semilogx(x, r50 / r_opt, color=C3, label="fixed 50% interval")
    ax[1].set_xlabel(r"residual scale $s/\sigma$")
    ax[1].set_ylabel("reward / optimal reward")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend(loc="lower left")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, "sps_optimum.pdf"))


# Hidden-test (Main Development Phase) submissions, chronological; final_score.
LADDER = [
    ("914517", 79.918), ("915432", 80.555), ("916272", 80.578), ("917209", 80.610),
    ("918216", 80.252), ("919570", 80.594), ("921029", 80.796), ("921773", 80.698),
    ("923193", 80.782), ("924154", 80.820), ("925219", 80.779), ("926736", 80.856),
    ("929156", 81.635), ("930959", 81.698), ("933111", 81.808), ("935867", 81.863),
    ("939796", 81.863), ("940321", 81.812), ("942156", 81.861), ("943846", 81.878),
    ("945473", 81.846), ("947215", 81.885),
]
# index -> (label, vertical offset in score units)
NOTES = {0: ("2-D spectral net", -0.10), 1: ("conv ensemble", 0.12),
         4: ("constant-width probe", -0.10)}


def fig_ladder():
    y = np.array([v for _, v in LADDER])
    x = np.arange(1, len(y) + 1)
    best = np.maximum.accumulate(y)
    fig, ax = plt.subplots(figsize=(5.6, 2.1))
    ax.plot(x, best, color=C1, drawstyle="steps-post", linewidth=1.2, alpha=0.5,
            label="best so far")
    ax.plot(x, y, "o", color=C1, markersize=3.5, markeredgecolor=SURF,
            markeredgewidth=0.8, label="submission")
    for i, (t, dy) in NOTES.items():
        ax.annotate(t, (x[i], y[i]), (x[i], y[i] + dy), fontsize=6.5, color=INK2,
                    ha="center", va="bottom" if dy > 0 else "top")
    ax.annotate("", (13, 81.60), (13, 80.89),
                arrowprops=dict(arrowstyle="->", color=C2, lw=1.2))
    ax.text(13.3, 81.15, "width table refit: +0.78,\nweights unchanged", fontsize=6.5, color=INK2, va="center")
    ax.text(18, 82.02, "4-member blend, variance head, retrains", fontsize=6.5,
            color=INK2, ha="center", va="bottom")
    ax.set_xlabel("hidden-test submission (chronological)")
    ax.set_ylabel("final_score")
    ax.set_xticks(x[::3])
    ax.set_ylim(79.7, 82.3)
    ax.set_xlim(0, 23)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ladder.pdf"))


def fig_data():
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
    # E95 (scale-matched joint sim+real) and E108 (+ PIV mask and noise): mean of 2 seeds.
    d = [0.0, 0.25, 0.5, 0.75]
    e95 = [0.0, np.mean([-0.056, -0.164]), np.mean([-0.416, -0.512]), np.mean([-0.980, -1.001])]
    d108 = [0.0, 0.25, 0.5]
    e108 = [0.0, np.mean([-0.034, 0.008]), np.mean([-0.238, -0.154])]
    ax[0].axhline(0, color=INK2, linewidth=0.8)
    ax[0].plot(d, e95, "-o", color=C2, markersize=3.5, label="scale-matched sim")
    ax[0].plot(d108, e108, "-o", color=C1, markersize=3.5, label="+ PIV mask and noise")
    ax[0].set_xlabel("fraction of sim windows per batch")
    ax[0].set_ylabel(r"$\Delta$ objective vs real-only")
    ax[0].legend(loc="lower left")
    # E106: grouped learning curve over whole trajectories.
    n = [15, 30, 50, 73]
    c = [80.4785, 80.7478, 80.8671, 80.9842]
    ax[1].semilogx(n, c, "-o", color=C1, markersize=3.5)
    ax[1].set_xticks(n)
    ax[1].set_xticklabels([str(v) for v in n])
    ax[1].minorticks_off()
    ax[1].set_xlabel("real training trajectories")
    ax[1].set_ylabel("held-out-Re objective")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(OUT, "data_regime.pdf"))


if __name__ == "__main__":
    fig_sps()
    fig_ladder()
    fig_data()
    k = gauss_opt_k(1.0)
    print("s=sigma: k*=%.3f cov=%.3f" % (k, math.erf(k / math.sqrt(2))))
