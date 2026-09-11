#!/usr/bin/env python3
"""
make_figures.py
===============
Regenerates the results figures from the protocol re-evaluation, so that every
figure is drawn from the same runs and the same seeds as Tables 3 to 7.

Figure 4  classical cascade (both setpoints swept) against the learned front,
          on the twenty held-out test seeds.
            inputs : runs_multiseed/classical_grid_test.csv
                     runs_multiseed/protocol_results.csv
            output : runs/fig4_classical_vs_learned.png (and .pdf)

Figure 3  operating points reachable under the two scalarizations.
          Drawn only if Tchebycheff runs exist under the same protocol
          (runs_multiseed_tch/protocol_results.csv). Otherwise skipped with a
          message: the figure in the manuscript predates the corrected training
          signal and must not be reused.
            output : runs/fig3_scalarization.png (and .pdf)

Figures 1 and 2 are properties of the plant model alone and are unaffected by
the retraining; they are not regenerated here.

Run:  python make_figures.py
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "runs_multiseed"
TCH = "runs_multiseed_tch"          # optional: Tchebycheff runs, same protocol
FIGDIR = "runs"
PREF_LABEL = {0: "[0.70, 0.10, 0.20]", 1: "[0.50, 0.30, 0.20]", 2: "[0.34, 0.33, 0.33]",
              3: "[0.20, 0.60, 0.20]", 4: "[0.20, 0.20, 0.60]", 5: "[0.29, 0.51, 0.20]"}
BAND = (2.151, 2.213)               # classical energy band of the submitted version

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 170, "savefig.dpi": 300})


def read(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run reeval_protocol.py first")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def staircase(points):
    """Non-dominated staircase: maximize y for increasing x."""
    xs, ys = [], []
    for x, y in sorted(points):
        if not ys or y > ys[-1]:
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def learned_points(path):
    """Per-preference mean over training seeds, plus the per-seed cloud."""
    rows = [r for r in read(path) if r["algo"] == "SAC"]
    by = {}
    for r in rows:
        by.setdefault(int(r["pref_i"]), []).append((float(r["G_total"]), float(r["extraction"])))
    means = {p: (float(np.mean([g for g, _ in v])), float(np.mean([e for _, e in v]))) for p, v in by.items()}
    return means, by


# ------------------------------------------------------------------ Figure 4
def figure4():
    grid = [r for r in read(os.path.join(OUT, "classical_grid_test.csv")) if r["admissible"] == "1"]
    means, cloud = learned_points(os.path.join(OUT, "protocol_results.csv"))

    G = np.array([float(r["G_total"]) for r in grid])
    E = np.array([float(r["extraction"]) for r in grid])
    T = np.array([float(r["T12"]) for r in grid])
    cx, cy = staircase(zip(G, E))

    lg = np.array([means[p][0] for p in sorted(means)])
    le = np.array([means[p][1] for p in sorted(means)])
    lx, ly = staircase(zip(lg, le))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sc = ax.scatter(G, E, c=T, s=34, cmap="coolwarm", zorder=3, linewidths=.3,
                    edgecolors="0.25", label="classical cascade, both setpoints swept")
    cb = plt.colorbar(sc, ax=ax); cb.set_label("settled temperature, °C")
    ax.plot(cx, cy, "-", color="0.30", lw=1.6, zorder=4,
            label=f"classical front ({len(cx)} points)")

    # per-seed cloud, then the preference means
    for p, v in cloud.items():
        ax.scatter([g for g, _ in v], [e for _, e in v], s=13, color="tab:blue",
                   alpha=.30, zorder=5, edgecolors="none")
    ax.plot(lx, ly, "-o", color="tab:blue", lw=1.8, ms=6.5, zorder=6,
            label=f"learned front ({len(lx)} points)")
    # label preferences; the cold cluster overlaps, so those are stacked and leadered
    order = sorted(means, key=lambda p: means[p][0])
    cold = sorted([p for p in order if means[p][0] < 2.10], key=lambda p: means[p][0])
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - 0.32 * (y1 - y0), y1 + 0.22 * (y1 - y0))
    yb = ax.get_ylim()[0]
    for k, p in enumerate(cold):
        g, e = means[p]
        ax.annotate(PREF_LABEL[p], xy=(g, e), xytext=(1.90 + 0.115 * k, yb + 0.035 * (k % 2) + 0.02),
                    fontsize=6.2, color="tab:blue", ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=.4, color="tab:blue", shrinkA=0, shrinkB=2.5))
    for p in order:
        if p in cold: continue
        g, e = means[p]
        ax.annotate(PREF_LABEL[p], (g, e), textcoords="offset points", xytext=(7, -10),
                    fontsize=6.4, color="tab:blue")

    ax.axvspan(*BAND, color="0.85", alpha=.55, zorder=1)
    ax.annotate("classical band of the\nsubmitted version", xy=(np.mean(BAND), 0.02),
                xycoords=("data", "axes fraction"), fontsize=6.3, color="0.4",
                ha="center", va="bottom")

    ax.set_xlabel("total steam, diffuser + evaporation, kg/s")
    ax.set_ylabel("extraction efficiency, %")
    ax.legend(fontsize=7.4, loc="upper left", framealpha=.95, borderpad=.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig4_classical_vs_learned.{ext}"), bbox_inches="tight")
    plt.close(fig)

    lo, hi = max(lx.min(), cx.min()), min(lx.max(), cx.max())
    g = np.linspace(lo, hi, 60)
    d = np.interp(g, lx, ly) - np.interp(g, cx, cy)
    print(f"Figure 4 written. admissible classical {len(grid)}; front {len(cx)} vs learned {len(lx)}")
    print(f"  overlap {lo:.3f}-{hi:.3f} kg/s | learned - classical: mean {d.mean():+.4f}, "
          f"range {d.min():+.4f} to {d.max():+.4f}, learned ahead {100*np.mean(d>0):.0f}%")
    print("  -> these must match the numbers in Table 4 of the manuscript")


# ------------------------------------------------------------------ Figure 3
def figure3():
    p_lin = os.path.join(OUT, "protocol_results.csv")
    p_tch = os.path.join(TCH, "protocol_results.csv")
    if not os.path.exists(p_tch):
        print(f"Figure 3 skipped: {p_tch} not found.")
        print("  The manuscript's Figure 3 was drawn before the training-signal fix and")
        print("  must not be reused. To regenerate it, set SCALARIZATION = 'tchebycheff'")
        print("  and Z_STAR in train_multiseed.py, train with OUT_DIR = 'runs_multiseed_tch',")
        print("  re-run reeval_protocol.py --stage select, then run this script again.")
        return
    lin, _ = learned_points(p_lin)
    tch, _ = learned_points(p_tch)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for d_, c, m, lab in ((lin, "tab:blue", "o", "weighted sum"),
                          (tch, "tab:red", "s", "augmented Tchebycheff")):
        g = [d_[p][0] for p in sorted(d_)]; e = [d_[p][1] for p in sorted(d_)]
        ax.plot(*staircase(zip(g, e)), "-", color=c, lw=1.6, zorder=3)
        ax.scatter(g, e, s=52, color=c, marker=m, edgecolors="k", linewidths=.5, zorder=4, label=lab)
    ax.axvspan(*BAND, color="0.85", alpha=.55, zorder=1)
    ax.text(np.mean(BAND), ax.get_ylim()[1], "classical band ", fontsize=6.6, va="top", ha="center", color="0.35")
    ax.set_xlabel("total steam, diffuser + evaporation, kg/s")
    ax.set_ylabel("extraction efficiency, %")
    ax.legend(fontsize=7.8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"fig3_scalarization.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 written.")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    figure4()
    figure3()
