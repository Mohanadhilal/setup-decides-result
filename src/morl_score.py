#!/usr/bin/env python3
"""
morl_score.py
=============
Single source of truth for the scalarized score used to rank checkpoints.

Every selection script must agree with the reward the agent was actually
trained on. When the energy objective moved from diffuser steam to
plant-level steam, the scoring code embedded in the selection scripts did
not move with it, so those scripts would have ranked candidates by an
objective nobody optimises any more. Keeping the definition in one file
removes that failure mode.

If sugar_extraction_env.py changes again, change it here too, and nowhere
else.
"""

import numpy as np

# ---- must match sugar_extraction_env.py -------------------------------
# Scaling comes from scaling.get_scaling(), so the selection score always uses
# the same scale the agent was trained on (oracle by default).
from scaling import get_scaling as _get_scaling
_S = _get_scaling()
EXT_OFFSET, EXT_SCALE = _S["ext_off"], _S["ext_sc"]
GT_BASE, GT_SCALE = _S["gt_base"], _S["gt_sc"]
CC_TARGET, CC_WIDTH = _S["cc_target"], _S["cc_width"]
T_LO, T_HI = 70.0, 78.0                  # thermal window

KAPPA = 0.30        # kg steam per kg water evaporated
CC_THICK = 65.0     # thick juice concentration, %
G_SOLID = 3.2       # beet feed, kg/s
C_BEET0 = 0.175     # beet sugar fraction
# -----------------------------------------------------------------------


def total_steam(Gp, extraction_pct, Cc_pct):
    """Diffuser steam plus the evaporation duty implied by the juice
    concentration. Sugar output follows from the beet feed and the extraction
    efficiency, so diluting the juice adds evaporation load without adding
    product."""
    sugar = G_SOLID * C_BEET0 * extraction_pct / 100.0
    cc = max(float(Cc_pct), 1e-3)
    water = max(sugar * 100.0 * (1.0 / cc - 1.0 / CC_THICK), 0.0)
    return Gp + KAPPA * water


def penalty(T12):
    dev = max(T_LO - T12, 0.0) + max(T12 - T_HI, 0.0)
    return min(1.0 * dev + 0.5 * dev ** 2, 5.0)


def objectives(mean):
    """The three scalarized objectives from an evaluated operating point.
    `mean` is the dict returned by unified_evaluation.evaluate()['mean']."""
    r_ext = (mean["extraction"] / 100.0 - EXT_OFFSET) / EXT_SCALE
    r_en = (GT_BASE - total_steam(mean["Gp"], mean["extraction"],
                                  mean["Cc"])) / GT_SCALE
    r_cn = float(np.exp(-((mean["Cc"] - CC_TARGET) / CC_WIDTH) ** 2))
    return r_ext, r_en, r_cn


def score(mean, w):
    """Preference-weighted score with the safety channel outside the
    scalarization, exactly as the agent sees it.

    The penalty is taken as the mean of the per-step penalty when the
    evaluation supplies it. Falling back to penalty(mean T12) understates
    the cost of any policy that oscillates across a limit, because the
    penalty is convex; that fallback exists only for older result files and
    should not be relied on.
    """
    r_ext, r_en, r_cn = objectives(mean)
    if "mean_penalty" in mean:
        pen = mean["mean_penalty"]
    else:
        pen = penalty(mean["T12"])
    return w[0] * r_ext + w[1] * r_en + w[2] * r_cn - pen


if __name__ == "__main__":
    # A quick self-check against the reported operating points, so a wrong
    # constant shows up immediately rather than silently misranking policies.
    pts = [("PID", 1.2054, 96.9218, 12.3060, 73.499),
           ("MPC", 1.2325, 97.0036, 12.3115, 73.843),
           ("Fuzzy", 1.1864, 96.9091, 12.3088, 73.466),
           ("MORL balanced", 0.7073, 96.7119, 11.9430, 73.202)]
    w = [1 / 3, 1 / 3, 1 / 3]
    print(f"{'point':<16}{'G_total':>9}{'r_ext':>8}{'r_en':>8}"
          f"{'r_conc':>8}{'score':>9}")
    for n, gp, e, cc, t in pts:
        m = dict(Gp=gp, extraction=e, Cc=cc, T12=t)
        rt = total_steam(gp, e, cc)
        a, b, c = objectives(m)
        print(f"{n:<16}{rt:>9.4f}{a:>8.3f}{b:>8.3f}{c:>8.3f}"
              f"{score(m, w):>9.4f}")
