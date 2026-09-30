#!/usr/bin/env python3
"""
scaling.py
==========
Objective scaling for the sugar-beet MORL formulation, with two sources.

ORACLE     The constants used for every result in the submitted paper. They
           are taken from the exhaustive sweep of the attainable set
           (Section 5.1), so the formulation is given prior knowledge of the
           objective geometry. This is the circularity raised in R2-6.

TRAINING   The same four constants derived only from data an agent collects
           before it learns anything: the WARMUP steps of uniformly random
           actions that train_multiseed.py runs on the training seeds, with
           domain randomization on, exactly as in training. No front is
           enumerated, no Pareto-optimal point is identified, and no
           validation or test seed is touched.

Both sources follow one rule, so that only the information changes and not
the design: each objective is mapped so that its observed range spans one
unit.

    extraction   r = (eta - lo) / (hi - lo)          lo, hi of extraction
    energy       r = (hi - G_total) / (hi - lo)      lo, hi of G_total
    concentration r = exp(-((Cc - hi) / (hi - lo))^2) hi = target, span = width

For ORACLE, lo and hi are the extremes of the attainable set. For TRAINING
they are the 2.5th and 97.5th percentiles of the warm-up data, which is what
a practitioner would use to discard transient outliers.

Which source is active is chosen by the environment variable MORL_SCALING,
default "oracle", so every existing result reproduces unchanged. The
environment and morl_score.py both read it through get_scaling(), which keeps
the training reward and the selection score on the same scale.

Usage
-----
    python scaling.py --derive            # writes scaling_training.json
    python scaling.py --show              # prints both, side by side
"""
import os, json, argparse, datetime
import numpy as np

ORACLE = dict(ext_off=0.96292, ext_sc=0.01469,
              gt_base=2.6737, gt_sc=0.7629,
              cc_target=12.974, cc_width=1.4505,
              source="oracle: extremes of the exhaustive attainable-set sweep (Section 5.1)")

JSON_PATH = os.environ.get("MORL_SCALING_FILE",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "scaling_training.json"))
KEYS = ("ext_off", "ext_sc", "gt_base", "gt_sc", "cc_target", "cc_width")
_CACHE = {}


def get_scaling():
    """The active scaling. Cached per process; the choice is fixed for a run."""
    mode = os.environ.get("MORL_SCALING", "oracle").strip().lower()
    if mode in _CACHE:
        return _CACHE[mode]
    if mode == "oracle":
        s = dict(ORACLE)
    elif mode == "training":
        if not os.path.exists(JSON_PATH):
            raise SystemExit(f"MORL_SCALING=training but {JSON_PATH} is missing. "
                             f"Run:  python scaling.py --derive")
        s = json.load(open(JSON_PATH))
        missing = [k for k in KEYS if k not in s]
        if missing:
            raise SystemExit(f"{JSON_PATH} lacks {missing}")
    else:
        raise SystemExit(f"MORL_SCALING must be 'oracle' or 'training', got '{mode}'")
    for k in ("ext_sc", "gt_sc", "cc_width"):
        if not s[k] > 1e-9:
            raise SystemExit(f"degenerate scaling: {k} = {s[k]}")
    s["mode"] = mode
    _CACHE[mode] = s
    return s


def derive_from_training_data(seeds=(0, 1, 2, 3, 4), steps=5000, lo_pct=2.5, hi_pct=97.5):
    """Replay the warm-up phase of train_multiseed.py and record the outputs.

    Mirrors training exactly: domain randomization and disturbances on, the
    same episode length, and one uniformly random action per step, as in
    `if step <= WARMUP: a = env.action_space.sample()`.
    """
    # The derivation reads physical outputs only, never the reward, so the
    # scaling mode is irrelevant here; force oracle so it runs before the
    # training file exists.
    os.environ["MORL_SCALING"] = "oracle"
    from sugar_extraction_env import SugarExtractionEnv
    import morl_score as MS
    ext, gt, cc = [], [], []
    for seed in seeds:
        env = SugarExtractionEnv(seed=seed, enable_disturbance=True, domain_randomize=True)
        env.action_space.seed(seed)
        env.reset(seed=seed, options={"sample_preference": True})
        for _ in range(steps):
            a = env.action_space.sample()
            _, _, term, trunc, info = env.step(a)
            e = float(info["extraction"])                        # fraction
            ext.append(e)
            gt.append(MS.total_steam(float(info["Gp"]), 100.0 * e, float(info["Cc"])))
            cc.append(float(info["Cc"]))
            if term or trunc:
                env.reset(options={"sample_preference": True})
    ext, gt, cc = map(np.asarray, (ext, gt, cc))
    q = lambda x: (float(np.percentile(x, lo_pct)), float(np.percentile(x, hi_pct)))
    (e_lo, e_hi), (g_lo, g_hi), (c_lo, c_hi) = q(ext), q(gt), q(cc)
    return dict(
        ext_off=e_lo, ext_sc=e_hi - e_lo,
        gt_base=g_hi, gt_sc=g_hi - g_lo,
        cc_target=c_hi, cc_width=c_hi - c_lo,
        source=(f"training data: warm-up replay, seeds {list(seeds)}, {steps} random-action steps "
                f"each, domain randomization on, percentiles {lo_pct}-{hi_pct}"),
        n_samples=int(ext.size),
        raw_ranges=dict(extraction=[e_lo, e_hi], G_total=[g_lo, g_hi], Cc=[c_lo, c_hi]),
        created=datetime.datetime.now().isoformat(timespec="seconds"))


def show(tr=None):
    o = ORACLE
    rows = [("extraction offset", "ext_off"), ("extraction scale", "ext_sc"),
            ("energy base", "gt_base"), ("energy scale", "gt_sc"),
            ("concentration target", "cc_target"), ("concentration width", "cc_width")]
    print(f"{'constant':24s}{'oracle':>12}{'training':>12}{'ratio':>9}")
    for lab, k in rows:
        t = tr[k] if tr else float('nan')
        r = t / o[k] if tr else float('nan')
        print(f"{lab:24s}{o[k]:>12.5f}{t:>12.5f}{r:>9.2f}")
    if tr:
        print(f"\n{tr['source']}\nsamples: {tr.get('n_samples')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--steps", type=int, default=5000)
    a = ap.parse_args()
    if a.derive:
        s = derive_from_training_data(steps=a.steps)
        json.dump(s, open(JSON_PATH, "w"), indent=2)
        print(f"wrote {JSON_PATH}\n"); show(s)
    elif a.show:
        show(json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else None)
    else:
        ap.print_help()
