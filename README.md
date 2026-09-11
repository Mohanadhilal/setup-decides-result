# When the Setup Decides the Result

Code, configurations, seeds and raw outputs for the paper

> **When the Setup Decides the Result: Objective Formulation and Baseline Design in Multi-Objective
> Reinforcement Learning for Sugar Beet Extraction**
> *International Journal of Intelligent Engineering and Systems* (under review).

The paper asks how much of a reported advantage for a preference-driven learned controller is produced by
the learning and how much by five configuration choices: the objective boundary, the scalarization, the
objective scaling, the placement of the safety constraint, and the degrees of freedom granted to the
baseline. Four of the five move the headline number and two reverse its sign.

Because the paper's argument is that configuration decides the outcome, every configuration it used is in
this repository. **The commit that produced the submitted results is tagged `paper-v1`
(commit `<FILL IN: 40-character SHA>`).** Results reported in the paper should be reproduced from that tag,
not from `main`.

---

## 1. What is here

```
.
├── src/                         the plant, the controllers, the learner, the scoring
│   ├── sugar_extraction_model.py    12-cell diffuser model, Eqs. (1)-(4)
│   ├── sugar_extraction_env.py      Gymnasium environment, vector reward, Eqs. (9)-(12)
│   ├── baseline_controllers.py      PID, MPC, fuzzy, single-variable versions
│   ├── baseline_controllers_mv.py   the same three with the draft loop (Section 5.3)
│   ├── morl_score.py                plant-level energy, Eq. (13); scalarizations, Eqs. (14)-(15)
│   └── compare_rl_algorithms.py     SAC, TD3, PPO implementations
├── scripts/                     everything that produces a number in the paper
│   ├── train_multiseed.py           multi-seed training + seed-separated selection (Section 4.4)
│   ├── reeval_protocol.py           protocol-exact re-selection, classical grid, matched comparison
│   ├── reselect_gate_sweep.py       safety-gate strictness sweep (Section 5.7)
│   ├── robustness_protocol.py       step disturbances on the test seeds (Section 5.5)
│   ├── eval_classical_seedsets.py   seed-set offset control experiment (Section 4.4)
│   ├── classical_grid_sweep.py      two-dimensional baseline set-point sweep (Table 4)
│   ├── unified_evaluation.py        the single evaluation protocol used everywhere
│   ├── diagnose_extraction.py       draft-sensitivity diagnostic (Section 3.4)
│   └── make_figures.py              Figures 2 and 3 from the stored outputs
├── configs/                     hyperparameters, seed sets, preference sets, gate thresholds
├── results/                     raw CSV outputs behind Tables 3-8 (see docs/REPRODUCING.md)
│   ├── runs_multiseed/              corrected runs: checkpoints, selections, per-seed test scores
│   └── runs_multiseed_BUGGED_keep/  the superseded runs affected by error (vii); see docs/ERRATA.md
└── docs/
    ├── REPRODUCING.md               table-by-table: which command produces which file
    └── ERRATA.md                    the seven errors found in our own tooling
```

`src/` and `scripts/` are separated only for readability; the scripts expect `src/` on the import path,
which `scripts/run_all.sh` and `configs/paths.py` handle.

---

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.11 and 3.12, CPU only; no GPU is required or used. `requirements.txt` pins exact
versions. The full pipeline takes roughly 30 to 40 CPU-hours; with `--workers 5` on a six-core machine the
wall-clock time is about eight hours, dominated by training.

---

## 3. The seed protocol

Every learned-controller number in the paper comes from three **disjoint** seed sets. This is the
correction described in Section 4.4 and it is enforced in code, not by convention.

| Set | Seeds | Used for | Never used for |
|---|---|---|---|
| Training | 0–4 | network initialization, environment randomness during training | evaluation of any kind |
| Validation | 100–109 | scoring checkpoints, applying the safety and ripple gates, selecting the installed checkpoint | any reported number |
| Test | 200–219 | every value in Tables 3–8 and every figure | selection of anything |

The selected checkpoint of each run is frozen in `selection_protocol.json` inside that run's directory,
recording the checkpoint index, the gate that admitted it, and both seed lists. A reported number can
therefore be traced to a specific checkpoint file.

Admissibility gate (Section 4.4): a checkpoint is admissible only if, on **every** validation seed, the
steady-state temperature stays inside [70, 78] °C and the temperature ripple is at most 0.15 °C.

---

## 4. Reproducing the paper

Run in this order. Each step writes into `results/` and is skippable if its outputs are already present.

```bash
# 0. sanity check, ~30 seconds, no training
python scripts/train_multiseed.py --mock

# 1. algorithm comparison, balanced preference          -> Table 7
python scripts/train_multiseed.py --algos SAC TD3 PPO --prefs 2 --seeds 0 1 2 3 4 --workers 5

# 2. the rest of the preference set                     -> Table 3
python scripts/train_multiseed.py --algos SAC --prefs 0 1 3 4 5 --seeds 0 1 2 3 4 --workers 5

# 3. protocol-exact re-selection, classical grid,
#    and the matched-energy comparison                  -> Tables 3, 4, 5, 7
python scripts/reeval_protocol.py --stage all --workers 5

# 4. safety-gate strictness sweep                       -> Table 9
python scripts/reselect_gate_sweep.py --algos SAC --workers 5

# 5. step disturbances on the test seeds                -> Table 8
python scripts/robustness_protocol.py --pref 2 --workers 5

# 6. seed-set offset control (classical, no selection)  -> Section 4.4
python scripts/eval_classical_seedsets.py

# 7. figures                                            -> Figures 2, 3
python scripts/make_figures.py
```

`scripts/run_all.sh` executes the same sequence.

Step 3 depends on steps 1 and 2; steps 4 and 5 depend on step 3. Steps 6 and 7 are independent of
training and take minutes.

Table 6 (the correction sequence) is a summary of configurations that are individually reproduced by the
steps above together with the superseded configurations preserved in
`results/runs_multiseed_BUGGED_keep/`; `docs/REPRODUCING.md` maps each of its six rows.

---

## 5. Exactness and what will not match to the last digit

Floating-point reduction order differs across BLAS builds and thread counts, so a rerun may differ in the
third or fourth decimal of a scalarized return. Nothing in the paper rests on that precision; the reported
differences are between 0.03 and 0.5 percentage points of extraction, two to three orders of magnitude
larger. Set `OMP_NUM_THREADS=1` and `--workers 1` for bitwise-comparable runs at the cost of speed.

Training is seeded but **not** deterministic across PyTorch versions. `requirements.txt` pins the version
used. If you must verify the exact numbers rather than the conclusions, evaluate the shipped checkpoints
instead of retraining:

```bash
python scripts/reeval_protocol.py --stage select --workers 5
```

---

## 6. Errors found in our own tooling

Seven errors were found in the scoring, selection, controller and training code written for this study,
and none of them was neutral in effect. They are documented individually in
[`docs/ERRATA.md`](docs/ERRATA.md), with what each one biased and how it was found.

The seventh sat in the training loop rather than the evaluation: the replay buffer stored the
environment's default scalar reward instead of the preference-scalarized return, so all preferences
trained on the same objective and the Pareto front collapsed to a single operating point. That collapse
was initially read as a finding. The affected runs are kept in
`results/runs_multiseed_BUGGED_keep/` so that the incorrect result can be reproduced and compared against
the corrected one; they are **not** the runs reported in the paper.

---

## 7. Citation

```bibtex
@article{<FILL IN: key>,
  title   = {When the Setup Decides the Result: Objective Formulation and Baseline Design in
             Multi-Objective Reinforcement Learning for Sugar Beet Extraction},
  author  = {<FILL IN: author list>},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {<FILL IN>},
  note    = {Code: https://github.com/<FILL IN>/<FILL IN>, tag \texttt{paper-v1}}
}
```

## 8. License

Code released under the MIT License (`LICENSE`). The diffuser model is a re-implementation of published
equations, cited in the paper; it is not plant data and contains no proprietary information.

## 9. Contact

Issues and questions: please open a GitHub issue rather than emailing, so that answers are visible to
other readers. Corresponding author: `<FILL IN>`.
