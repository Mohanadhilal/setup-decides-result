# When the Setup Decides the Result

Code, configurations, seeds and raw per-seed outputs for:

> M. H. Alruyshid, N. A. Malk, M. A. Al-Murshidawy, O. F. Lutfy and Z. A. Kareem,
> "When the Setup Decides the Result: Objective Formulation and Baseline Design in
> Multi-Objective Reinforcement Learning for Sugar Beet Extraction",
> *International Journal of Intelligent Engineering and Systems*, 2026.

Every number in the paper was produced by the code at tag **`v1.0.2`**. The exact
commit is given in the paper's Code and data availability section.

## Contents

```
src/        plant model, environment, controllers, learners, scaling
scripts/    everything that produces a number, table or figure in the paper
configs/    baselines.yaml (every PID, MPC, fuzzy and draft-loop parameter),
            training.yaml, protocol.yaml, preferences.yaml
results/    raw per-seed outputs behind Tables 3 to 8 and the scaling experiment
docs/       ERRATA.md (the seven tooling errors), REPRODUCING.md
```

`configs/baselines.yaml` holds the complete specification of the three classical
baselines: the fuzzy controller's triangular membership functions and its nine-rule
base, the MPC horizon, weights and first-order internal model, the PID and draft-loop
gains, and the 8 x 4 set-point grid of Table 4. The same values are the ones the code
in `src/` uses.

## Install

```bash
git clone https://github.com/Mohanadhilal/setup-decides-result.git
cd setup-decides-result
git checkout v1.0.2
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used to produce the results.

## Seed protocol

| Set | Seeds | Used for |
|---|---|---|
| Training | 0 to 4 | network initialisation and training randomness only |
| Validation | 100 to 109 | checkpoint selection only |
| Test | 200 to 219 | every reported value; never used for selection |

A checkpoint is admissible only if it stays inside the 70 to 78 degree window on every
validation seed and its temperature ripple is at most 0.15 degrees. The installed
checkpoint of each run is frozen in `selection_protocol.json` before any test seed is used.

## Reproducing the paper

| Paper item | Command |
|---|---|
| Tables 3 and 8, selection | `python scripts/reeval_protocol.py --stage select` |
| Table 4, classical grid | `python scripts/reeval_protocol.py --stage classical` |
| Table 5, matched energy | `python scripts/reeval_protocol.py --stage headline` |
| Table 7, disturbances | `python scripts/robustness_protocol.py --pref 2 --workers 5` |
| Figures | `python scripts/make_figures.py` |

Retraining from scratch: `python scripts/train_multiseed.py --algos SAC TD3 PPO --prefs 2 --seeds 0 1 2 3 4 --workers 5`.

## Non-oracle scaling experiment (Section 5.2)

The objective scaling used for the main results takes its ranges from the exhaustive
sweep of the attainable set. Section 5.2 tests an alternative that uses training data
only: the six scaling constants are derived from the warm-up data an agent collects
before learning (uniform random actions on the training seeds, domain randomization
on, 2.5th to 97.5th percentiles). The active scaling is chosen by the environment
variable `MORL_SCALING` (`oracle` by default, which reproduces the paper exactly).

```powershell
python src/scaling.py --derive                 # writes scaling_training.json
$env:MORL_SCALING = "training"
$env:MORL_OUT_DIR = "results/runs_multiseed_training"
python scripts/train_multiseed.py --algos SAC --prefs 0 2 3 --seeds 0 1 2 3 4 --workers 5
python scripts/reeval_protocol.py --stage select --algos SAC
Remove-Item Env:MORL_SCALING; Remove-Item Env:MORL_OUT_DIR
python scripts/compare_scaling.py
```

The derived constants, the fifteen runs, the per-preference comparison and the verdict
are in `results/scaling/`.

## Known limitations

These are stated in the paper and bound what the code can show. The draft does not
move pulp loss in this model; the extraction relation is calibrated, not identified;
the predictive baseline is under-horizoned and its internal model miscalibrated; supply
pressure is available to the learned controller but fixed for the baselines; and the
delivered front depends on the attainable-range scaling, as the experiment above shows.

## Licence

MIT; see `LICENSE`. Citation metadata is in `CITATION.cff`.
