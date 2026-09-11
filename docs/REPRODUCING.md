# Reproducing each table and figure

Every reported number traces to a command and an output file. Paths are relative to the repository root.
Run the commands in the order given in the README; the dependencies are noted below.

## Tables

| Paper | Produced by | Raw output | Notes |
|---|---|---|---|
| Table 1 — nominal operating point | `python src/sugar_extraction_model.py` | printed | model only, no learning, no seeds |
| Table 2 — settings | none; it *is* the configuration | `configs/` | seed sets, gates, hyperparameters |
| Table 3 — delivered operating points | steps 1–3 | `results/runs_multiseed/protocol_summary.csv` | rows are the six preferences, mean ± sd over five training runs on twenty test seeds |
| Table 4 — classical front, both set points swept | step 3 (`--stage classical`) | `results/runs_multiseed/classical_grid_test.csv` | 96 grid points, 68 admissible; front is the non-dominated staircase |
| Table 5 — matched-energy paired comparison | step 3 (`--stage matched`) | `results/runs_multiseed/matched_comparison.csv`, `matched_summary.csv` | p-values are over matched pairs, not over runs; see the caption |
| Table 6 — correction sequence | see below | mixed | one row per configuration; the first five are superseded configurations, the sixth is the current one |
| Table 7 — algorithms under a matched budget | steps 1, 3 | `results/runs_multiseed/protocol_summary.csv` (rows `TD3`, `PPO`, and `SAC` at pref 2) | same runs as the balanced row of Table 3 |
| Table 8 — step disturbances | step 5 | `results/runs_multiseed/robustness_summary.csv`, `robustness_perseed.csv`, `robustness_tests.csv` | twenty test seeds, five runs, three scenarios |
| Table 9 — safety-gate strictness | step 4 | `results/runs_multiseed/gate_sweep_summary.csv`, `gate_sweep_width.csv` | re-selection only, no retraining |
| Table 10 — the protocol | none; it is the checklist | — | each row cites the section that measured its cost |

### Table 6 row by row

| Row | Configuration | Where it comes from |
|---|---|---|
| 1 | diffuser steam only, single policy | superseded configuration; energy objective without the evaporation term of Eq. (13) |
| 2 | plant-level energy, before rescaling | same runs scored with Eq. (13) |
| 3 | plant-level energy, after rescaling | objectives normalized by attainable ranges (Section 4.3) |
| 4 | baselines given a draft loop | `baseline_controllers_mv.py` instead of `baseline_controllers.py` |
| 5 | baselines with both set points swept | step 3 `--stage classical`, one set point vs both (Table 4 columns) |
| 6 | five training seeds, disjoint test seeds | steps 1–3, the current protocol |

## Figures

| Paper | Produced by | Output |
|---|---|---|
| Figure 1 — extraction against temperature | `python src/sugar_extraction_model.py --sweep-temperature` | model only |
| Figure 2 — exact Pareto front and convex hull | `python scripts/make_figures.py` | `results/fig2_pareto_hull.png` |
| Figure 3 — classical front against the learned front | `python scripts/make_figures.py` | `results/fig3_classical_vs_learned.png` |

`make_figures.py` prints the same overlap statistics that appear in Table 4; if the printed numbers and
the table disagree, one of them was generated from a stale run.

## Checkpoints

Model weights are large and are not committed to git. Each run directory contains
`selection_protocol.json`, which names the installed checkpoint, the gate that admitted it, and both seed
lists, so the selection is auditable without the weights.

The weights themselves are released as a versioned archive: `<FILL IN: Zenodo DOI or GitHub release URL>`.
Unpack it into `results/` to evaluate the exact policies reported in the paper without retraining:

```bash
python scripts/reeval_protocol.py --stage select --workers 5
```

## Runtime

Measured on a six-core desktop CPU, `--workers 5`:

| Step | Wall clock |
|---|---|
| 1 — three algorithms × five seeds | ~4 h |
| 2 — SAC × five preferences × five seeds | ~6 h |
| 3 — re-selection and classical grid | ~2.5 h |
| 4 — gate sweep | ~40 min |
| 5 — disturbances | ~20 min |
| 6, 7 — offset control, figures | ~10 min |
