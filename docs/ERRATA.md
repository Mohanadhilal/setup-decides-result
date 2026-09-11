# Errors found in our own tooling

Seven errors were found in the scoring, selection, controller and training code written for this study.
None of them was neutral in effect: four flattered the learned controller, one exaggerated its defeat,
one would have flattered it again by crippling a baseline, and one produced a coherent negative result
that was entirely an artefact.

They are listed here because a paper arguing that configuration decides the outcome has no standing to
keep its own configuration errors private, and because the pattern is more useful than any single entry:
each error biased the result towards whichever conclusion was being pursued at the time it was written.

| # | Where | What went wrong | Effect | How it surfaced |
|---|---|---|---|---|
| i | scoring | an operating point was assumed independent of the preference input, which it is not, so an assignment was built on scores that did not describe the installed behaviour | flattered the learned controller | preference sweep gave the same point for very different weights |
| ii | scoring | the safety penalty was charged on the mean temperature rather than as the mean of the per-step penalty | flattered the learned controller | Jensen's inequality: the penalty of the mean understates the mean of a convex penalty, most for the oscillating policies that most need charging |
| iii | scoring | the averaging window differed from the one used for every other reported quantity | flattered the learned controller | a policy drifting out of the window near the end of an episode looked nearly safe |
| iv | checkpoint selection | the highest-scoring policy was installed with no safety condition attached | flattered the learned controller | installed policies with excursions up to 2.16 °C |
| v | controller | a bias correction intended to remove the tracking offset of the predictive controller compared the measurement against a one-step prediction rather than against the model steady state at the applied input | would have crippled a baseline | the correction converged to the one-step error, about 0.12 °C when the offset needing removal was about 9.5 °C |
| vi | comparison | a comparison script selected the learned checkpoint by filename order rather than by the pre-registered criterion | exaggerated the defeat of the learned controller | the tested policy sat at 0.047 kg/s of energy away from the one that met the criterion, breaking the energy matching for two of three baselines |
| vii | **training loop** | the replay buffer stored the environment's default scalar reward, the unweighted sum of the four channels, rather than the preference-scalarized return `u_w` | produced a coherent negative result that was entirely an artefact | see below |

## Error (vii) in detail

All five preferences trained on the same objective, because the stored reward did not depend on the
preference vector at all; the preference entered the observation but never the return. The consequence
was that the Pareto front collapsed to a single operating point: a steam range of 0.16 kg/s across the
five preferences, a temperature range of 1.2 °C, and Kruskal–Wallis p = 0.66, that is, the five
preferences were statistically indistinguishable.

That collapse was initially read as a finding, and it was a plausible one: it would have meant that
preference selection buys nothing once the constraint is placed correctly. It was diagnosed only when a
gate-strictness sweep failed to widen the front under any selection rule, which excluded the safety gate
as the cause and pointed at the training signal.

With the corrected signal the five preferences are distinct on every objective (steam p = 0.002,
temperature p = 0.0009, concentration p = 0.001, extraction p = 0.0009) and the front spans 0.57 kg/s.

The affected runs are preserved in `results/runs_multiseed_BUGGED_keep/` so that the artefact can be
reproduced and compared against the corrected runs in `results/runs_multiseed/`. They are not the runs
reported in the paper.

## What changed as a result

Errors (i) to (iv) are the reason the evaluation protocol is now defined once, in
`scripts/unified_evaluation.py`, and used by every script rather than reimplemented per experiment.
Error (vi) is the reason checkpoint selection is frozen to disk in `selection_protocol.json` instead of
being recomputed at comparison time. Error (vii) is the reason the training reward is scalarized
explicitly in the adapter and unit-tested: `scripts/train_multiseed.py` asserts that two different
preference vectors produce two different stored returns for the same transition.

---

## A note on what the twenty-seed protocol revealed

The corrections above were found by auditing code. One finding of the same kind came instead from
running the corrected protocol, and it is recorded here because it is the strongest single instance of
the paper's argument.

Section 5.5 originally averaged three seeds and reported the learned controller as three to seven times
worse than the baselines in peak thermal excursion: a quantitative disadvantage, inside a safe window.
Re-run on the twenty held-out test seeds and all five training runs, the picture is different in kind.
No classical controller left the thermal window in any of one hundred and eighty controller-seed-scenario
runs. One learned policy in five left it on every one of the twenty seeds, falling to 64.2 degrees, 5.8
below the floor, and never returning within the episode. That policy had passed the steady-state
admissibility gate on every validation seed and appears in Table 3 as an ordinary member of the balanced
ensemble.

The lesson is not that the learned controller is unsafe in general, but that a gate applied to settled
behaviour cannot detect a transient failure mode, and that a three-seed average cannot detect a mode that
affects one run in five. Raw per-seed outputs are in `results/tables/robustness_perseed.csv`.
