# Before making the repository public

Six steps, in order. The whole thing takes under an hour.

## 1. Add the four missing source files

They exist in your project folder but were not in the upload used to build this archive:

    src/sugar_extraction_model.py
    src/sugar_extraction_env.py
    src/morl_score.py
    src/baseline_controllers_mv.py

## 2. Copy the raw outputs

Into `results/tables/`, the files listed in `results/tables/README.md`.
Into `results/runs_multiseed/`, the per-run directories, each keeping at minimum:

    selection_protocol.json      which checkpoint was installed, and under which gate
    validation_protocol.csv      the evidence for that choice
    test_protocol.csv            the per-seed numbers that reached the paper

Checkpoint weights (`ckpt_*.pt`) are large. Either commit only the installed one per run,
or attach all of them to a GitHub Release or Zenodo deposit and link it from the README.

## 3. Preserve the superseded runs

Copy `runs_multiseed_BUGGED_keep/` into `results/buggy_training_signal/`. Section 5.8 and
`docs/ERRATA.md` both state that these are kept; if they are absent the paper contradicts
its own repository.

## 4. Create the repository and tag the commit

    git init && git add -A
    git commit -m "Code and data for the IJIES submission"
    git tag -a paper-v1 -m "Results as submitted"
    git remote add origin https://github.com/<user>/<repo>.git
    git push -u origin main --tags

Then read the 40-character SHA:

    git rev-parse paper-v1

## 5. Fill in the four placeholders

In `README.md`, `CITATION.cff` and `docs/PAPER_SECTION.md`, replace:

    <FILL IN: 40-character SHA>     the output of the command above
    https://github.com/<FILL IN>    the repository URL
    <FILL IN: DOI or release URL>   the weights archive, if you deposit one

## 6. Verify from a clean clone

    git clone https://github.com/<user>/<repo>.git /tmp/check && cd /tmp/check
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scripts/reeval_protocol.py --stage matched

The numbers it prints must match Table 5. If they do not, the archive is incomplete and a
reviewer will find that out before you do.
