# Non-oracle scaling experiment (Reviewer 2, comment 6).  Run from the project folder.
# 1) derive the scaling from warm-up data on the training seeds only
Remove-Item Env:MORL_SCALING -ErrorAction SilentlyContinue
python scaling.py --derive
# 2) train and select under that scaling, in a separate folder
$env:MORL_SCALING = "training"
$env:MORL_OUT_DIR = "runs_multiseed_training"
python train_multiseed.py --algos SAC --prefs 0 2 3 --seeds 0 1 2 3 4 --workers 5
python reeval_protocol.py --stage select --algos SAC
# 3) compare with the oracle runs, in physical units
Remove-Item Env:MORL_SCALING; Remove-Item Env:MORL_OUT_DIR
python compare_scaling.py
