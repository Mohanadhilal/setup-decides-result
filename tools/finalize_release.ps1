# Items 1, 4, 5 and 6 of the supervisor's list. Run from the repository root,
# with the virtual environment that PRODUCED THE RESULTS activated.
# Set $Project to the folder that holds the scaling experiment outputs.
param([string]$Project = "E:\SugerExtraction")
$ErrorActionPreference = "Stop"

# ---- 1. requirements.txt from the environment that produced the results ----
$pkgs = "numpy|scipy|torch|gymnasium|matplotlib"
$lines = pip freeze | Select-String -Pattern "^($pkgs)==" | ForEach-Object {
    # torch builds carry a local suffix (+cpu, +cu121) that PyPI cannot resolve;
    # pin the release, and record the build in a comment
    if ($_.Line -match '^(torch==[^+]+)\+(.+)$') { "# built with torch local version +$($Matches[2])"; $Matches[1] }
    else { $_.Line }
}
# ASCII: Windows PowerShell 5 writes UTF-8 with a byte-order mark, which breaks the first line
$lines | Set-Content -Encoding ascii requirements.txt
Write-Host "requirements.txt:"; Get-Content requirements.txt

# ---- 4. internal notes out of the public repository ----
Remove-Item docs\UPLOAD_CHECKLIST.md, docs\PAPER_SECTION.md, STEPS_AR.md -ErrorAction SilentlyContinue
Remove-Item experiment -Recurse -ErrorAction SilentlyContinue   # copied into src\ and scripts\ below

# ---- 5. the non-oracle scaling experiment: code and results ----
New-Item -ItemType Directory -Force results\scaling | Out-Null
Copy-Item "$Project\scaling.py"              src\      -Force
Copy-Item "$Project\sugar_extraction_env.py" src\      -Force
Copy-Item "$Project\morl_score.py"           src\      -Force
Copy-Item "$Project\compare_scaling.py"      scripts\  -Force
Copy-Item "$Project\train_multiseed.py"      scripts\  -Force
Copy-Item "$Project\run_nonoracle.ps1"       scripts\  -Force
Copy-Item "$Project\scaling_training.json"   results\scaling\ -Force
Copy-Item "$Project\scaling_comparison.csv"  results\scaling\ -Force
Copy-Item "$Project\scaling_verdict.txt"     results\scaling\ -Force
robocopy "$Project\runs_multiseed_training" results\runs_multiseed_training /E /XF ckpt_*.pt | Out-Null
# runs affected by the training-signal error of Section 5.8 (the paper and the letter say they are kept)
robocopy "$Project\runs_multiseed_BUGGED_keep" results\buggy_training_signal /E /XF ckpt_*.pt | Out-Null

# ---- gate: nothing is committed while a placeholder or gap remains ----
python tools\check_repo.py --pre-commit
if ($LASTEXITCODE -ne 0) { Write-Host "`nFix the problems above, then run this script again." -ForegroundColor Red; exit 1 }

# ---- 6. commit, tag, push ----
git add -A
git commit -m "Complete configurations and dependencies; add the non-oracle scaling experiment"
git tag -a v1.0.0 -m "Results as reported in the paper"
git push origin HEAD
git push origin v1.0.0
python tools\check_repo.py --remote
$h = git rev-list -n 1 v1.0.0
Write-Host "`nSend this commit to update the manuscript (item 7):  $h" -ForegroundColor Green
