# Item 8: what a reviewer does. Clone into an empty folder, install, and regenerate Table 5.
$ErrorActionPreference = "Stop"
$tmp = Join-Path $env:TEMP ("clean_" + [guid]::NewGuid().ToString("N").Substring(0,6))
git clone https://github.com/Mohanadhilal/setup-decides-result.git $tmp
Set-Location $tmp
Write-Host "`nremote tags:"; git ls-remote --tags origin
git checkout v1.0.0
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip | Out-Null
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install pyyaml | Out-Null
.\.venv\Scripts\python tools\check_repo.py --remote
$env:PYTHONPATH  = "$tmp\src;$tmp\scripts"
$env:MORL_OUT_DIR = "results\runs_multiseed"
.\.venv\Scripts\python scripts\reeval_protocol.py --stage headline
Write-Host "`nCompare the table above with Table 5 of the manuscript. Clean clone at: $tmp" -ForegroundColor Green
