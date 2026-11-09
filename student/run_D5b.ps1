# D5b on the tower (PC2): privileged data regenerated on CPU, distillation on GPU with blended labels,
# then agreement and evaluation. Trains no RL and never touches submissions/. Never R1 in the data.
#   powershell -ExecutionPolicy Bypass -File student\run_D5b.ps1        (run from the challenge root)
$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:TANKS_PC = "PC2"
$PY = ".venv-gpu\Scripts\python.exe"
$DATA = "models/checkpoints_D5b-distill/data.npz"
$OPP = '{"model:models/league/A-g6_4000k.zip":0.075,"my:stoch:models/league/A-g6_4000k.zip":0.075,"model:models/league/A-g5_24000k.zip":0.05,"model:models/league/B-g4_3000k.zip":0.075,"model:models/league/C-g4_6000k.zip":0.075,"baseline:hard":0.075,"baseline:medium":0.05,"basic":0.025,"aggressive":0.025,"defensive":0.025,"my:rusher":0.05,"my:bushcamper":0.2,"my:hitandhide":0.2}'

function Step($name, $argv) {
    Write-Output "=== $name $(Get-Date -Format HH:mm)"
    & $PY @argv
    if ($LASTEXITCODE -ne 0) { throw "$name failed ($LASTEXITCODE)" }
}

Step "gen" @("student/distill.py", "gen", "--privileged", "--teacher", "models/league/A-g17_8000k.zip",
    "--steps", "1500000", "--n-envs", "6", "--maps", "mixed2", "--scenario-prob", "0.3", "--opponents", $OPP,
    "--seed", "6", "--data", $DATA)
Step "train" @("student/distill.py", "train", "--data", $DATA, "--init", "models/D5-distill.zip", "--device", "cuda",
    "--lr", "2e-4", "--ce-coef", "0.5", "--priv-mix", "0.5", "--select", "acc", "--max-minutes", "90",
    "--epochs", "500", "--threads", "4", "--out", "models/D5b-distill.zip")
Step "agree" @("student/distill.py", "agree", "--data", $DATA, "--threads", "4",
    "--models", "models/D2b-distill.zip", "models/D5-distill.zip", "models/D5b-distill.zip")
# R1 is only the judge (read in place, never in any data or pool)
$R1 = @("models/eval_pc4/R1.zip", "models/reserved/R1.zip") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $R1) { throw "R1.zip not found in models/eval_pc4/ or models/reserved/" }
Step "eval" @("student/eval_final.py", "--judge", $R1, "--procs", "6", "--out", "results/eval_D5b.md", "--models",
    "A-g17_8000k=models/league/A-g17_8000k.zip", "D5-distill=models/D5-distill.zip", "D5b-distill=models/D5b-distill.zip",
    "--bots", "my:bushcamper", "my:hitandhide", "--bot-seeds", "3")
Write-Output "=== done $(Get-Date -Format HH:mm)"
