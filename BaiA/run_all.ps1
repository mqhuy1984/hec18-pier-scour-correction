# Reproduce every result of the manuscript, in order.
# Usage (from the repository root):
#   powershell -ExecutionPolicy Bypass -File BaiA/run_all.ps1            # full run: 10 repetitions, 60 Optuna trials (about 12 h on 15 cores)
#   powershell -ExecutionPolicy Bypass -File BaiA/run_all.ps1 -Quick     # smoke test: 1 repetition, 5 trials (minutes)
#   powershell -ExecutionPolicy Bypass -File BaiA/run_all.ps1 -Jobs 8
#   -Quick overwrites BaiA/outputs; it stops if full results exist unless -Overwrite is given.
param([switch]$Quick, [switch]$Overwrite, [int]$Jobs = 15)

$ErrorActionPreference = "Stop"
if ($Quick -and -not $Overwrite -and (Test-Path BaiA/outputs/week3_metrics_summary.csv)) {
    throw "BaiA/outputs already holds results; -Quick would overwrite them. Add -Overwrite to proceed."
}
$env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$reps = 10; $trials = 60
if ($Quick) { $reps = 1; $trials = 5 }

function Invoke-Step([string]$name, [string[]]$cmd) {
    Write-Host "=== $name ===" -ForegroundColor Cyan
    $t = Get-Date
    & python @cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("    done in {0:N1} min" -f ((Get-Date) - $t).TotalMinutes)
}

New-Item -ItemType Directory -Force BaiA/outputs, BaiA/figures, BaiA/figures_en | Out-Null
Invoke-Step "0. Clean PSDb-2014 and add quality flags" @("-I", "PSDb_clean/clean_psdb.py", "ds845_psdb-2014_version1.0.xlsx", "PSDb_clean")
Invoke-Step "1. Baseline equations, M0-M3"            @("BaiA/week1_baselines.py")
Invoke-Step "2. HEC-18 bias vs covariates (RQ1)"      @("BaiA/week2_analysis.py")
Invoke-Step "3. ML corrections, bridge vs random CV"  @("BaiA/week3_xgboost.py", "--reps", "$reps", "--trials", "$trials", "--jobs", "$Jobs")
Invoke-Step "4. Conformal bounds and extrapolation"   @("BaiA/week4_conformal_extrap.py", "--reps", "$reps", "--trials", "$trials", "--jobs", "$Jobs")
Invoke-Step "5. Closed form, SHAP, sensitivity, lab-to-field" @("BaiA/week5_formula_sensitivity.py")
Invoke-Step "S2. Choice of terms in the closed form" @("BaiA/formula_variant_check.py")
Invoke-Step "S4. CIRIA C742 baseline"                @("BaiA/ciria_check.py")
Invoke-Step "S7-S8. Refined closed form, timing, k multipliers" @("BaiA/revision_analyses.py")
Invoke-Step "S8. Final Eq. 2: bed class, GPR level, k multipliers" @("BaiA/revision2_analyses.py")
Invoke-Step "S8. Refill factor and maximum-scour multiplier (Eq. 3)" @("BaiA/revision3_analyses.py")
Invoke-Step "Figures (English, 300 dpi)"             @("BaiA/make_figures_en.py")
Invoke-Step "Supplement tables"                      @("BaiA/make_supplement.py")
Invoke-Step "Supplement as LaTeX (compile to Supplemental_Materials.pdf)" @("BaiA/make_supplement_tex.py")
Write-Host "All steps finished. Outputs: BaiA/outputs, BaiA/figures_en, BaiA/Supplement_BaiA.md" -ForegroundColor Green
