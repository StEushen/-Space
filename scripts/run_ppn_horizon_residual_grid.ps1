param(
    [string]$EnvName = "py39env",
    [int]$PredLen = 96,
    [int]$Epochs = 3,
    [int]$Patience = 2,
    [int[]]$Seeds = @(42, 52, 62),
    [double[]]$Ratios = @(0.1, 0.05),
    [double[]]$LambdaList = @(0.03, 0.05, 0.08),
    [double[]]$ResidualScaleList = @(0.05, 0.1, 0.2),
    [string]$TagPrefix = "grid"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Path logs -Force | Out-Null

$masterCsv = "logs/ppn_horizon_residual_${TagPrefix}_master.csv"
$rankCsv = "logs/ppn_horizon_residual_${TagPrefix}_rank.csv"

"tag,lambda_tau,residual_scale,ratio,variant,n,mse_mean,mse_std,mae_mean,mae_std" | Set-Content $masterCsv -Encoding UTF8

foreach ($lambdaTau in $LambdaList) {
    foreach ($resScale in $ResidualScaleList) {
        $lTag = ("{0}" -f $lambdaTau).Replace(".", "p")
        $rTag = ("{0}" -f $resScale).Replace(".", "p")
        $tag = "${TagPrefix}_l${lTag}_r${rTag}"

        Write-Host "[GRID-RUN] tag=$tag lambda=$lambdaTau residual=$resScale" -ForegroundColor Cyan

        $childArgs = @{
            EnvName = $EnvName
            PredLen = $PredLen
            Epochs = $Epochs
            Patience = $Patience
            Seeds = $Seeds
            Ratios = $Ratios
            LambdaTau = $lambdaTau
            ResidualScale = $resScale
            Tag = $tag
        }

        & .\scripts\run_ppn_horizon_residual_fewshot.ps1 @childArgs

        $summaryPath = "logs/ppn_horizon_residual_${tag}_summary.csv"
        if (-not (Test-Path $summaryPath)) {
            Write-Host "[WARN] missing summary: $summaryPath" -ForegroundColor Yellow
            continue
        }

        $rows = Import-Csv $summaryPath
        foreach ($row in $rows) {
            "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f `
                $tag, $lambdaTau, $resScale, $row.ratio, $row.variant, $row.n, $row.mse_mean, $row.mse_std, $row.mae_mean, $row.mae_std | `
                Add-Content $masterCsv -Encoding UTF8
        }
    }
}

$masterRows = Import-Csv $masterCsv
$hresRows = $masterRows | Where-Object { $_.variant -eq "horizon_residual" }

$rankOut = @()
foreach ($group in ($hresRows | Group-Object ratio)) {
    $idx = 1
    foreach ($r in ($group.Group | Sort-Object {[double]$_.mse_mean}, {[double]$_.mae_mean}, {[double]$_.mse_std})) {
        $rankOut += [PSCustomObject]@{
            rank = $idx
            ratio = $r.ratio
            tag = $r.tag
            lambda_tau = $r.lambda_tau
            residual_scale = $r.residual_scale
            mse_mean = $r.mse_mean
            mse_std = $r.mse_std
            mae_mean = $r.mae_mean
            mae_std = $r.mae_std
        }
        $idx += 1
    }
}

$rankOut | Export-Csv -Path $rankCsv -NoTypeInformation -Encoding UTF8

Write-Host "Grid scan done." -ForegroundColor Green
Write-Host "Master: $masterCsv"
Write-Host "Rank:   $rankCsv"