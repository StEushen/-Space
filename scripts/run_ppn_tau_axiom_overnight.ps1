param(
    [string]$WorkspaceRoot = "D:\PPn_submit",
    [int[]]$PredLens = @(96, 192, 336, 720),
    [int[]]$Seeds = @(42, 52, 62, 72, 82),
    [double[]]$Ratios = @(0.1, 0.05),
    [int]$Epochs = 8,
    [int]$Patience = 3,
    [double]$LambdaTau = 0.05,
    [double]$LambdaTauSmooth = 0.01,
    [double]$LambdaTauRecon = 0.05,
    [double]$LambdaTauFlat = 0.01,
    [double]$LambdaTauMono = 0.01,
    [double]$ResidualScale = 0.1,
    [string]$TagPrefix = "tau_axiom_overnight",
    [int]$Resume = 1
)

$ErrorActionPreference = "Stop"
Set-Location $WorkspaceRoot

$startTime = Get-Date
$masterLog = "logs/${TagPrefix}_master.log"
if ($Resume -eq 1 -and (Test-Path $masterLog)) {
    "[$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))] Resume overnight run" | Add-Content $masterLog -Encoding UTF8
}
else {
    "[$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))] Start overnight run" | Set-Content $masterLog -Encoding UTF8
}

foreach ($predLen in $PredLens) {
    $tag = "${TagPrefix}_h${predLen}_e${Epochs}_s$($Seeds.Count)_r$($Ratios.Count)"
    $summaryCsv = "logs/ppn_horizon_residual_${tag}_summary.csv"

    if ($Resume -eq 1 -and (Test-Path $summaryCsv)) {
        $lineCount = (Get-Content $summaryCsv | Measure-Object -Line).Lines
        if ($lineCount -gt 1) {
            $skip = "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] SKIP pred_len=$predLen (summary exists: $summaryCsv)"
            $skip | Add-Content $masterLog -Encoding UTF8
            Write-Host $skip -ForegroundColor Yellow
            continue
        }
    }

    $line = "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] RUN pred_len=$predLen tag=$tag"
    $line | Add-Content $masterLog -Encoding UTF8
    Write-Host $line -ForegroundColor Cyan

    powershell -ExecutionPolicy Bypass -Command "& { .\scripts\run_ppn_horizon_residual_fewshot.ps1 `
        -Epochs $Epochs `
        -Patience $Patience `
        -PredLen $predLen `
        -Seeds @($($Seeds -join ',')) `
        -Ratios @($($Ratios -join ',')) `
        -LambdaTau $LambdaTau `
        -LambdaTauSmooth $LambdaTauSmooth `
        -LambdaTauRecon $LambdaTauRecon `
        -LambdaTauFlat $LambdaTauFlat `
        -LambdaTauMono $LambdaTauMono `
        -ResidualScale $ResidualScale `
        -UseTauSpacePredictor 1 `
        -TauGlobalScale 1.0 `
        -TauCrossAdjustScale 0.2 `
        -UseTauCrossAdjustGate 1 `
        -DisableAccelTauLossInAxiomMode 1 `
        -IncludeGatedResidual 0 `
        -IncludeWarmupResidual 0 `
        -IncludeTauFieldVariant 0 `
        -Tag $tag }"

    if ($LASTEXITCODE -ne 0) {
        $err = "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] FAILED pred_len=$predLen exit=$LASTEXITCODE"
        $err | Add-Content $masterLog -Encoding UTF8
        Write-Host $err -ForegroundColor Red
    }
    else {
        $ok = "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] DONE pred_len=$predLen"
        $ok | Add-Content $masterLog -Encoding UTF8
        Write-Host $ok -ForegroundColor Green
    }
}

$endTime = Get-Date
$elapsed = $endTime - $startTime
"[$($endTime.ToString('yyyy-MM-dd HH:mm:ss'))] End overnight run" | Add-Content $masterLog -Encoding UTF8
"Elapsed: $($elapsed.ToString())" | Add-Content $masterLog -Encoding UTF8

Write-Host "Overnight run finished. Elapsed: $($elapsed.ToString())" -ForegroundColor Green
Write-Host "Master log: $masterLog" -ForegroundColor Green
