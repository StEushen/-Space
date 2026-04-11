param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int]$Epochs = 3,
    [int]$Patience = 2,
    [int[]]$Seeds = @(42, 52, 62),
    [double[]]$Ratios = @(0.1, 0.05),
    [double]$LambdaTau = 0.05,
    [double]$LambdaTauSmooth = 0.0,
    [double]$LambdaTauRecon = 0.05,
    [double]$LambdaTauFlat = 0.01,
    [double]$LambdaTauMono = 0.01,
    [int]$UseTauSpacePredictor = 1,
    [double]$TauGlobalScale = 1.0,
    [double]$TauCrossAdjustScale = 0.2,
    [int]$UseTauCrossAdjustGate = 1,
    [int]$DisableAccelTauLossInAxiomMode = 1,
    [int]$UseTauField = 0,
    [double]$TauFieldLocalScale = 0.5,
    [double]$ResidualScale = 0.1,
    [int]$IncludeTauFieldVariant = 0,
    [int]$IncludeGatedResidual = 1,
    [int]$IncludeWarmupResidual = 1,
    [int]$WarmupEpochs = 2,
    [string]$Tag = "hres"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/ppn_horizon_residual_${Tag}.csv"
$summaryCsv = "logs/ppn_horizon_residual_${Tag}_summary.csv"
$rankCsv = "logs/ppn_horizon_residual_${Tag}_rank.csv"

"variant,ratio,seed,mse,mae" | Set-Content $runCsv -Encoding UTF8

$variants = @(
    @{ Name = "base"; HorizonResidual = 0; HorizonResidualGate = 0; HorizonResidualWarmup = 0; UseTauField = 0 },
    @{ Name = "horizon_residual"; HorizonResidual = 1; HorizonResidualGate = 0; HorizonResidualWarmup = 0; UseTauField = 0 }
)

if ($IncludeGatedResidual -eq 1) {
    $variants += @{ Name = "horizon_residual_gate"; HorizonResidual = 1; HorizonResidualGate = 1; HorizonResidualWarmup = 0; UseTauField = 0 }
}

if ($IncludeWarmupResidual -eq 1) {
    $variants += @{ Name = "horizon_residual_warmup"; HorizonResidual = 1; HorizonResidualGate = 0; HorizonResidualWarmup = 1; UseTauField = 0 }
}

if ($IncludeTauFieldVariant -eq 1) {
    $variants += @{ Name = "horizon_residual_taufield"; HorizonResidual = 1; HorizonResidualGate = 0; HorizonResidualWarmup = 0; UseTauField = 1 }
}

foreach ($seed in $Seeds) {
    foreach ($v in $variants) {
        $variantName = $v["Name"]
        $useHorizonResidual = $v["HorizonResidual"]
        $useHorizonResidualGate = $v["HorizonResidualGate"]
        $useHorizonResidualWarmup = $v["HorizonResidualWarmup"]
        $variantUseTauField = $v["UseTauField"]

        foreach ($ratio in $Ratios) {
            $ratioTag = ("{0}" -f $ratio).Replace(".", "p")
            $modelId = "ppn_${variantName}_fs_${ratioTag}_h${PredLen}_s${seed}"

            Write-Host "[RUN] variant=$variantName ratio=$ratio seed=$seed model_id=$modelId" -ForegroundColor Cyan

            conda run -n $EnvName python run.py `
                --task_name long_term_forecast `
                --is_training 1 `
                --model_id $modelId `
                --model PPN `
                --data $Data `
                --root_path ./dataset/ETT-small/ `
                --data_path $DataPath `
                --features M `
                --target OT `
                --freq h `
                --seq_len 96 `
                --label_len 48 `
                --pred_len $PredLen `
                --enc_in 7 `
                --dec_in 7 `
                --c_out 7 `
                --e_layers 1 `
                --d_layers 1 `
                --factor 3 `
                --d_model 128 `
                --d_ff 128 `
                --train_epochs $Epochs `
                --patience $Patience `
                --batch_size 32 `
                --learning_rate 0.0003 `
                --lradj cosine `
                --num_workers 2 `
                --itr 1 `
                --des horizon_residual `
                --checkpoints C:/tmp/ppn_latest_ckpt `
                --seed $seed `
                --ppn_use_tau_loss 1 `
                --ppn_lambda_tau $LambdaTau `
                --ppn_use_tau_field $variantUseTauField `
                --ppn_tau_field_local_scale $TauFieldLocalScale `
                --ppn_lambda_tau_smooth $LambdaTauSmooth `
                --ppn_lambda_tau_recon $LambdaTauRecon `
                --ppn_lambda_tau_flat $LambdaTauFlat `
                --ppn_lambda_tau_mono $LambdaTauMono `
                --ppn_use_tau_space_predictor $UseTauSpacePredictor `
                --ppn_tau_global_scale $TauGlobalScale `
                --ppn_tau_cross_adjust_scale $TauCrossAdjustScale `
                --ppn_use_tau_cross_adjust_gate $UseTauCrossAdjustGate `
                --ppn_disable_accel_tau_loss_in_axiom_mode $DisableAccelTauLossInAxiomMode `
                --ppn_use_horizon_residual $useHorizonResidual `
                --ppn_use_horizon_residual_gate $useHorizonResidualGate `
                --ppn_horizon_residual_scale $ResidualScale `
                --ppn_use_horizon_residual_warmup $useHorizonResidualWarmup `
                --ppn_horizon_residual_warmup_epochs $WarmupEpochs `
                --train_subset_ratio $ratio `
                --subset_seed $seed `
                --use_amp `
                --use_gpu `
                --gpu_type cuda `
                --gpu 0

            $runExitCode = $LASTEXITCODE
            if ($runExitCode -ne 0) {
                Write-Host "[WARN] Training command failed (exit=$runExitCode) for model_id=$modelId" -ForegroundColor Yellow
                continue
            }

            $settingPattern = "long_term_forecast_${modelId}_PPN"
            $tail = Get-Content result_long_term_forecast.txt -Tail 260
            $start = -1
            for ($i = $tail.Count - 1; $i -ge 0; $i--) {
                if ($tail[$i] -like "*$settingPattern*") { $start = $i; break }
            }

            if ($start -ge 0 -and $start + 1 -lt $tail.Count) {
                $metricLine = $tail[$start + 1]
                $mse = [regex]::Match($metricLine, "mse:([0-9\.Ee\-]+)").Groups[1].Value
                $mae = [regex]::Match($metricLine, "mae:([0-9\.Ee\-]+)").Groups[1].Value
                if ($mse -and $mae) {
                    "{0},{1},{2},{3},{4}" -f $variantName, $ratio, $seed, $mse, $mae | Add-Content $runCsv -Encoding UTF8
                    Write-Host "[DONE] variant=$variantName ratio=$ratio seed=$seed mse=$mse mae=$mae" -ForegroundColor Green
                }
                else {
                    Write-Host "[WARN] Failed to parse metric line: $metricLine" -ForegroundColor Yellow
                }
            }
            else {
                Write-Host "[WARN] Could not locate result block for model_id=$modelId" -ForegroundColor Yellow
            }
        }
    }
}

$rows = Import-Csv $runCsv
$groups = $rows | Group-Object variant, ratio

"variant,ratio,n,mse_mean,mse_std,mae_mean,mae_std" | Set-Content $summaryCsv -Encoding UTF8
$summaryObjects = @()

foreach ($g in $groups) {
    $mses = $g.Group | ForEach-Object { [double]$_.mse }
    $maes = $g.Group | ForEach-Object { [double]$_.mae }
    $n = $mses.Count

    $mseMean = ($mses | Measure-Object -Average).Average
    $maeMean = ($maes | Measure-Object -Average).Average

    if ($n -gt 1) {
        $mseStd = [math]::Sqrt((($mses | ForEach-Object { ($_ - $mseMean) * ($_ - $mseMean) } | Measure-Object -Sum).Sum) / ($n - 1))
        $maeStd = [math]::Sqrt((($maes | ForEach-Object { ($_ - $maeMean) * ($_ - $maeMean) } | Measure-Object -Sum).Sum) / ($n - 1))
    }
    else {
        $mseStd = 0.0
        $maeStd = 0.0
    }

    $variant = $g.Group[0].variant
    $ratio = [double]$g.Group[0].ratio

    "{0},{1},{2},{3},{4},{5},{6}" -f $variant, $ratio, $n, $mseMean, $mseStd, $maeMean, $maeStd | Add-Content $summaryCsv -Encoding UTF8

    $summaryObjects += [PSCustomObject]@{
        variant = $variant
        ratio = $ratio
        n = $n
        mse_mean = $mseMean
        mse_std = $mseStd
        mae_mean = $maeMean
        mae_std = $maeStd
    }
}

$rankFinal = @()
foreach ($ratioGroup in ($summaryObjects | Group-Object ratio)) {
    $idx = 1
    foreach ($r in ($ratioGroup.Group | Sort-Object mse_mean, mae_mean, mse_std)) {
        $rankFinal += [PSCustomObject]@{
            rank = $idx
            variant = $r.variant
            ratio = $r.ratio
            n = $r.n
            mse_mean = $r.mse_mean
            mse_std = $r.mse_std
            mae_mean = $r.mae_mean
            mae_std = $r.mae_std
        }
        $idx += 1
    }
}

$rankFinal | Export-Csv -Path $rankCsv -NoTypeInformation -Encoding UTF8

Write-Host "All done." -ForegroundColor Green
Write-Host "Run-level:     $runCsv"
Write-Host "Group summary: $summaryCsv"
Write-Host "Ranking:       $rankCsv"