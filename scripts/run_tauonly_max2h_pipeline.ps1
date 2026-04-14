param(
    [string]$EnvName = "py39env",
    [string]$Tag = "tauonly_max2h_v1",
    [int]$PretrainPredLen = 24,
    [int]$PretrainEpochs = 5,
    [double]$PretrainRatio = 1.0,
    [int]$FinetunePredLen = 96,
    [int]$FinetuneEpochs = 3,
    [double]$FinetuneRatio = 0.1,
    [int]$Patience = 2,
    [int]$BatchSize = 16
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null
New-Item -ItemType Directory -Path C:/tmp/ppn_max2h_ckpt -Force | Out-Null

$runsCsv = "logs/${Tag}_runs.csv"
$rankCsv = "logs/${Tag}_rank.csv"
$summaryTxt = "logs/${Tag}_summary.txt"
$ckptCsv = "logs/${Tag}_checkpoints.csv"

"stage,seed,pred_len,ratio,mse,mae,model_id,checkpoint_path,checkpoint_exists,loaded_from" | Set-Content $runsCsv -Encoding UTF8
"stage,seed,model_id,checkpoint_path,checkpoint_exists" | Set-Content $ckptCsv -Encoding UTF8

function Get-LatestCheckpointDirByModelId {
    param(
        [string]$Root,
        [string]$ModelId,
        [string]$ModelName
    )

    if (-not (Test-Path $Root)) {
        return $null
    }

    $needle = "*${ModelId}_${ModelName}_*"
    $dir = Get-ChildItem -Path $Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like $needle } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($dir) {
        return $dir.FullName
    }

    return $null
}

function Parse-LatestMetricsForModelId {
    param(
        [string]$ModelId,
        [string]$ModelName
    )

    if (-not (Test-Path ".\\result_long_term_forecast.txt")) {
        return $null
    }

    $tail = Get-Content .\result_long_term_forecast.txt -Tail 1200
    $pattern = "long_term_forecast_${ModelId}_${ModelName}"
    $idx = -1
    for ($i = $tail.Count - 1; $i -ge 0; $i--) {
        if ($tail[$i] -like "*$pattern*") {
            $idx = $i
            break
        }
    }

    if ($idx -lt 0 -or $idx + 1 -ge $tail.Count) {
        return $null
    }

    $metricLine = $tail[$idx + 1]
    $mse = [regex]::Match($metricLine, "mse:([0-9\.Ee\-]+)").Groups[1].Value
    $mae = [regex]::Match($metricLine, "mae:([0-9\.Ee\-]+)").Groups[1].Value

    if (-not $mse -or -not $mae) {
        return $null
    }

    return @{ MSE = $mse; MAE = $mae }
}

$baseArgs = @(
    "--task_name", "long_term_forecast",
    "--model", "TauOnly",
    "--data", "ETTh1",
    "--root_path", "./dataset/ETT-small/",
    "--data_path", "ETTh1.csv",
    "--features", "M",
    "--target", "OT",
    "--freq", "h",
    "--seq_len", "96",
    "--label_len", "48",
    "--enc_in", "7",
    "--dec_in", "7",
    "--c_out", "7",
    "--e_layers", "1",
    "--d_layers", "1",
    "--factor", "3",
    "--d_model", "128",
    "--d_ff", "128",
    "--batch_size", "$BatchSize",
    "--learning_rate", "0.0003",
    "--lradj", "cosine",
    "--num_workers", "2",
    "--itr", "1",
    "--des", $Tag,
    "--checkpoints", "C:/tmp/ppn_max2h_ckpt",
    "--ppn_use_transformer", "1",
    "--ppn_max_tau_integration_steps", "8",
    "--ppn_tau_use_residual_refine", "1",
    "--ppn_use_tau_phase_schedule", "1",
    "--ppn_tau_phase1_ratio", "0.6",
    "--ppn_tau_phase2_ratio", "0.3",
    "--ppn_tau_phase1_aux_scale", "0.0",
    "--ppn_tau_phase2_aux_scale", "0.4",
    "--ppn_tau_phase3_aux_scale", "1.0",
    "--ppn_lambda_tau_smooth", "0.02",
    "--ppn_lambda_tau_mono", "0.01",
    "--ppn_lambda_tau_recon", "0.05",
    "--ppn_lambda_tau_contrast", "0.1",
    "--ppn_lambda_proj_cycle", "0.05",
    "--ppn_lambda_tau_flat", "0.01",
    "--tauonly_anchor_delay_epochs", "1",
    "--tauonly_anchor_scale", "1.0",
    "--use_amp",
    "--use_gpu",
    "--gpu_type", "cuda",
    "--gpu", "0"
)

# Stage 1: short-horizon pretrain
$preSeed = 42
$preModelId = "${Tag}_pre_p${PretrainPredLen}_s${preSeed}"
Write-Host "[STAGE1] pretrain model_id=$preModelId pred_len=$PretrainPredLen epochs=$PretrainEpochs ratio=$PretrainRatio" -ForegroundColor Cyan

$preCmd = @(
    "conda", "run", "-n", $EnvName, "python", "run.py",
    "--is_training", "1",
    "--model_id", $preModelId,
    "--pred_len", "$PretrainPredLen",
    "--train_epochs", "$PretrainEpochs",
    "--patience", "$Patience",
    "--seed", "$preSeed",
    "--train_subset_ratio", "$PretrainRatio",
    "--subset_seed", "$preSeed"
) + $baseArgs

& $preCmd[0] $preCmd[1..($preCmd.Length - 1)]
if ($LASTEXITCODE -ne 0) {
    throw "Stage1 pretrain failed with exit code $LASTEXITCODE"
}

$preMetrics = Parse-LatestMetricsForModelId -ModelId $preModelId -ModelName "TauOnly"
if (-not $preMetrics) {
    throw "Failed to parse Stage1 metrics for $preModelId"
}

$preCkptDir = Get-LatestCheckpointDirByModelId -Root "C:/tmp/ppn_max2h_ckpt" -ModelId $preModelId -ModelName "TauOnly"
if (-not $preCkptDir) {
    throw "Failed to locate Stage1 checkpoint directory for $preModelId"
}
$preCkptPath = Join-Path $preCkptDir "checkpoint.pth"
$preCkptExists = if (Test-Path $preCkptPath) { "1" } else { "0" }
if ($preCkptExists -ne "1") {
    throw "Stage1 checkpoint not found: $preCkptPath"
}

"{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f "pretrain", $preSeed, $PretrainPredLen, $PretrainRatio, $preMetrics.MSE, $preMetrics.MAE, $preModelId, $preCkptPath, $preCkptExists, "" | Add-Content $runsCsv -Encoding UTF8
"{0},{1},{2},{3},{4}" -f "pretrain", $preSeed, $preModelId, $preCkptPath, $preCkptExists | Add-Content $ckptCsv -Encoding UTF8
Write-Host "[STAGE1 DONE] mse=$($preMetrics.MSE) mae=$($preMetrics.MAE) ckpt=$preCkptPath" -ForegroundColor Green

# Stage 2: long-horizon finetune for multi-seed
$ftSeeds = @(42, 52, 62)
foreach ($seed in $ftSeeds) {
    $ratioTag = ("{0}" -f $FinetuneRatio).Replace('.', 'p')
    $ftModelId = "${Tag}_ft_p${FinetunePredLen}_r${ratioTag}_s${seed}"

    Write-Host "[STAGE2] finetune seed=$seed model_id=$ftModelId pred_len=$FinetunePredLen epochs=$FinetuneEpochs ratio=$FinetuneRatio" -ForegroundColor Cyan

    $ftCmd = @(
        "conda", "run", "-n", $EnvName, "python", "run.py",
        "--is_training", "1",
        "--model_id", $ftModelId,
        "--pred_len", "$FinetunePredLen",
        "--train_epochs", "$FinetuneEpochs",
        "--patience", "$Patience",
        "--seed", "$seed",
        "--train_subset_ratio", "$FinetuneRatio",
        "--subset_seed", "$seed",
        "--load_checkpoint", $preCkptPath,
        "--load_checkpoint_strict", "0"
    ) + $baseArgs

    & $ftCmd[0] $ftCmd[1..($ftCmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Stage2 failed for seed=$seed" -ForegroundColor Yellow
        continue
    }

    $ftMetrics = Parse-LatestMetricsForModelId -ModelId $ftModelId -ModelName "TauOnly"
    if (-not $ftMetrics) {
        Write-Host "[WARN] Failed parsing Stage2 metrics for seed=$seed" -ForegroundColor Yellow
        continue
    }

    $ftCkptDir = Get-LatestCheckpointDirByModelId -Root "C:/tmp/ppn_max2h_ckpt" -ModelId $ftModelId -ModelName "TauOnly"
    $ftCkptPath = ""
    $ftCkptExists = "0"
    if ($ftCkptDir) {
        $ftCkptPath = Join-Path $ftCkptDir "checkpoint.pth"
        if (Test-Path $ftCkptPath) {
            $ftCkptExists = "1"
        }
    }

    "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f "finetune", $seed, $FinetunePredLen, $FinetuneRatio, $ftMetrics.MSE, $ftMetrics.MAE, $ftModelId, $ftCkptPath, $ftCkptExists, $preCkptPath | Add-Content $runsCsv -Encoding UTF8
    "{0},{1},{2},{3},{4}" -f "finetune", $seed, $ftModelId, $ftCkptPath, $ftCkptExists | Add-Content $ckptCsv -Encoding UTF8
    Write-Host "[STAGE2 DONE] seed=$seed mse=$($ftMetrics.MSE) mae=$($ftMetrics.MAE) ckpt=$ftCkptExists" -ForegroundColor Green
}

# Aggregate finetune metrics only
$ftRows = Import-Csv $runsCsv | Where-Object { $_.stage -eq "finetune" -and $_.mse -and $_.mae }
if ($ftRows.Count -gt 0) {
    $ftRows | Sort-Object {[double]$_.mse}, {[double]$_.mae} | Export-Csv -Path $rankCsv -NoTypeInformation -Encoding UTF8

    $mseValues = $ftRows | ForEach-Object { [double]$_.mse }
    $maeValues = $ftRows | ForEach-Object { [double]$_.mae }
    $n = $mseValues.Count

    $mseMean = ($mseValues | Measure-Object -Average).Average
    $maeMean = ($maeValues | Measure-Object -Average).Average

    if ($n -gt 1) {
        $mseVar = ($mseValues | ForEach-Object { ([double]$_ - $mseMean) * ([double]$_ - $mseMean) } | Measure-Object -Sum).Sum / ($n - 1)
        $maeVar = ($maeValues | ForEach-Object { ([double]$_ - $maeMean) * ([double]$_ - $maeMean) } | Measure-Object -Sum).Sum / ($n - 1)
        $mseStd = [Math]::Sqrt($mseVar)
        $maeStd = [Math]::Sqrt($maeVar)
    } else {
        $mseStd = 0.0
        $maeStd = 0.0
    }

    @(
        "tag=$Tag",
        "stage1=pred${PretrainPredLen}_ep${PretrainEpochs}_ratio${PretrainRatio}",
        "stage2=pred${FinetunePredLen}_ep${FinetuneEpochs}_ratio${FinetuneRatio}",
        "finetune_seeds=$($ftSeeds -join ',')",
        "finetune_count=$n",
        "mse_mean=$mseMean",
        "mse_std=$mseStd",
        "mae_mean=$maeMean",
        "mae_std=$maeStd",
        "stage1_checkpoint=$preCkptPath"
    ) | Set-Content $summaryTxt -Encoding UTF8

    Write-Host "Runs file:    $runsCsv" -ForegroundColor Green
    Write-Host "Rank file:    $rankCsv" -ForegroundColor Green
    Write-Host "Summary file: $summaryTxt" -ForegroundColor Green
    Write-Host "Ckpt file:    $ckptCsv" -ForegroundColor Green
    Write-Host ("Finetune MSE mean±std: {0:F6} ± {1:F6}" -f $mseMean, $mseStd) -ForegroundColor Green
    Write-Host ("Finetune MAE mean±std: {0:F6} ± {1:F6}" -f $maeMean, $maeStd) -ForegroundColor Green
} else {
    Write-Host "[WARN] No valid finetune rows were parsed." -ForegroundColor Yellow
}
