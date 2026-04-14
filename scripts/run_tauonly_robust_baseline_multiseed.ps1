param(
    [string]$EnvName = "py39env",
    [double]$Ratio = 0.1,
    [int]$Epochs = 5,
    [int]$Patience = 3,
    [int]$BatchSize = 16,
    [string]$Tag = "tauonly_robust_baseline_v1"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/${Tag}_runs.csv"
$rankCsv = "logs/${Tag}_rank.csv"
$summaryTxt = "logs/${Tag}_summary.txt"
$ckptCsv = "logs/${Tag}_checkpoints.csv"

"seed,mse,mae,model_id,checkpoint_path,checkpoint_exists" | Set-Content $runCsv -Encoding UTF8
"seed,model_id,checkpoint_path,checkpoint_exists" | Set-Content $ckptCsv -Encoding UTF8

$seeds = @(42, 52, 62)
$mseList = @()
$maeList = @()

function Get-LastMetricValue {
    param(
        [string[]]$Lines,
        [string]$Pattern
    )
    for ($i = $Lines.Count - 1; $i -ge 0; $i--) {
        $m = [regex]::Match($Lines[$i], $Pattern)
        if ($m.Success) {
            return $m.Groups[1].Value
        }
    }
    return ""
}

foreach ($seed in $seeds) {
    $ratioTag = ("{0}" -f $Ratio).Replace('.', 'p')
    $modelId = "${Tag}_r${ratioTag}_s${seed}"
    $seedLog = "logs/${Tag}_seed${seed}.log"

    Write-Host "[RUN] seed=$seed epochs=$Epochs ratio=$Ratio" -ForegroundColor Cyan

    conda run -n $EnvName python run.py `
        --task_name long_term_forecast `
        --is_training 1 `
        --model_id $modelId `
        --model TauOnly `
        --data ETTh1 `
        --root_path ./dataset/ETT-small/ `
        --data_path ETTh1.csv `
        --features M `
        --target OT `
        --freq h `
        --seq_len 96 `
        --label_len 48 `
        --pred_len 96 `
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
        --batch_size $BatchSize `
        --learning_rate 0.0003 `
        --lradj cosine `
        --num_workers 2 `
        --itr 1 `
        --des $Tag `
        --checkpoints C:/tmp/ppn_latest_ckpt `
        --seed $seed `
        --train_subset_ratio $Ratio `
        --subset_seed $seed `
        --ppn_use_transformer 1 `
        --ppn_max_tau_integration_steps 8 `
        --ppn_tau_use_residual_refine 1 `
        --ppn_use_tau_phase_schedule 1 `
        --ppn_tau_phase1_ratio 0.6 `
        --ppn_tau_phase2_ratio 0.3 `
        --ppn_tau_phase1_aux_scale 0.0 `
        --ppn_tau_phase2_aux_scale 0.4 `
        --ppn_tau_phase3_aux_scale 1.0 `
        --ppn_lambda_tau_smooth 0.01 `
        --ppn_lambda_tau_contrast 0.0 `
        --ppn_lambda_proj_cycle 0.0 `
        --ppn_lambda_tau_recon 0.05 `
        --ppn_lambda_tau_flat 0.0 `
        --ppn_lambda_tau_mono 0.01 `
        --ppn_disable_accel_tau_loss_in_axiom_mode 0 `
        --tauonly_anchor_delay_epochs 1 `
        --tauonly_anchor_scale 1.0 `
        --use_amp `
        --use_gpu `
        --gpu_type cuda `
        --gpu 0 2>&1 | Tee-Object -FilePath $seedLog

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] failed seed=$seed" -ForegroundColor Yellow
        continue
    }

    $tail = Get-Content $seedLog -Tail 1200
    $mse = Get-LastMetricValue -Lines $tail -Pattern "mse:([0-9\.Ee\-]+)"
    $mae = Get-LastMetricValue -Lines $tail -Pattern "mae:([0-9\.Ee\-]+)"

    $setting = ""
    for ($i = $tail.Count - 1; $i -ge 0; $i--) {
        if ($tail[$i] -like "*>>>>>>>testing : *<<<<<<<<*") {
            $setting = ($tail[$i] -replace "^.*>>>>>>>testing : ","") -replace "<<<<<<<<.*$", ""
            $setting = $setting.Trim()
            break
        }
    }

    $checkpointPath = ""
    if ($setting) {
        $checkpointPath = Join-Path "C:/tmp/ppn_latest_ckpt" "$setting/checkpoint.pth"
    }
    $checkpointExists = if ($checkpointPath -and (Test-Path $checkpointPath)) { "1" } else { "0" }

    if ($mse -and $mae) {
        "{0},{1},{2},{3},{4},{5}" -f $seed, $mse, $mae, $modelId, $checkpointPath, $checkpointExists | Add-Content $runCsv -Encoding UTF8
        "{0},{1},{2},{3}" -f $seed, $modelId, $checkpointPath, $checkpointExists | Add-Content $ckptCsv -Encoding UTF8
        $mseList += [double]$mse
        $maeList += [double]$mae
        Write-Host "[DONE] seed=$seed mse=$mse mae=$mae ckpt=$checkpointExists" -ForegroundColor Green
    } else {
        Write-Host "[WARN] failed parsing metrics seed=$seed" -ForegroundColor Yellow
    }
}

$rows = Import-Csv $runCsv | Where-Object { $_.mse -and $_.mae }
if ($rows.Count -gt 0) {
    $rows | Sort-Object {[double]$_.mse}, {[double]$_.mae} | Export-Csv -Path $rankCsv -NoTypeInformation -Encoding UTF8

    $mseValues = $rows | ForEach-Object { [double]$_.mse }
    $maeValues = $rows | ForEach-Object { [double]$_.mae }
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
        "epochs=$Epochs",
        "ratio=$Ratio",
        "seeds=$($seeds -join ',')",
        "count=$n",
        "mse_mean=$mseMean",
        "mse_std=$mseStd",
        "mae_mean=$maeMean",
        "mae_std=$maeStd"
    ) | Set-Content $summaryTxt -Encoding UTF8

    Write-Host "Run file: $runCsv" -ForegroundColor Green
    Write-Host "Rank file: $rankCsv" -ForegroundColor Green
    Write-Host "Summary:  $summaryTxt" -ForegroundColor Green
    Write-Host "Ckpt file: $ckptCsv" -ForegroundColor Green
    Write-Host ("MSE mean±std: {0:F6} ± {1:F6}" -f $mseMean, $mseStd) -ForegroundColor Green
    Write-Host ("MAE mean±std: {0:F6} ± {1:F6}" -f $maeMean, $maeStd) -ForegroundColor Green
} else {
    Write-Host "[WARN] no valid rows parsed; skip summary" -ForegroundColor Yellow
}
