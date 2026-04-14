param(
    [string]$EnvName = "py39env",
    [int[]]$Seeds = @(42, 52),
    [double[]]$Ratios = @(0.1, 0.05),
    [int]$Epochs = 1,
    [int]$Patience = 1,
    [int]$BatchSize = 16,
    [string]$Tag = "tau_iayn_fast"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/${Tag}_runs.csv"
$summaryCsv = "logs/${Tag}_summary.csv"
$rankCsv = "logs/${Tag}_rank.csv"

"model,ratio,seed,mse,mae" | Set-Content $runCsv -Encoding UTF8

$models = @("TauOnly", "DLinear", "PatchTST")

foreach ($seed in $Seeds) {
    foreach ($ratio in $Ratios) {
        $ratioTag = ("{0}" -f $ratio).Replace(".", "p")
        foreach ($m in $models) {
            $modelId = "${Tag}_${m}_r${ratioTag}_s${seed}"
            Write-Host "[RUN] model=$m ratio=$ratio seed=$seed" -ForegroundColor Cyan

            conda run -n $EnvName python run.py `
                --task_name long_term_forecast `
                --is_training 1 `
                --model_id $modelId `
                --model $m `
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
                --train_subset_ratio $ratio `
                --subset_seed $seed `
                --ppn_use_transformer 0 `
                --ppn_max_tau_integration_steps 8 `
                --ppn_tau_use_residual_refine 1 `
                --use_amp `
                --use_gpu `
                --gpu_type cuda `
                --gpu 0

            if ($LASTEXITCODE -ne 0) {
                Write-Host "[WARN] training failed: $m ratio=$ratio seed=$seed" -ForegroundColor Yellow
                continue
            }

            $tail = Get-Content .\result_long_term_forecast.txt -Tail 420
            $settingPattern = "long_term_forecast_${modelId}_${m}"
            $start = -1
            for ($i = $tail.Count - 1; $i -ge 0; $i--) {
                if ($tail[$i] -like "*$settingPattern*") { $start = $i; break }
            }

            if ($start -ge 0 -and $start + 1 -lt $tail.Count) {
                $metricLine = $tail[$start + 1]
                $mse = [regex]::Match($metricLine, "mse:([0-9\.Ee\-]+)").Groups[1].Value
                $mae = [regex]::Match($metricLine, "mae:([0-9\.Ee\-]+)").Groups[1].Value
                if ($mse -and $mae) {
                    "{0},{1},{2},{3},{4}" -f $m, $ratio, $seed, $mse, $mae | Add-Content $runCsv -Encoding UTF8
                    Write-Host "[DONE] model=$m ratio=$ratio seed=$seed mse=$mse mae=$mae" -ForegroundColor Green
                }
            }
        }
    }
}

$rows = Import-Csv $runCsv
$groups = $rows | Group-Object model, ratio
"model,ratio,n,mse_mean,mse_std,mae_mean,mae_std" | Set-Content $summaryCsv -Encoding UTF8

$summary = @()
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

    $model = $g.Group[0].model
    $ratio = [double]$g.Group[0].ratio

    "{0},{1},{2},{3},{4},{5},{6}" -f $model, $ratio, $n, $mseMean, $mseStd, $maeMean, $maeStd | Add-Content $summaryCsv -Encoding UTF8

    $summary += [PSCustomObject]@{
        model = $model
        ratio = $ratio
        n = $n
        mse_mean = $mseMean
        mse_std = $mseStd
        mae_mean = $maeMean
        mae_std = $maeStd
    }
}

$rank = @()
foreach ($ratioGroup in ($summary | Group-Object ratio)) {
    $idx = 1
    foreach ($r in ($ratioGroup.Group | Sort-Object mse_mean, mae_mean, mse_std)) {
        $rank += [PSCustomObject]@{
            rank = $idx
            model = $r.model
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

$rank | Export-Csv -Path $rankCsv -NoTypeInformation -Encoding UTF8
Write-Host "Run-level: $runCsv" -ForegroundColor Green
Write-Host "Summary:   $summaryCsv" -ForegroundColor Green
Write-Host "Rank:      $rankCsv" -ForegroundColor Green
