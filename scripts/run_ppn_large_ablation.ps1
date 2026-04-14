param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int[]]$Seeds = @(42, 52),
    [double[]]$Ratios = @(0.1, 0.05),
    [int]$Epochs = 1,
    [int]$Patience = 1,
    [int]$BatchSize = 16,
    [int]$MaxTauSteps = 8,
    [double]$LambdaTau = 0.2,
    [string]$Tag = "large_ablation"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/ppn_${Tag}_runs.csv"
$summaryCsv = "logs/ppn_${Tag}_summary.csv"
$rankCsv = "logs/ppn_${Tag}_rank.csv"

"variant,ratio,seed,mse,mae" | Set-Content $runCsv -Encoding UTF8

$variants = @(
    @{ Name = "gru_full"; UseTransformer = 0; LambdaContrast = 0.01; LambdaCycle = 0.01; UseResidualRefine = 1 },
    @{ Name = "tf_full"; UseTransformer = 1; LambdaContrast = 0.01; LambdaCycle = 0.01; UseResidualRefine = 1 },
    @{ Name = "gru_no_contrast"; UseTransformer = 0; LambdaContrast = 0.0; LambdaCycle = 0.01; UseResidualRefine = 1 },
    @{ Name = "tf_no_contrast"; UseTransformer = 1; LambdaContrast = 0.0; LambdaCycle = 0.01; UseResidualRefine = 1 },
    @{ Name = "gru_no_cycle"; UseTransformer = 0; LambdaContrast = 0.01; LambdaCycle = 0.0; UseResidualRefine = 1 },
    @{ Name = "tf_no_cycle"; UseTransformer = 1; LambdaContrast = 0.01; LambdaCycle = 0.0; UseResidualRefine = 1 },
    @{ Name = "gru_no_refine"; UseTransformer = 0; LambdaContrast = 0.01; LambdaCycle = 0.01; UseResidualRefine = 0 },
    @{ Name = "tf_no_refine"; UseTransformer = 1; LambdaContrast = 0.01; LambdaCycle = 0.01; UseResidualRefine = 0 }
)

foreach ($seed in $Seeds) {
    foreach ($ratio in $Ratios) {
        $ratioTag = ("{0}" -f $ratio).Replace(".", "p")

        foreach ($v in $variants) {
            $variantName = $v.Name
            $useTransformer = [int]$v.UseTransformer
            $lambdaContrast = [double]$v.LambdaContrast
            $lambdaCycle = [double]$v.LambdaCycle
            $useResidualRefine = [int]$v.UseResidualRefine

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
                --batch_size $BatchSize `
                --learning_rate 0.0003 `
                --lradj cosine `
                --num_workers 2 `
                --itr 1 `
                --des $Tag `
                --checkpoints C:/tmp/ppn_latest_ckpt `
                --seed $seed `
                --ppn_use_transformer $useTransformer `
                --ppn_use_tau_loss 1 `
                --ppn_lambda_tau $LambdaTau `
                --ppn_lambda_proj_cycle $lambdaCycle `
                --ppn_lambda_tau_contrast $lambdaContrast `
                --ppn_tau_use_residual_refine $useResidualRefine `
                --ppn_max_tau_integration_steps $MaxTauSteps `
                --train_subset_ratio $ratio `
                --subset_seed $seed `
                --use_amp `
                --use_gpu `
                --gpu_type cuda `
                --gpu 0

            if ($LASTEXITCODE -ne 0) {
                Write-Host "[WARN] Training command failed for $modelId (exit=$LASTEXITCODE)" -ForegroundColor Yellow
                continue
            }

            $settingPattern = "long_term_forecast_${modelId}_PPN"
            $tail = Get-Content result_long_term_forecast.txt -Tail 320
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
                    Write-Host "[WARN] Parse failed for metric line: $metricLine" -ForegroundColor Yellow
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
