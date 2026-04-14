param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int[]]$Seeds = @(42, 52, 62),
    [double[]]$Ratios = @(0.1, 0.05),
    [int]$Epochs = 3,
    [int]$Patience = 2,
    [double]$LambdaTau = 0.05,
    [string]$TauScales = "1,2,4",
    [string]$Tag = "multiscale_tau"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/ppn_multiscale_tau_${Tag}.csv"
$summaryCsv = "logs/ppn_multiscale_tau_${Tag}_summary.csv"

"variant,ratio,seed,mse,mae" | Set-Content $runCsv -Encoding UTF8

foreach ($seed in $Seeds) {
    foreach ($ratio in $Ratios) {
        $ratioTag = ("{0}" -f $ratio).Replace(".", "p")
        foreach ($variant in @(@{Name="base";MultiTau=0}, @{Name="multi_tau";MultiTau=1})) {
            $modelId = "ppn_{0}_fs_{1}_h{2}_s{3}" -f $variant.Name, $ratioTag, $PredLen, $seed
            Write-Host "[RUN] variant=$($variant.Name) ratio=$ratio seed=$seed model_id=$modelId" -ForegroundColor Cyan

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
                --des multiscale_tau_fewshot `
                --checkpoints C:/tmp/ppn_latest_ckpt `
                --seed $seed `
                --ppn_use_tau_loss 1 `
                --ppn_lambda_tau $LambdaTau `
                --ppn_use_multi_scale_tau $variant.MultiTau `
                --ppn_tau_scales $TauScales `
                --train_subset_ratio $ratio `
                --subset_seed $seed `
                --use_amp `
                --use_gpu `
                --gpu_type cuda `
                --gpu 0

            $settingPattern = "long_term_forecast_${modelId}_PPN"
            $tail = Get-Content result_long_term_forecast.txt -Tail 220
            $start = -1
            for ($i = $tail.Count - 1; $i -ge 0; $i--) {
                if ($tail[$i] -like "*$settingPattern*") { $start = $i; break }
            }

            if ($start -ge 0 -and $start + 1 -lt $tail.Count) {
                $metricLine = $tail[$start + 1]
                $mse = [regex]::Match($metricLine, "mse:([0-9\.Ee\-]+)").Groups[1].Value
                $mae = [regex]::Match($metricLine, "mae:([0-9\.Ee\-]+)").Groups[1].Value
                if ($mse -and $mae) {
                    "{0},{1},{2},{3},{4}" -f $variant.Name, $ratio, $seed, $mse, $mae | Add-Content $runCsv -Encoding UTF8
                    Write-Host "[DONE] variant=$($variant.Name) ratio=$ratio seed=$seed mse=$mse mae=$mae" -ForegroundColor Green
                }
            }
        }
    }
}

$rows = Import-Csv $runCsv
$groups = $rows | Group-Object variant, ratio
"variant,ratio,n,mse_mean,mse_std,mae_mean,mae_std" | Set-Content $summaryCsv -Encoding UTF8
foreach ($g in $groups) {
    $mses = $g.Group | ForEach-Object { [double]$_.mse }
    $maes = $g.Group | ForEach-Object { [double]$_.mae }
    $n = $mses.Count
    $mseMean = ($mses | Measure-Object -Average).Average
    $maeMean = ($maes | Measure-Object -Average).Average
    $mseStd = if ($n -gt 1) { [math]::Sqrt((($mses | ForEach-Object { ($_ - $mseMean) * ($_ - $mseMean) } | Measure-Object -Sum).Sum) / ($n - 1)) } else { 0.0 }
    $maeStd = if ($n -gt 1) { [math]::Sqrt((($maes | ForEach-Object { ($_ - $maeMean) * ($_ - $maeMean) } | Measure-Object -Sum).Sum) / ($n - 1)) } else { 0.0 }
    "{0},{1},{2},{3},{4},{5},{6}" -f $g.Group[0].variant, [double]$g.Group[0].ratio, $n, $mseMean, $mseStd, $maeMean, $maeStd | Add-Content $summaryCsv -Encoding UTF8
}

Write-Host "All done. Summary saved to $summaryCsv" -ForegroundColor Green