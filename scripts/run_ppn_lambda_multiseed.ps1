param(
    [string]$EnvName = "py39env",
    [double]$TrainRatio = 0.1,
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int]$Epochs = 5,
    [int]$Patience = 3
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null
$summaryCsv = "logs/ppn_lambda_multiseed.csv"
"ratio,lambda_tau,seed,mse,mae" | Set-Content $summaryCsv -Encoding UTF8

$seeds = @(42, 52, 62)
$lambdas = @(0.05, 0.3)

foreach ($lam in $lambdas) {
    foreach ($seed in $seeds) {
        $ratioTag = ("{0}" -f $TrainRatio).Replace('.', 'p')
        $lamTag = ("{0}" -f $lam).Replace('.', 'p')
        $modelId = "ppn_ms_l${lamTag}_r${ratioTag}_h${PredLen}_s${seed}"

        Write-Host "[RUN] ratio=$TrainRatio lambda_tau=$lam seed=$seed" -ForegroundColor Cyan

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
            --des lambda_multiseed `
            --checkpoints C:/tmp/ppn_latest_ckpt `
            --seed $seed `
            --ppn_use_tau_loss 1 `
            --ppn_lambda_tau $lam `
            --train_subset_ratio $TrainRatio `
            --subset_seed $seed `
            --use_amp `
            --use_gpu `
            --gpu_type cuda `
            --gpu 0

        $settingPattern = "long_term_forecast_${modelId}_PPN"
        $tail = Get-Content result_long_term_forecast.txt -Tail 240
        $start = -1
        for ($i = $tail.Count - 1; $i -ge 0; $i--) {
            if ($tail[$i] -like "*$settingPattern*") { $start = $i; break }
        }

        if ($start -ge 0 -and $start + 1 -lt $tail.Count) {
            $metricLine = $tail[$start + 1]
            $mse = [regex]::Match($metricLine, "mse:([0-9\.Ee\-]+)").Groups[1].Value
            $mae = [regex]::Match($metricLine, "mae:([0-9\.Ee\-]+)").Groups[1].Value
            if ($mse -and $mae) {
                "{0},{1},{2},{3},{4}" -f $TrainRatio, $lam, $seed, $mse, $mae | Add-Content $summaryCsv -Encoding UTF8
                Write-Host "[DONE] lambda_tau=$lam seed=$seed mse=$mse mae=$mae" -ForegroundColor Green
            } else {
                Write-Host "[WARN] Failed to parse metric for $modelId" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[WARN] Could not locate result block for $modelId" -ForegroundColor Yellow
        }
    }
}

Write-Host "Multiseed done. Summary -> $summaryCsv" -ForegroundColor Green
