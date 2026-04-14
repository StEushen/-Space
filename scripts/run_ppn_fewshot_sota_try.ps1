param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int]$Seed = 42,
    [int]$Epochs = 3,
    [int]$Patience = 2
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null
$summaryCsv = "logs/ppn_fewshot_sota_try.csv"
"variant,ratio,seed,mse,mae" | Set-Content $summaryCsv -Encoding UTF8

$ratios = @(0.1, 0.05)
$variants = @(
    @{ Name = "base"; Patch = 0; Gate = 0 },
    @{ Name = "sota_try"; Patch = 1; Gate = 1 }
)

foreach ($v in $variants) {
    $variantName = $v['Name']
    $variantPatch = $v['Patch']
    $variantGate = $v['Gate']
    foreach ($ratio in $ratios) {
        $tag = ("{0}" -f $ratio).Replace(".", "p")
        $modelId = "ppn_${variantName}_fs_${tag}_etth1_h${PredLen}_s${Seed}"

        Write-Host "[RUN] variant=$variantName ratio=$ratio seed=$Seed model_id=$modelId" -ForegroundColor Cyan

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
            --des fewshot_sota_try `
            --checkpoints C:/tmp/ppn_latest_ckpt `
            --seed $Seed `
            --ppn_use_tau_loss 1 `
            --ppn_lambda_tau 0.2 `
            --ppn_use_patch_embed $variantPatch `
            --ppn_patch_len 8 `
            --ppn_patch_stride 4 `
            --ppn_use_var_tau_gate $variantGate `
            --ppn_var_tau_gate_scale 0.25 `
            --train_subset_ratio $ratio `
            --subset_seed $Seed `
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
                "{0},{1},{2},{3},{4}" -f $variantName, $ratio, $Seed, $mse, $mae | Add-Content $summaryCsv -Encoding UTF8
                Write-Host "[DONE] variant=$variantName ratio=$ratio mse=$mse mae=$mae" -ForegroundColor Green
            } else {
                Write-Host "[WARN] Failed to parse metric line: $metricLine" -ForegroundColor Yellow
            }
        } else {
            Write-Host "[WARN] Could not locate result block for model_id=$modelId" -ForegroundColor Yellow
        }
    }
}

Write-Host "All done. Summary saved to $summaryCsv" -ForegroundColor Green
