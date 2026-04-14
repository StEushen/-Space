param(
    [string]$EnvName = "py39env",
    [double]$Ratio = 0.1,
    [int]$Seed = 42,
    [int]$Epochs = 1,
    [int]$Patience = 1,
    [int]$BatchSize = 16,
    [string]$Tag = "tauonly_strategy"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/${Tag}_runs.csv"
"exp,warmup,anchor_delay,aux_mode,mse,mae" | Set-Content $runCsv -Encoding UTF8

$exps = @(
    @{ Name = "Baseline"; Warmup = "current"; AnchorDelay = 0; AuxMode = "full"; P1 = 0.3; P2 = 0.5; S1 = 0.3; S2 = 1.0; S3 = 1.2; LContrast = 0.01; LCycle = 0.01; LFlat = 0.01; LRecon = 0.05; LMono = 0.01 },
    @{ Name = "A1"; Warmup = "aggressive"; AnchorDelay = 0; AuxMode = "lite"; P1 = 0.6; P2 = 0.3; S1 = 0.0; S2 = 0.4; S3 = 1.0; LContrast = 0.0; LCycle = 0.0; LFlat = 0.005; LRecon = 0.03; LMono = 0.01 },
    @{ Name = "A2"; Warmup = "aggressive"; AnchorDelay = 1; AuxMode = "lite"; P1 = 0.6; P2 = 0.3; S1 = 0.0; S2 = 0.4; S3 = 1.0; LContrast = 0.0; LCycle = 0.0; LFlat = 0.005; LRecon = 0.03; LMono = 0.01 },
    @{ Name = "A3"; Warmup = "conservative"; AnchorDelay = 1; AuxMode = "lite"; P1 = 0.4; P2 = 0.4; S1 = 0.1; S2 = 0.7; S3 = 1.0; LContrast = 0.0; LCycle = 0.0; LFlat = 0.005; LRecon = 0.03; LMono = 0.01 }
)

foreach ($e in $exps) {
    $ratioTag = ("{0}" -f $Ratio).Replace('.', 'p')
    $modelId = "${Tag}_${($e.Name)}_r${ratioTag}_s${Seed}"
    Write-Host "[RUN] exp=$($e.Name) warmup=$($e.Warmup) anchor_delay=$($e.AnchorDelay) aux=$($e.AuxMode)" -ForegroundColor Cyan

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
        --seed $Seed `
        --train_subset_ratio $Ratio `
        --subset_seed $Seed `
        --ppn_use_transformer 1 `
        --ppn_max_tau_integration_steps 8 `
        --ppn_tau_use_residual_refine 1 `
        --ppn_use_tau_phase_schedule 1 `
        --ppn_tau_phase1_ratio $($e.P1) `
        --ppn_tau_phase2_ratio $($e.P2) `
        --ppn_tau_phase1_aux_scale $($e.S1) `
        --ppn_tau_phase2_aux_scale $($e.S2) `
        --ppn_tau_phase3_aux_scale $($e.S3) `
        --ppn_lambda_tau_contrast $($e.LContrast) `
        --ppn_lambda_proj_cycle $($e.LCycle) `
        --ppn_lambda_tau_flat $($e.LFlat) `
        --ppn_lambda_tau_recon $($e.LRecon) `
        --ppn_lambda_tau_mono $($e.LMono) `
        --tauonly_anchor_delay_epochs $($e.AnchorDelay) `
        --tauonly_anchor_scale 1.0 `
        --use_amp `
        --use_gpu `
        --gpu_type cuda `
        --gpu 0

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] failed exp=$($e.Name)" -ForegroundColor Yellow
        continue
    }

    $tail = Get-Content .\result_long_term_forecast.txt -Tail 420
    $pattern = "long_term_forecast_${modelId}_TauOnly"
    $idx = -1
    for ($i = $tail.Count - 1; $i -ge 0; $i--) {
        if ($tail[$i] -like "*$pattern*") { $idx = $i; break }
    }

    if ($idx -ge 0 -and $idx + 1 -lt $tail.Count) {
        $line = $tail[$idx + 1]
        $mse = [regex]::Match($line, "mse:([0-9\.Ee\-]+)").Groups[1].Value
        $mae = [regex]::Match($line, "mae:([0-9\.Ee\-]+)").Groups[1].Value
        if ($mse -and $mae) {
            "{0},{1},{2},{3},{4},{5}" -f $e.Name, $e.Warmup, $e.AnchorDelay, $e.AuxMode, $mse, $mae | Add-Content $runCsv -Encoding UTF8
            Write-Host "[DONE] exp=$($e.Name) mse=$mse mae=$mae" -ForegroundColor Green
        }
    }
}

$rows = Import-Csv $runCsv
$rows | Sort-Object {[double]$_.mse}, {[double]$_.mae} | Export-Csv -Path "logs/${Tag}_rank.csv" -NoTypeInformation -Encoding UTF8
Write-Host "Run file: logs/${Tag}_runs.csv" -ForegroundColor Green
Write-Host "Rank file: logs/${Tag}_rank.csv" -ForegroundColor Green
