param(
    [string]$EnvName = "py39env",
    [string]$Tag = "tauonly_short_verify_tau_loss"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null
New-Item -ItemType Directory -Path C:/tmp/ppn_verify_ckpt -Force | Out-Null

$runsCsv = "logs/${Tag}_runs.csv"
 $trainLog = "logs/${Tag}_train.log"
"pred_len,epochs,ratio,seed,test_mse,test_mae,tau_loss_final,tau_smooth_final,proj_cycle_final" | Set-Content $runsCsv -Encoding UTF8

$modelId = "${Tag}_p24_s42"
$seed = 42

Write-Host "[QUICK VERIFY] Testing Tau Loss activation with corrected parameters" -ForegroundColor Cyan
Write-Host "[CONFIG] pred_len=24, epochs=3, ratio=1.0, seed=$seed" -ForegroundColor Cyan

$trainOutput = conda run -n $EnvName python run.py `
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
    --pred_len 24 `
    --enc_in 7 `
    --dec_in 7 `
    --c_out 7 `
    --e_layers 1 `
    --d_layers 1 `
    --factor 3 `
    --d_model 128 `
    --d_ff 128 `
    --train_epochs 3 `
    --patience 2 `
    --batch_size 32 `
    --learning_rate 0.0003 `
    --lradj cosine `
    --num_workers 2 `
    --itr 1 `
    --des $Tag `
    --checkpoints C:/tmp/ppn_verify_ckpt `
    --seed $seed `
    --train_subset_ratio 1.0 `
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
    --ppn_lambda_tau_smooth 0.02 `
    --ppn_lambda_tau_mono 0.01 `
    --ppn_lambda_tau_recon 0.05 `
    --ppn_lambda_tau_contrast 0.1 `
    --ppn_lambda_proj_cycle 0.05 `
    --ppn_lambda_tau_flat 0.01 `
    --ppn_disable_accel_tau_loss_in_axiom_mode 0 `
    --tauonly_anchor_delay_epochs 1 `
    --tauonly_anchor_scale 1.0 `
    --use_amp `
    --use_gpu `
    --gpu_type cuda `
    --gpu 0 2>&1 | Tee-Object -FilePath $trainLog

if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}

$tail = @($trainOutput)

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
    return "0"
}

$mse = Get-LastMetricValue -Lines $tail -Pattern "mse:([0-9\.Ee\-]+)"
$mae = Get-LastMetricValue -Lines $tail -Pattern "mae:([0-9\.Ee\-]+)"
$tau_loss_final = Get-LastMetricValue -Lines $tail -Pattern "Tau Loss: ([0-9\.Ee\-]+)"
$tau_smooth_final = Get-LastMetricValue -Lines $tail -Pattern "Tau Smooth Loss: ([0-9\.Ee\-]+)"
$proj_cycle_final = Get-LastMetricValue -Lines $tail -Pattern "Tau Proj Cycle: ([0-9\.Ee\-]+)"

"24,3,1.0,$seed,$mse,$mae,$tau_loss_final,$tau_smooth_final,$proj_cycle_final" | Add-Content $runsCsv -Encoding UTF8

Write-Host "[VERIFY RESULT]" -ForegroundColor Green
Write-Host "  Test MSE: $mse" -ForegroundColor Green
Write-Host "  Test MAE: $mae" -ForegroundColor Green
Write-Host "  Final Tau Loss: $tau_loss_final (should be > 0)" -ForegroundColor Cyan
Write-Host "  Final Tau Smooth Loss: $tau_smooth_final" -ForegroundColor Cyan
Write-Host "  Final Proj Cycle Loss: $proj_cycle_final (should be reasonable, not huge)" -ForegroundColor Cyan

if ([double]$tau_loss_final -gt 0.001) {
    Write-Host "✅ TAU LOSS ACTIVATED! Proceeding with full pipeline." -ForegroundColor Green
} else {
    Write-Host "❌ WARNING: Tau Loss still inactive or too small. Check parameters." -ForegroundColor Yellow
}

Write-Host "CSV saved: $runsCsv" -ForegroundColor Green
