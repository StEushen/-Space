param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [double]$Ratio = 0.1,
    [int[]]$Seeds = @(42, 52, 62),
    [int]$Epochs = 5,
    [string]$Tag = "exp2_patchtst_teacher_gru_student"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/${Tag}_runs.csv"
$summaryCsv = "logs/${Tag}_summary.csv"
"seed,mse,mae,model_id" | Set-Content $runCsv -Encoding UTF8

function Get-LastMetricValue {
    param([string[]]$Lines, [string]$Pattern)
    for ($i = $Lines.Count - 1; $i -ge 0; $i--) {
        $m = [regex]::Match($Lines[$i], $Pattern)
        if ($m.Success) { return $m.Groups[1].Value }
    }
    return ""
}

foreach ($seed in $Seeds) {
    $modelId = "${Tag}_s${seed}"
    $seedLog = "logs/${Tag}_seed${seed}.log"
    Write-Host "[RUN] seed=$seed" -ForegroundColor Cyan

    $runOptions = @(
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", $modelId,
        "--model", "TauFusion",
        "--tau_base_model", "PatchTST",
        "--tau_fuse_alpha", "0.3",
        "--data", $Data,
        "--root_path", "./dataset/ETT-small/",
        "--data_path", $DataPath,
        "--features", "M",
        "--target", "OT",
        "--freq", "h",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", "$PredLen",
        "--enc_in", "7",
        "--dec_in", "7",
        "--c_out", "7",
        "--e_layers", "1",
        "--d_layers", "1",
        "--factor", "3",
        "--d_model", "512",
        "--d_ff", "2048",
        "--n_heads", "2",
        "--patch_len", "16",
        "--train_epochs", "$Epochs",
        "--patience", "2",
        "--batch_size", "32",
        "--learning_rate", "0.0003",
        "--lradj", "cosine",
        "--num_workers", "2",
        "--itr", "1",
        "--des", $Tag,
        "--checkpoints", "C:/tmp/${Tag}_ckpt",
        "--seed", "$seed",
        "--train_subset_ratio", "$Ratio",
        "--subset_seed", "$seed",
        "--ppn_use_transformer", "0",
        "--ppn_disable_accel_tau_loss_in_axiom_mode", "0",
        "--use_amp",
        "--use_gpu",
        "--gpu_type", "cuda",
        "--gpu", "0"
    )

    & conda run -n $EnvName python run.py @runOptions 2>&1 | Tee-Object -FilePath $seedLog
    if ($LASTEXITCODE -ne 0) { throw "Run failed for seed=$seed" }

    $tail = Get-Content $seedLog -Tail 1200
    $mse = Get-LastMetricValue -Lines $tail -Pattern "mse:([0-9\.Ee\-]+)"
    $mae = Get-LastMetricValue -Lines $tail -Pattern "mae:([0-9\.Ee\-]+)"
    if (-not $mse -or -not $mae) { throw "Metric parse failed for seed=$seed" }

    "{0},{1},{2},{3}" -f $seed, $mse, $mae, $modelId | Add-Content $runCsv -Encoding UTF8
    Write-Host "[DONE] seed=$seed mse=$mse mae=$mae" -ForegroundColor Green
}

$rows = Import-Csv $runCsv
$mseValues = $rows | ForEach-Object { [double]$_.mse }
$maeValues = $rows | ForEach-Object { [double]$_.mae }
$n = $mseValues.Count
$mseMean = ($mseValues | Measure-Object -Average).Average
$maeMean = ($maeValues | Measure-Object -Average).Average
if ($n -gt 1) {
    $mseStd = [Math]::Sqrt((($mseValues | ForEach-Object { ($_ - $mseMean) * ($_ - $mseMean) } | Measure-Object -Sum).Sum) / ($n - 1))
    $maeStd = [Math]::Sqrt((($maeValues | ForEach-Object { ($_ - $maeMean) * ($_ - $maeMean) } | Measure-Object -Sum).Sum) / ($n - 1))
} else {
    $mseStd = 0.0
    $maeStd = 0.0
}

"tag,n,mse_mean,mse_std,mae_mean,mae_std" | Set-Content $summaryCsv -Encoding UTF8
"{0},{1},{2},{3},{4},{5}" -f $Tag, $n, $mseMean, $mseStd, $maeMean, $maeStd | Add-Content $summaryCsv -Encoding UTF8
Write-Host "Run file: $runCsv" -ForegroundColor Green
Write-Host "Summary : $summaryCsv" -ForegroundColor Green
