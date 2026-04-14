param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int]$Seed = 42,
    [int]$Epochs = 10,
    [int]$Patience = 4
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
New-Item -ItemType Directory -Path logs -Force | Out-Null

$ratios = @(1.0, 0.3, 0.1, 0.05)
$summaryCsv = "logs/ppn_fewshot_curve.csv"
"ratio,seed,mse,mae" | Set-Content $summaryCsv -Encoding UTF8

foreach ($ratio in $ratios) {
    $tag = ("{0}" -f $ratio).Replace(".", "p")
    $modelId = "ppn_fs_${tag}_etth1_h${PredLen}_s${Seed}"

    $cmd = @(
        "conda", "run", "-n", $EnvName, "python", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", $modelId,
        "--model", "PPN",
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
        "--d_model", "128",
        "--d_ff", "128",
        "--train_epochs", "$Epochs",
        "--patience", "$Patience",
        "--batch_size", "32",
        "--learning_rate", "0.0003",
        "--lradj", "cosine",
        "--num_workers", "2",
        "--itr", "1",
        "--des", "fewshot_curve",
        "--checkpoints", "C:/tmp/ppn_latest_ckpt",
        "--seed", "$Seed",
        "--ppn_use_tau_loss", "1",
        "--ppn_lambda_tau", "0.2",
        "--train_subset_ratio", "$ratio",
        "--subset_seed", "$Seed",
        "--use_amp",
        "--use_gpu",
        "--gpu_type", "cuda",
        "--gpu", "0"
    )

    Write-Host "[RUN] ratio=$ratio seed=$Seed"
    $runOut = & $cmd[0] $cmd[1..($cmd.Length - 1)] 2>&1
    $runOut | Set-Content "logs/ppn_fs_${tag}.log" -Encoding UTF8
    $runOut | ForEach-Object { $_ }

    $metricLine = Select-String -Path "logs/ppn_fs_${tag}.log" -Pattern "mse:" | Select-Object -Last 1
    if ($metricLine) {
        $txt = $metricLine.Line
        $mse = [regex]::Match($txt, "mse:([0-9\.Ee\-]+)").Groups[1].Value
        $mae = [regex]::Match($txt, "mae:([0-9\.Ee\-]+)").Groups[1].Value
        if ($mse -and $mae) {
            "{0},{1},{2},{3}" -f $ratio, $Seed, $mse, $mae | Add-Content $summaryCsv -Encoding UTF8
            Write-Host "[DONE] ratio=$ratio mse=$mse mae=$mae" -ForegroundColor Green
        } else {
            Write-Host "[WARN] metric parse failed for ratio=$ratio" -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARN] no mse line found for ratio=$ratio" -ForegroundColor Yellow
    }
}

Write-Host "Few-shot curve done. Summary -> $summaryCsv"
