param(
    [string]$EnvName = "py39env",
    [int]$PredLen = 96,
    [int[]]$Seeds = @(42),
    [double[]]$Ratios = @(0.1, 0.05),
    [int]$Epochs = 3,
    [int]$Patience = 2,
    [string]$Tag = "sota_etth1_h96"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
New-Item -ItemType Directory -Path logs -Force | Out-Null

$summaryCsv = "logs/${Tag}_summary.csv"
$rankCsv = "logs/${Tag}_rank.csv"
"model,ratio,seed,mse,mae" | Set-Content $summaryCsv -Encoding UTF8

$runs = @(
    @{ Name = "DLinear"; Model = "DLinear"; Data = "ETTh1"; DataPath = "ETTh1.csv"; EncIn = 7; ELayers = 2; DLayers = 1; DModel = 512; DFF = 2048; NHeads = 8; Factor = 3; UsePatch = $false },
    @{ Name = "PatchTST"; Model = "PatchTST"; Data = "ETTh1"; DataPath = "ETTh1.csv"; EncIn = 7; ELayers = 1; DLayers = 1; DModel = 512; DFF = 2048; NHeads = 2; Factor = 3; UsePatch = $true },
    @{ Name = "iTransformer"; Model = "iTransformer"; Data = "ETTh1"; DataPath = "ETTh1.csv"; EncIn = 7; ELayers = 2; DLayers = 1; DModel = 128; DFF = 128; NHeads = 8; Factor = 3; UsePatch = $false }
)

foreach ($seed in $Seeds) {
    foreach ($run in $runs) {
        foreach ($ratio in $Ratios) {
            $ratioTag = ("{0}" -f $ratio).Replace(".", "p")
            $modelId = "$($run.Name)_fs_${ratioTag}_h${PredLen}_s${seed}"

            Write-Host "[RUN] model=$($run.Name) ratio=$ratio seed=$seed model_id=$modelId" -ForegroundColor Cyan

            $args = @(
                "--task_name", "long_term_forecast",
                "--is_training", "1",
                "--model_id", $modelId,
                "--model", $run.Model,
                "--data", $run.Data,
                "--root_path", "./dataset/ETT-small/",
                "--data_path", $run.DataPath,
                "--features", "M",
                "--target", "OT",
                "--freq", "h",
                "--seq_len", "96",
                "--label_len", "48",
                "--pred_len", "$PredLen",
                "--enc_in", "$($run.EncIn)",
                "--dec_in", "$($run.EncIn)",
                "--c_out", "$($run.EncIn)",
                "--e_layers", "$($run.ELayers)",
                "--d_layers", "$($run.DLayers)",
                "--factor", "$($run.Factor)",
                "--d_model", "$($run.DModel)",
                "--d_ff", "$($run.DFF)",
                "--n_heads", "$($run.NHeads)",
                "--train_epochs", "$Epochs",
                "--patience", "$Patience",
                "--batch_size", "32",
                "--learning_rate", "0.0003",
                "--lradj", "cosine",
                "--num_workers", "2",
                "--itr", "1",
                "--des", $Tag,
                "--checkpoints", "C:/tmp/${Tag}_ckpt",
                "--seed", "$seed",
                "--train_subset_ratio", "$ratio",
                "--subset_seed", "$seed",
                "--use_amp",
                "--use_gpu",
                "--gpu_type", "cuda",
                "--gpu", "0"
            )

            if ($run.UsePatch) {
                $args += @("--patch_len", "16")
            }

            & conda run -n $EnvName python run.py @args
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[WARN] run failed for model_id=$modelId" -ForegroundColor Yellow
                continue
            }

            $settingPattern = "long_term_forecast_${modelId}_${run.Model}"
            $tail = Get-Content result_long_term_forecast.txt -Tail 260
            $start = -1
            for ($i = $tail.Count - 1; $i -ge 0; $i--) {
                if ($tail[$i] -like "*$settingPattern*") { $start = $i; break }
            }

            if ($start -ge 0 -and $start + 1 -lt $tail.Count) {
                $metricLine = $tail[$start + 1]
                $mse = [regex]::Match($metricLine, "mse:([0-9\.Ee\-]+)").Groups[1].Value
                $mae = [regex]::Match($metricLine, "mae:([0-9\.Ee\-]+)").Groups[1].Value
                if ($mse -and $mae) {
                    "{0},{1},{2},{3},{4}" -f $run.Name, $ratio, $seed, $mse, $mae | Add-Content $summaryCsv -Encoding UTF8
                    Write-Host "[DONE] model=$($run.Name) ratio=$ratio seed=$seed mse=$mse mae=$mae" -ForegroundColor Green
                }
            }
        }
    }
}

$rows = Import-Csv $summaryCsv
$ranked = $rows | Sort-Object {[double]$_.mse}, {[double]$_.mae}
"rank,model,ratio,seed,mse,mae" | Set-Content $rankCsv -Encoding UTF8
$rank = 1
foreach ($row in $ranked) {
    "{0},{1},{2},{3},{4},{5}" -f $rank, $row.model, $row.ratio, $row.seed, $row.mse, $row.mae | Add-Content $rankCsv -Encoding UTF8
    $rank += 1
}

Write-Host "SOTA few-shot benchmark done." -ForegroundColor Green
Write-Host "Summary: $summaryCsv"
Write-Host "Rank: $rankCsv"