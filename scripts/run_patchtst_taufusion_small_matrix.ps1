param(
    [string]$EnvName = "py39env",
    [string]$Tag = "patchtst_taufusion_small_v1",
    [double]$Ratio = 0.1,
    [int]$Epochs = 3,
    [int]$Patience = 2,
    [int[]]$Seeds = @(42, 52, 62)
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null

$runCsv = "logs/${Tag}_runs.csv"
$rankCsv = "logs/${Tag}_rank.csv"
$summaryCsv = "logs/${Tag}_summary.csv"

"model,ratio,seed,mse,mae,model_id" | Set-Content $runCsv -Encoding UTF8

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
    return ""
}

$runs = @(
    @{ Name = "PatchTST"; Model = "PatchTST"; ExtraArgs = @("--patch_len", "16") },
    @{ Name = "TauFusion_PatchTST"; Model = "TauFusion"; ExtraArgs = @("--tau_base_model", "PatchTST", "--tau_fuse_alpha", "0.3", "--patch_len", "16", "--ppn_disable_accel_tau_loss_in_axiom_mode", "0") }
)

foreach ($seed in $Seeds) {
    foreach ($run in $runs) {
        $ratioTag = ("{0}" -f $Ratio).Replace('.', 'p')
        $modelId = "${Tag}_$($run.Name)_r${ratioTag}_s${seed}"
        $seedLog = "logs/${Tag}_${run.Name}_seed${seed}.log"

        Write-Host "[RUN] model=$($run.Name) ratio=$Ratio seed=$seed" -ForegroundColor Cyan

        $args = @(
            "--task_name", "long_term_forecast",
            "--is_training", "1",
            "--model_id", $modelId,
            "--model", $run.Model,
            "--data", "ETTh1",
            "--root_path", "./dataset/ETT-small/",
            "--data_path", "ETTh1.csv",
            "--features", "M",
            "--target", "OT",
            "--freq", "h",
            "--seq_len", "96",
            "--label_len", "48",
            "--pred_len", "96",
            "--enc_in", "7",
            "--dec_in", "7",
            "--c_out", "7",
            "--e_layers", "1",
            "--d_layers", "1",
            "--factor", "3",
            "--d_model", "512",
            "--d_ff", "2048",
            "--n_heads", "2",
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
            "--train_subset_ratio", "$Ratio",
            "--subset_seed", "$seed",
            "--use_amp",
            "--use_gpu",
            "--gpu_type", "cuda",
            "--gpu", "0"
        ) + $run.ExtraArgs

        & conda run -n $EnvName python run.py @args 2>&1 | Tee-Object -FilePath $seedLog

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] run failed model=$($run.Name) seed=$seed" -ForegroundColor Yellow
            continue
        }

        $tail = Get-Content $seedLog -Tail 1200
        $mse = Get-LastMetricValue -Lines $tail -Pattern "mse:([0-9\.Ee\-]+)"
        $mae = Get-LastMetricValue -Lines $tail -Pattern "mae:([0-9\.Ee\-]+)"

        if ($mse -and $mae) {
            "{0},{1},{2},{3},{4},{5}" -f $run.Name, $Ratio, $seed, $mse, $mae, $modelId | Add-Content $runCsv -Encoding UTF8
            Write-Host "[DONE] model=$($run.Name) seed=$seed mse=$mse mae=$mae" -ForegroundColor Green
        } else {
            Write-Host "[WARN] metric parse failed model=$($run.Name) seed=$seed" -ForegroundColor Yellow
        }
    }
}

$rows = Import-Csv $runCsv
if ($rows.Count -eq 0) {
    throw "No valid rows parsed in $runCsv"
}

$rows | Sort-Object {[double]$_.mse}, {[double]$_.mae} | Export-Csv -Path $rankCsv -NoTypeInformation -Encoding UTF8

$summary = @()
$groups = $rows | Group-Object model
foreach ($g in $groups) {
    $mseValues = $g.Group | ForEach-Object { [double]$_.mse }
    $maeValues = $g.Group | ForEach-Object { [double]$_.mae }
    $n = $mseValues.Count
    $mseMean = ($mseValues | Measure-Object -Average).Average
    $maeMean = ($maeValues | Measure-Object -Average).Average
    if ($n -gt 1) {
        $mseVar = ($mseValues | ForEach-Object { ($_ - $mseMean) * ($_ - $mseMean) } | Measure-Object -Sum).Sum / ($n - 1)
        $maeVar = ($maeValues | ForEach-Object { ($_ - $maeMean) * ($_ - $maeMean) } | Measure-Object -Sum).Sum / ($n - 1)
        $mseStd = [Math]::Sqrt($mseVar)
        $maeStd = [Math]::Sqrt($maeVar)
    } else {
        $mseStd = 0.0
        $maeStd = 0.0
    }
    $summary += [PSCustomObject]@{
        model = $g.Name
        ratio = $Ratio
        n = $n
        mse_mean = $mseMean
        mse_std = $mseStd
        mae_mean = $maeMean
        mae_std = $maeStd
    }
}

$summary | Sort-Object mse_mean | Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8

Write-Host "Run file: $runCsv" -ForegroundColor Green
Write-Host "Rank file: $rankCsv" -ForegroundColor Green
Write-Host "Summary:  $summaryCsv" -ForegroundColor Green
