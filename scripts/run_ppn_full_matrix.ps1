param(
    [string]$CondaEnv = "py39env",
    [switch]$DryRun,
    [int]$MaxRuns = 0
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

New-Item -ItemType Directory -Force -Path logs | Out-Null
$statusFile = "logs/ppn_full_status.txt"
$runLog = "logs/ppn_full_runs.log"

$datasets = @(
    @{ Name = "ETTh1"; Data = "ETTh1"; RootPath = "./dataset/ETT-small/"; DataPath = "ETTh1.csv"; EncIn = 7; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "h" },
    @{ Name = "ETTh2"; Data = "ETTh2"; RootPath = "./dataset/ETT-small/"; DataPath = "ETTh2.csv"; EncIn = 7; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "h" },
    @{ Name = "ETTm1"; Data = "ETTm1"; RootPath = "./dataset/ETT-small/"; DataPath = "ETTm1.csv"; EncIn = 7; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "t" },
    @{ Name = "ETTm2"; Data = "ETTm2"; RootPath = "./dataset/ETT-small/"; DataPath = "ETTm2.csv"; EncIn = 7; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "t" },
    @{ Name = "Weather"; Data = "custom"; RootPath = "./dataset/tslib_hf/weather/"; DataPath = "weather.csv"; EncIn = 21; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "h" },
    @{ Name = "Electricity"; Data = "custom"; RootPath = "./dataset/tslib_hf/electricity/"; DataPath = "electricity.csv"; EncIn = 321; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "h" },
    @{ Name = "Traffic"; Data = "custom"; RootPath = "./dataset/tslib_hf/traffic/"; DataPath = "traffic.csv"; EncIn = 862; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "h" },
    @{ Name = "Exchange"; Data = "custom"; RootPath = "./dataset/tslib_hf/exchange_rate/"; DataPath = "exchange_rate.csv"; EncIn = 8; SeqLen = 96; LabelLen = 48; Horizons = @(96,192,336,720); Freq = "d" },
    @{ Name = "ILI"; Data = "custom"; RootPath = "./dataset/tslib_hf/illness/"; DataPath = "national_illness.csv"; EncIn = 7; SeqLen = 36; LabelLen = 18; Horizons = @(24,36,48,60); Freq = "w" }
)
$seeds = @(42,52,62,72,82)

$jobs = @()
foreach ($ds in $datasets) {
    foreach ($h in $ds.Horizons) {
        foreach ($seed in $seeds) {
            $jobs += [PSCustomObject]@{
                Name = $ds.Name
                Data = $ds.Data
                RootPath = $ds.RootPath
                DataPath = $ds.DataPath
                EncIn = $ds.EncIn
                SeqLen = $ds.SeqLen
                LabelLen = $ds.LabelLen
                Horizon = $h
                Seed = $seed
                Freq = $ds.Freq
            }
        }
    }
}

if ($MaxRuns -gt 0) {
    $jobs = $jobs | Select-Object -First $MaxRuns
}

$total = $jobs.Count
$idx = 0

"PPN full matrix start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content $statusFile -Encoding UTF8
"Total jobs: $total" | Add-Content $statusFile -Encoding UTF8

foreach ($job in $jobs) {
    $idx += 1
    $modelId = "ppn_{0}_h{1}_s{2}" -f $job.Name, $job.Horizon, $job.Seed

    $status = "[$idx/$total] RUN $modelId"
    $status | Tee-Object -FilePath $statusFile -Append | Out-Null

    $cmd = @(
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", $modelId,
        "--model", "PPN",
        "--data", $job.Data,
        "--root_path", $job.RootPath,
        "--data_path", $job.DataPath,
        "--features", "M",
        "--target", "OT",
        "--freq", $job.Freq,
        "--seq_len", $job.SeqLen,
        "--label_len", $job.LabelLen,
        "--pred_len", $job.Horizon,
        "--enc_in", $job.EncIn,
        "--dec_in", $job.EncIn,
        "--c_out", $job.EncIn,
        "--e_layers", "1",
        "--d_layers", "1",
        "--factor", "3",
        "--d_model", "128",
        "--d_ff", "128",
        "--train_epochs", "10",
        "--patience", "3",
        "--batch_size", "32",
        "--learning_rate", "0.0005",
        "--num_workers", "2",
        "--itr", "1",
        "--des", "ppn_full",
        "--checkpoints", "C:/tmp/ppn_full_ckpt",
        "--seed", $job.Seed,
        "--use_amp",
        "--gpu", "0"
    )

    if ($DryRun) {
        "DRY_RUN conda run -n $CondaEnv python $($cmd -join ' ')" | Add-Content $runLog -Encoding UTF8
        continue
    }

    & conda run -n $CondaEnv python @cmd 2>&1 | Tee-Object -FilePath $runLog -Append
    if ($LASTEXITCODE -ne 0) {
        "[$idx/$total] FAIL $modelId exit=$LASTEXITCODE" | Tee-Object -FilePath $statusFile -Append | Out-Null
    }
    else {
        "[$idx/$total] DONE $modelId" | Tee-Object -FilePath $statusFile -Append | Out-Null
    }
}

"PPN full matrix end: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $statusFile -Encoding UTF8
