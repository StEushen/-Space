param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int]$Seed = 42,
    [double[]]$Ratios = @(0.1, 0.05),
    [int]$PretrainEpochs = 10,
    [int]$FinetuneEpochs = 3,
    [int]$Patience = 2,
    [double]$LambdaTau = 0.05,
    [double]$ResidualScale = 0.1,
    [int]$WarmupEpochs = 2,
    [string]$StageTag = "two_stage"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"
New-Item -ItemType Directory -Path logs -Force | Out-Null
New-Item -ItemType Directory -Path C:/tmp/ppn_pretrain_ckpt -Force | Out-Null

$baseArgs = @(
    "--task_name", "long_term_forecast",
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
    "--batch_size", "32",
    "--learning_rate", "0.0003",
    "--lradj", "cosine",
    "--num_workers", "2",
    "--itr", "1",
    "--des", $StageTag,
    "--use_amp",
    "--use_gpu",
    "--gpu_type", "cuda",
    "--gpu", "0",
    "--ppn_use_tau_loss", "1",
    "--ppn_lambda_tau", "$LambdaTau"
)

$pretrainModelId = "ppn_pretrain_${Data}_h${PredLen}_s${Seed}"
$pretrainCmd = @(
    "conda", "run", "-n", $EnvName, "python", "run.py",
    "--is_training", "1",
    "--model_id", $pretrainModelId,
    "--train_epochs", "$PretrainEpochs",
    "--patience", "3",
    "--checkpoints", "C:/tmp/ppn_pretrain_ckpt",
    "--seed", "$Seed",
    "--train_subset_ratio", "1.0"
) + $baseArgs

Write-Host "[STAGE 1] pretrain: $pretrainModelId" -ForegroundColor Cyan
& $pretrainCmd[0] $pretrainCmd[1..($pretrainCmd.Length - 1)]
if ($LASTEXITCODE -ne 0) { throw "Pretrain stage failed with exit code $LASTEXITCODE" }

$pretrainRoot = "C:/tmp/ppn_pretrain_ckpt"
$pretrainCkpt = Get-ChildItem -Path $pretrainRoot -Recurse -Filter checkpoint.pth | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $pretrainCkpt) { throw "No checkpoint found under $pretrainRoot" }
Write-Host "[STAGE 1] checkpoint: $($pretrainCkpt.FullName)" -ForegroundColor Green

foreach ($ratio in $Ratios) {
    $ratioTag = ("{0}" -f $ratio).Replace(".", "p")
    $ftModelId = "ppn_ft_${Data}_r${ratioTag}_h${PredLen}_s${Seed}"
    $logTag = "$StageTag`_r$ratioTag"

    Write-Host "[STAGE 2] finetune ratio=$ratio model_id=$ftModelId" -ForegroundColor Cyan
    $ftCmd = @(
        "conda", "run", "-n", $EnvName, "python", "run.py",
        "--is_training", "1",
        "--model_id", $ftModelId,
        "--train_epochs", "$FinetuneEpochs",
        "--patience", "$Patience",
        "--checkpoints", "C:/tmp/ppn_finetune_ckpt",
        "--seed", "$Seed",
        "--train_subset_ratio", "$ratio",
        "--subset_seed", "$Seed",
        "--load_checkpoint", $pretrainCkpt.FullName,
        "--ppn_use_horizon_residual", "1",
        "--ppn_horizon_residual_scale", "$ResidualScale",
        "--ppn_use_horizon_residual_warmup", "1",
        "--ppn_horizon_residual_warmup_epochs", "$WarmupEpochs"
    ) + $baseArgs

    & $ftCmd[0] $ftCmd[1..($ftCmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "Finetune stage failed with exit code $LASTEXITCODE" }
}

Write-Host "Two-stage run finished." -ForegroundColor Green