param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [int]$Seed = 42
)

$rootPath = "./dataset/ETT-small/"
$modelId = "ppn_opt_{0}_h{1}_s{2}" -f $Data, $PredLen, $Seed

$cmd = @(
    "conda", "run", "-n", $EnvName, "python", "run.py",
    "--task_name", "long_term_forecast",
    "--is_training", "1",
    "--model_id", $modelId,
    "--model", "PPN",
    "--data", $Data,
    "--root_path", $rootPath,
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
    "--train_epochs", "30",
    "--patience", "10",
    "--batch_size", "32",
    "--learning_rate", "0.0003",
    "--lradj", "cosine",
    "--num_workers", "2",
    "--itr", "1",
    "--des", "latest_ppn_opt",
    "--checkpoints", "C:/tmp/ppn_latest_ckpt",
    "--seed", "$Seed",
    "--ppn_use_tau_loss", "1",
    "--ppn_lambda_tau", "0.2",
    "--use_amp",
    "--gpu", "0"
)

Write-Host "Running optimized PPN command:" -ForegroundColor Cyan
Write-Host ($cmd -join " ")

& $cmd[0] $cmd[1..($cmd.Length - 1)]
