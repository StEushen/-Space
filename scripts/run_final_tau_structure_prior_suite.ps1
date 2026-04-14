param(
    [string]$EnvName = "py39env",
    [string]$Data = "ETTh1",
    [string]$DataPath = "ETTh1.csv",
    [int]$PredLen = 96,
    [double]$Ratio = 0.1,
    [int[]]$Seeds = @(42, 52, 62),
    [int]$TeacherEpochs = 5,
    [int]$StudentEpochs = 5,
    [int]$BaselineEpochs = 5,
    [string]$Tag = "final_tau_structure_prior_v1"
)

$ErrorActionPreference = "Stop"
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Path logs -Force | Out-Null
New-Item -ItemType Directory -Path C:/tmp -Force | Out-Null

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

function Invoke-ForecastRun {
    param(
        [string]$EnvName,
        [string[]]$CmdList,
        [string]$LogPath,
        [string]$Label
    )

    & conda run --no-capture-output -n $EnvName python run.py @CmdList 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed"
    }

    $tail = Get-Content $LogPath -Tail 1500
    $mse = Get-LastMetricValue -Lines $tail -Pattern "mse:([0-9\.Ee\-]+)"
    $mae = Get-LastMetricValue -Lines $tail -Pattern "mae:([0-9\.Ee\-]+)"

    if (-not $mse -or -not $mae) {
        throw "Metric parse failed for $Label"
    }

    return [PSCustomObject]@{
        mse = [double]$mse
        mae = [double]$mae
    }
}

function Find-LatestCheckpoint {
    param([string]$RootPath)

    $ckpt = Get-ChildItem -Path $RootPath -Recurse -Filter checkpoint.pth -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $ckpt) {
        throw "No checkpoint.pth found under $RootPath"
    }

    return $ckpt.FullName
}

function New-ForecastCmd {
    param(
        [string]$ModelId,
        [string]$Model,
        [string]$Data,
        [string]$DataPath,
        [int]$PredLen,
        [double]$Ratio,
        [int]$Seed,
        [int]$Epochs,
        [int]$Patience,
        [int]$BatchSize,
        [int]$DModel,
        [int]$DFF,
        [int]$NHeads,
        [int]$ELayers,
        [int]$DLayers,
        [int]$Factor,
        [int]$PatchLen,
        [string]$Des,
        [string]$CheckpointRoot,
        [string[]]$Extra
    )

    $cmd = @(
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", $ModelId,
        "--model", $Model,
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
        "--e_layers", "$ELayers",
        "--d_layers", "$DLayers",
        "--factor", "$Factor",
        "--d_model", "$DModel",
        "--d_ff", "$DFF",
        "--n_heads", "$NHeads",
        "--train_epochs", "$Epochs",
        "--patience", "$Patience",
        "--batch_size", "$BatchSize",
        "--learning_rate", "0.0003",
        "--lradj", "cosine",
        "--num_workers", "2",
        "--itr", "1",
        "--des", $Des,
        "--checkpoints", $CheckpointRoot,
        "--seed", "$Seed",
        "--train_subset_ratio", "$Ratio",
        "--subset_seed", "$Seed",
        "--use_amp",
        "--use_gpu",
        "--gpu_type", "cuda",
        "--gpu", "0"
    )

    if ($PatchLen -gt 0) {
        $cmd += @("--patch_len", "$PatchLen")
    }

    if ($Extra) {
        $cmd += $Extra
    }

    return $cmd
}

function Add-RunRow {
    param(
        [string]$CsvPath,
        [string[]]$Columns
    )

    ($Columns -join ",") | Add-Content -Path $CsvPath -Encoding UTF8
}

function Write-SummaryCsv {
    param(
        [string]$InputCsv,
        [string]$OutputCsv,
        [string]$GroupField = "model",
        [string]$FilterField = "",
        [string]$FilterValue = ""
    )

    $rows = Import-Csv $InputCsv
    if ($FilterField -ne "") {
        $rows = $rows | Where-Object { $_.$FilterField -eq $FilterValue }
    }

    if (-not $rows) {
        throw "No rows found in $InputCsv for summary"
    }

    $summaryRows = @()
    foreach ($group in ($rows | Group-Object $GroupField)) {
        $mseValues = $group.Group | ForEach-Object { [double]$_.mse }
        $maeValues = $group.Group | ForEach-Object { [double]$_.mae }
        $count = $mseValues.Count
        $mseMean = ($mseValues | Measure-Object -Average).Average
        $maeMean = ($maeValues | Measure-Object -Average).Average
        if ($count -gt 1) {
            $mseStd = [Math]::Sqrt((($mseValues | ForEach-Object { ($_ - $mseMean) * ($_ - $mseMean) } | Measure-Object -Sum).Sum) / ($count - 1))
            $maeStd = [Math]::Sqrt((($maeValues | ForEach-Object { ($_ - $maeMean) * ($_ - $maeMean) } | Measure-Object -Sum).Sum) / ($count - 1))
        } else {
            $mseStd = 0.0
            $maeStd = 0.0
        }

        $summaryRows += [PSCustomObject]@{
            group = $group.Name
            n = $count
            mse_mean = $mseMean
            mse_std = $mseStd
            mae_mean = $maeMean
            mae_std = $maeStd
        }
    }

    $summaryRows | Sort-Object mse_mean | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8
}

# Experiment 1: TauOnly teacher + fixed tau-clock weak GRU student.
$exp1Csv = "logs/${Tag}_exp1_runs.csv"
$exp1Summary = "logs/${Tag}_exp1_summary.csv"
"stage,role,seed,model,mse,mae,model_id,checkpoint" | Set-Content $exp1Csv -Encoding UTF8

foreach ($seed in $Seeds) {
    $teacherId = "${Tag}_exp1_TauOnlyTeacher_s${seed}"
    $teacherRoot = "C:/tmp/${Tag}_exp1_TauOnlyTeacher_s${seed}"
    $teacherLog = "logs/${Tag}_exp1_teacher_tauonly_seed${seed}.log"
    Write-Host "[EXP1] TauOnly teacher seed=$seed" -ForegroundColor Cyan

    $teacherCmd = New-ForecastCmd -ModelId $teacherId -Model "TauOnly" -Data $Data -DataPath $DataPath -PredLen $PredLen -Ratio 1.0 -Seed $seed -Epochs $TeacherEpochs -Patience 2 -BatchSize 16 -DModel 128 -DFF 128 -NHeads 2 -ELayers 1 -DLayers 1 -Factor 3 -PatchLen 0 -Des $Tag -CheckpointRoot $teacherRoot -Extra @("--tauonly_anchor_delay_epochs", "1", "--tauonly_anchor_scale", "1.0", "--ppn_disable_accel_tau_loss_in_axiom_mode", "0")
    $teacherMetric = Invoke-ForecastRun -EnvName $EnvName -CmdList $teacherCmd -LogPath $teacherLog -Label "EXP1 teacher seed=$seed"
    $teacherCkpt = Find-LatestCheckpoint -RootPath $teacherRoot
    Add-RunRow -CsvPath $exp1Csv -Columns @("exp1", "teacher", "$seed", "TauOnly", "$($teacherMetric.mse)", "$($teacherMetric.mae)", $teacherId, $teacherCkpt)

    $studentId = "${Tag}_exp1_FixedTauClockGRU_s${seed}"
    $studentRoot = "C:/tmp/${Tag}_exp1_FixedTauClockGRU_s${seed}"
    $studentLog = "logs/${Tag}_exp1_student_seed${seed}.log"
    Write-Host "[EXP1] TauClockFixedGRU student seed=$seed" -ForegroundColor Cyan

    $studentCmd = New-ForecastCmd -ModelId $studentId -Model "TauClockFixedGRU" -Data $Data -DataPath $DataPath -PredLen $PredLen -Ratio $Ratio -Seed $seed -Epochs $StudentEpochs -Patience 2 -BatchSize 16 -DModel 128 -DFF 128 -NHeads 2 -ELayers 1 -DLayers 1 -Factor 3 -PatchLen 0 -Des $Tag -CheckpointRoot $studentRoot -Extra @("--tau_teacher_model", "TauOnly", "--tau_teacher_checkpoint", $teacherCkpt, "--tau_teacher_trainable", "0", "--tau_student_hidden_dim", "128", "--tau_student_num_layers", "1", "--ppn_disable_accel_tau_loss_in_axiom_mode", "0")
    $studentMetric = Invoke-ForecastRun -EnvName $EnvName -CmdList $studentCmd -LogPath $studentLog -Label "EXP1 student seed=$seed"
    Add-RunRow -CsvPath $exp1Csv -Columns @("exp1", "student", "$seed", "TauClockFixedGRU", "$($studentMetric.mse)", "$($studentMetric.mae)", $studentId, $studentRoot)
}

Write-SummaryCsv -InputCsv $exp1Csv -OutputCsv $exp1Summary -FilterField "role" -FilterValue "student"

# Experiment 2: PatchTST teacher + fixed tau-clock weak GRU student.
$exp2Csv = "logs/${Tag}_exp2_runs.csv"
$exp2Summary = "logs/${Tag}_exp2_summary.csv"
"stage,role,seed,model,mse,mae,model_id,checkpoint" | Set-Content $exp2Csv -Encoding UTF8

foreach ($seed in $Seeds) {
    $teacherId = "${Tag}_exp2_PatchTSTTeacher_s${seed}"
    $teacherRoot = "C:/tmp/${Tag}_exp2_PatchTSTTeacher_s${seed}"
    $teacherLog = "logs/${Tag}_exp2_teacher_patchtst_seed${seed}.log"
    Write-Host "[EXP2] PatchTST teacher seed=$seed" -ForegroundColor Cyan

    $teacherCmd = New-ForecastCmd -ModelId $teacherId -Model "PatchTST" -Data $Data -DataPath $DataPath -PredLen $PredLen -Ratio 1.0 -Seed $seed -Epochs $TeacherEpochs -Patience 2 -BatchSize 32 -DModel 512 -DFF 2048 -NHeads 2 -ELayers 1 -DLayers 1 -Factor 3 -PatchLen 16 -Des $Tag -CheckpointRoot $teacherRoot -Extra @()
    $teacherMetric = Invoke-ForecastRun -EnvName $EnvName -CmdList $teacherCmd -LogPath $teacherLog -Label "EXP2 teacher seed=$seed"
    $teacherCkpt = Find-LatestCheckpoint -RootPath $teacherRoot
    Add-RunRow -CsvPath $exp2Csv -Columns @("exp2", "teacher", "$seed", "PatchTST", "$($teacherMetric.mse)", "$($teacherMetric.mae)", $teacherId, $teacherCkpt)

    $studentId = "${Tag}_exp2_FixedTauClockGRU_s${seed}"
    $studentRoot = "C:/tmp/${Tag}_exp2_FixedTauClockGRU_s${seed}"
    $studentLog = "logs/${Tag}_exp2_student_seed${seed}.log"
    Write-Host "[EXP2] TauClockFixedGRU student with PatchTST teacher seed=$seed" -ForegroundColor Cyan

    $studentCmd = New-ForecastCmd -ModelId $studentId -Model "TauClockFixedGRU" -Data $Data -DataPath $DataPath -PredLen $PredLen -Ratio $Ratio -Seed $seed -Epochs $StudentEpochs -Patience 2 -BatchSize 32 -DModel 512 -DFF 2048 -NHeads 2 -ELayers 1 -DLayers 1 -Factor 3 -PatchLen 16 -Des $Tag -CheckpointRoot $studentRoot -Extra @("--tau_teacher_model", "PatchTST", "--tau_teacher_checkpoint", $teacherCkpt, "--tau_teacher_trainable", "0", "--tau_student_hidden_dim", "128", "--tau_student_num_layers", "1", "--ppn_disable_accel_tau_loss_in_axiom_mode", "0")
    $studentMetric = Invoke-ForecastRun -EnvName $EnvName -CmdList $studentCmd -LogPath $studentLog -Label "EXP2 student seed=$seed"
    Add-RunRow -CsvPath $exp2Csv -Columns @("exp2", "student", "$seed", "TauClockFixedGRU", "$($studentMetric.mse)", "$($studentMetric.mae)", $studentId, $studentRoot)
}

Write-SummaryCsv -InputCsv $exp2Csv -OutputCsv $exp2Summary -FilterField "role" -FilterValue "student"

# Experiment 3: broader baseline comparison under the same few-shot setup.
$exp3Csv = "logs/${Tag}_exp3_runs.csv"
$exp3Summary = "logs/${Tag}_exp3_summary.csv"
"stage,seed,model,mse,mae,model_id" | Set-Content $exp3Csv -Encoding UTF8

$baselineRuns = @(
    @{ Name = "TauOnly"; Model = "TauOnly"; DModel = 128; DFF = 128; NHeads = 2; ELayers = 1; DLayers = 1; Factor = 3; PatchLen = 0; BatchSize = 16; Extra = @("--tauonly_anchor_delay_epochs", "1", "--tauonly_anchor_scale", "1.0", "--ppn_disable_accel_tau_loss_in_axiom_mode", "0") },
    @{ Name = "TauFusion_PatchTST"; Model = "TauFusion"; DModel = 512; DFF = 2048; NHeads = 2; ELayers = 1; DLayers = 1; Factor = 3; PatchLen = 16; BatchSize = 32; Extra = @("--tau_base_model", "PatchTST", "--tau_fuse_alpha", "0.3", "--ppn_disable_accel_tau_loss_in_axiom_mode", "0") },
    @{ Name = "PatchTST"; Model = "PatchTST"; DModel = 512; DFF = 2048; NHeads = 2; ELayers = 1; DLayers = 1; Factor = 3; PatchLen = 16; BatchSize = 32; Extra = @() },
    @{ Name = "DLinear"; Model = "DLinear"; DModel = 512; DFF = 2048; NHeads = 8; ELayers = 2; DLayers = 1; Factor = 3; PatchLen = 0; BatchSize = 32; Extra = @() },
    @{ Name = "iTransformer"; Model = "iTransformer"; DModel = 128; DFF = 128; NHeads = 8; ELayers = 2; DLayers = 1; Factor = 3; PatchLen = 0; BatchSize = 32; Extra = @() }
)

foreach ($seed in $Seeds) {
    foreach ($run in $baselineRuns) {
        $modelId = "${Tag}_exp3_$($run.Name)_s${seed}"
        $runRoot = "C:/tmp/${Tag}_exp3_$($run.Name)_s${seed}"
        $runLog = "logs/${Tag}_exp3_$($run.Name)_seed${seed}.log"
        Write-Host "[EXP3] model=$($run.Name) seed=$seed" -ForegroundColor Cyan

        $baselineCmd = New-ForecastCmd -ModelId $modelId -Model $run.Model -Data $Data -DataPath $DataPath -PredLen $PredLen -Ratio $Ratio -Seed $seed -Epochs $BaselineEpochs -Patience 2 -BatchSize $run.BatchSize -DModel $run.DModel -DFF $run.DFF -NHeads $run.NHeads -ELayers $run.ELayers -DLayers $run.DLayers -Factor $run.Factor -PatchLen $run.PatchLen -Des $Tag -CheckpointRoot $runRoot -Extra $run.Extra
        $metric = Invoke-ForecastRun -EnvName $EnvName -CmdList $baselineCmd -LogPath $runLog -Label "EXP3 $($run.Name) seed=$seed"
        Add-RunRow -CsvPath $exp3Csv -Columns @("exp3", "$seed", $run.Name, "$($metric.mse)", "$($metric.mae)", $modelId)
    }
}

Write-SummaryCsv -InputCsv $exp3Csv -OutputCsv $exp3Summary

$overallCsv = "logs/${Tag}_overall_summary.csv"
"stage,group,n,mse_mean,mse_std,mae_mean,mae_std" | Set-Content $overallCsv -Encoding UTF8

foreach ($path in @($exp1Summary, $exp2Summary, $exp3Summary)) {
    $rows = Import-Csv $path
    foreach ($row in $rows) {
        $stage = if ($path -like "*exp1*") { "exp1" } elseif ($path -like "*exp2*") { "exp2" } else { "exp3" }
        "{0},{1},{2},{3},{4},{5},{6}" -f $stage, $row.group, $row.n, $row.mse_mean, $row.mse_std, $row.mae_mean, $row.mae_std | Add-Content $overallCsv -Encoding UTF8
    }
}

Write-Host "Run files:" -ForegroundColor Green
Write-Host "  $exp1Csv" -ForegroundColor Green
Write-Host "  $exp2Csv" -ForegroundColor Green
Write-Host "  $exp3Csv" -ForegroundColor Green
Write-Host "Summary files:" -ForegroundColor Green
Write-Host "  $exp1Summary" -ForegroundColor Green
Write-Host "  $exp2Summary" -ForegroundColor Green
Write-Host "  $exp3Summary" -ForegroundColor Green
Write-Host "  $overallCsv" -ForegroundColor Green
