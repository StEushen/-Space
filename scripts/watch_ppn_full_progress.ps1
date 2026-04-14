param(
    [string]$StatusFile = "logs/ppn_full_status.txt",
    [string]$RunLog = "logs/ppn_full_runs.log",
    [string]$OutputFile = "logs/ppn_full_dashboard.txt",
    [int]$RefreshSeconds = 15,
    [ValidateSet("zh", "en")]
    [string]$Language = "zh"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

New-Item -ItemType Directory -Force -Path (Split-Path $OutputFile) | Out-Null

function Get-Labels {
    param([string]$Lang)

    if ($Lang -eq "en") {
        return [ordered]@{
            Title = "PPN Full Matrix Dashboard"
            Updated = "Updated"
            Done = "Done"
            Failed = "Failed"
            Running = "Running"
            Remaining = "Remaining"
            Progress = "Progress"
            CurrentTask = "Current Task"
            Metrics = "Latest Metrics"
            Events = "Recent Events"
            Epoch = "Epoch"
            Train = "Train"
            Val = "Val"
            Test = "Test"
            MSE = "MSE"
            MAE = "MAE"
            RecentLog = "Recent Log Tail"
            NoTask = "No active run line found."
            NoLog = "No run log yet."
            NoMetrics = "No final metrics detected yet."
            NoEvents = "No significant events detected yet."
            Separator = "=" * 78
            Section = "-" * 78
        }
    }

    return [ordered]@{
        Title = "PPN 全量矩阵看板"
        Updated = "更新时间"
        Done = "已完成"
        Failed = "失败"
        Running = "运行中"
        Remaining = "剩余"
        Progress = "进度"
        CurrentTask = "当前任务"
        Metrics = "最新指标"
        Events = "最近事件"
        Epoch = "轮次"
        Train = "训练"
        Val = "验证"
        Test = "测试"
        MSE = "均方误差"
        MAE = "平均绝对误差"
        RecentLog = "最近日志"
        NoTask = "未发现正在执行的任务。"
        NoLog = "暂无运行日志。"
        NoMetrics = "尚未检测到最终指标。"
        NoEvents = "尚未检测到关键事件。"
        Separator = "=" * 78
        Section = "-" * 78
    }
}

function Format-Bar {
    param(
        [double]$Percent,
        [int]$Width = 30
    )

    $filled = [Math]::Round(($Percent / 100) * $Width)
    $filled = [Math]::Max(0, [Math]::Min($Width, $filled))
    return ("█" * $filled) + ("░" * ($Width - $filled))
}

function Get-Stats {
    param([string]$Text)

    $done = ([regex]::Matches($Text, '\] DONE ')).Count
    $fail = ([regex]::Matches($Text, '\] FAIL ')).Count
    $run = ([regex]::Matches($Text, '\] RUN ')).Count
    $total = 180
    return [PSCustomObject]@{
        Done = $done
        Fail = $fail
        Running = $run
        Total = $total
    }
}

function Parse-TaskInfo {
    param([string]$Line)

    if (-not $Line) { return $null }
    if ($Line -match '^\[(\d+)\/(\d+)\] RUN (.+)$') {
        $runName = $Matches[3]
        $dataset = $null
        $horizon = $null
        $seed = $null
        if ($runName -match '^ppn_(.+?)_h(\d+)_s(\d+)$') {
            $dataset = $Matches[1]
            $horizon = $Matches[2]
            $seed = $Matches[3]
        }

        return [PSCustomObject]@{
            Raw = $Line
            RunName = $runName
            Dataset = $dataset
            Horizon = $horizon
            Seed = $seed
            Index = $Matches[1]
            Total = $Matches[2]
        }
    }

    return [PSCustomObject]@{
        Raw = $Line
        RunName = $null
        Dataset = $null
        Horizon = $null
        Seed = $null
        Index = $null
        Total = $null
    }
}

function Get-SignificantEvents {
    param(
        [string[]]$Lines,
        [int]$MaxEvents = 8
    )

    $events = New-Object System.Collections.Generic.List[string]
    foreach ($raw in $Lines) {
        $line = ($raw -replace "^\s+", "")
        if ($line -match '^iters:\s*\d+') { continue }
        if ($line -match '^speed:') { continue }

        if ($line -match '^Epoch:\s*(\d+)\s+cost time:\s*([0-9\.]+)') {
            $events.Add("Epoch $($Matches[1]) finished in $([math]::Round([double]$Matches[2], 2))s")
            continue
        }
        if ($line -match '^Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([0-9eE+\-.]+)\s*Vali Loss:\s*([0-9eE+\-.]+)\s*Test Loss:\s*([0-9eE+\-.]+)') {
            $events.Add("Epoch $($Matches[1]) | Train $($Matches[3]) | Val $($Matches[4]) | Test $($Matches[5])")
            continue
        }
        if ($line -match '^EarlyStopping counter:\s*(\d+) out of (\d+)') {
            $events.Add("EarlyStopping $($Matches[1]) / $($Matches[2])")
            continue
        }
        if ($line -match '^Updating learning rate to\s+(.+)$') {
            $events.Add("Learning rate -> $($Matches[1])")
            continue
        }
        if ($line -match '^Early stopping$') {
            $events.Add("Early stopping triggered")
            continue
        }
        if ($line -match '^>>>>>>>testing\s*:\s*(.+)$') {
            $events.Add("Testing: $($Matches[1])")
            continue
        }
        if ($line -match '^mse:([0-9eE+\-.]+),\s*mae:([0-9eE+\-.]+),\s*dtw:(.+)$') {
            $events.Add("Final metrics | MSE $($Matches[1]) | MAE $($Matches[2])")
            continue
        }
        if ($line -match '^test shape:\s*\((.+)\)\s*\((.+)\)$') {
            $events.Add("Test shape $($Matches[1])")
            continue
        }
    }

    if ($events.Count -eq 0) {
        return @()
    }

    return $events | Select-Object -Last $MaxEvents
}

function Format-FieldLine {
    param(
        [string]$Label,
        [string]$Value,
        [int]$Width = 13
    )

    return ("{0,-$Width}: {1}" -f $Label, $Value)
}

while ($true) {
    $now = Get-Date
    $labels = Get-Labels -Lang $Language
    $statusText = if (Test-Path $StatusFile) { Get-Content $StatusFile -Raw -ErrorAction SilentlyContinue } else { "" }
    $runText = if (Test-Path $RunLog) { Get-Content $RunLog -Tail 24 -ErrorAction SilentlyContinue | Out-String } else { "" }
    $runLines = if (Test-Path $RunLog) { Get-Content $RunLog -Tail 60 -ErrorAction SilentlyContinue } else { @() }
    $stats = Get-Stats -Text $statusText

    $currentLine = ($statusText -split "`r?`n" | Where-Object { $_ -match '^\[\d+/\d+\] RUN ' } | Select-Object -Last 1)
    $taskInfo = Parse-TaskInfo -Line $currentLine
    $latestRun = if ($currentLine) { $currentLine } else { $labels.NoTask }

    $remaining = [Math]::Max(0, $stats.Total - $stats.Done - $stats.Fail)
    $progress = if ($stats.Total -gt 0) { [Math]::Round((($stats.Done + $stats.Fail) / $stats.Total) * 100, 1) } else { 0 }
    $bar = Format-Bar -Percent $progress -Width 30
    $events = Get-SignificantEvents -Lines $runLines -MaxEvents 8

    $lines = @()
    $lines += $labels.Separator
    $lines += $labels.Title
    $lines += $labels.Separator
    $lines += ("{0}: {1}" -f $labels.Updated, $now.ToString("yyyy-MM-dd HH:mm:ss"))
    $lines += ("{0}: {1}" -f $labels.Done, $stats.Done)
    $lines += ("{0}: {1}" -f $labels.Failed, $stats.Fail)
    $lines += ("{0}: {1}" -f $labels.Running, $stats.Running)
    $lines += ("{0}: {1}" -f $labels.Remaining, $remaining)
    $lines += ("{0}: [{1}] {2}%" -f $labels.Progress, $bar, $progress)
    $lines += ""
    $lines += $labels.Section
    $lines += $labels.CurrentTask
    $lines += $labels.Section
    if ($taskInfo -and $taskInfo.RunName) {
        $lines += ("Run       : {0}" -f $taskInfo.RunName)
        if ($taskInfo.Dataset) { $lines += ("Dataset   : {0}" -f $taskInfo.Dataset) }
        if ($taskInfo.Horizon) { $lines += ("Horizon   : {0}" -f $taskInfo.Horizon) }
        if ($taskInfo.Seed) { $lines += ("Seed      : {0}" -f $taskInfo.Seed) }
        $lines += ("Raw line  : {0}" -f $latestRun)
    }
    else {
        $lines += $latestRun
    }
    $lines += ""
    $lines += $labels.Section
    $lines += $labels.Metrics
    $lines += $labels.Section
    $epochLine = $runLines | Where-Object { $_ -match '^Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([0-9eE+\-.]+)\s*Vali Loss:\s*([0-9eE+\-.]+)\s*Test Loss:\s*([0-9eE+\-.]+)' } | Select-Object -Last 1
    if ($epochLine -and $epochLine -match '^Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([0-9eE+\-.]+)\s*Vali Loss:\s*([0-9eE+\-.]+)\s*Test Loss:\s*([0-9eE+\-.]+)') {
        $lines += ("{0}" -f (Format-FieldLine -Label $labels.Epoch -Value $Matches[1]))
        $lines += ("Steps      : {0}" -f $Matches[2])
        $lines += ("{0}      : {1}" -f $labels.Train, $Matches[3])
        $lines += ("{0}        : {1}" -f $labels.Val, $Matches[4])
        $lines += ("{0}       : {1}" -f $labels.Test, $Matches[5])
    }
    else {
        $lines += $labels.NoMetrics
    }

    $lines += ""
    $lines += $labels.Section
    $lines += $labels.Events
    $lines += $labels.Section
    if ($events.Count -gt 0) {
        foreach ($event in $events) {
            $lines += ("- {0}" -f $event)
        }
    }
    else {
        $lines += $labels.NoEvents
    }

    $lines -join [Environment]::NewLine | Set-Content -Path $OutputFile -Encoding UTF8
    Start-Sleep -Seconds $RefreshSeconds
}
