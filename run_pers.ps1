$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "logs"

Set-Location -LiteralPath $root
New-Item -ItemType Directory -Path $logs -Force | Out-Null

foreach ($speaker in @("F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05")) {
    foreach ($minutes in @(0, 1, 2, 5)) {
        $log = Join-Path $logs "step7_pers_${speaker}_${minutes}min_b4_a2.log"
        $adapter = "outputs/m1_lora_r32/$speaker/adapter"
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        & $python "train_lora.py" --test_speaker $speaker --adapt_minutes $minutes `
            --init_adapter $adapter --epochs 10 --batch_size 4 --grad_accum 2 *> $log
        $exitCode = $LASTEXITCODE
        $watch.Stop()
        "COMMAND=python train_lora.py --test_speaker $speaker --adapt_minutes $minutes --init_adapter $adapter --epochs 10 --batch_size 4 --grad_accum 2" |
            Add-Content -LiteralPath $log
        "EXIT_CODE=$exitCode" | Add-Content -LiteralPath $log
        "ELAPSED_SECONDS=$($watch.Elapsed.TotalSeconds)" | Add-Content -LiteralPath $log
        Write-Output "$speaker ${minutes}min exit=$exitCode elapsed=$([math]::Round($watch.Elapsed.TotalSeconds, 1))s log=$log"
    }
}
