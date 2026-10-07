$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "logs"

Set-Location -LiteralPath $root
New-Item -ItemType Directory -Path $logs -Force | Out-Null

foreach ($speaker in @("F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05")) {
    $log = Join-Path $logs "step6b_strict_$($speaker)_b4_a4.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    & $python "train_lora.py" --test_speaker $speaker --drop_test_texts --name "m1_lora_r32_strict" --batch_size 4 --grad_accum 4 *> $log
    $exitCode = $LASTEXITCODE
    $watch.Stop()
    "COMMAND=python train_lora.py --test_speaker $speaker --drop_test_texts --name m1_lora_r32_strict --batch_size 4 --grad_accum 4" |
        Add-Content -LiteralPath $log
    "EXIT_CODE=$exitCode" | Add-Content -LiteralPath $log
    "ELAPSED_SECONDS=$($watch.Elapsed.TotalSeconds)" | Add-Content -LiteralPath $log
    Write-Output "$speaker exit=$exitCode elapsed=$([math]::Round($watch.Elapsed.TotalSeconds, 1))s log=$log"
}
