param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputZip = Join-Path $ProjectRoot "data\raw\crime.zip"

if (-not (Test-Path -LiteralPath $InputZip)) {
    throw "입력 파일이 없습니다: $InputZip"
}

$Scripts = @(
    "01_prepare_data.py",
    "02_extract_metrics.py",
    "03_analyze_timeseries.py",
    "04_create_figures.py",
    "05_extract_examples.py",
    "06_robustness.py",
    "07_create_robustness_figures.py",
    "08_qc_context.py",
    "09_supplementary_analysis.py"
)

foreach ($ScriptName in $Scripts) {
    $ScriptPath = Join-Path $ProjectRoot "src\$ScriptName"
    Write-Host "실행 중: $ScriptName"
    & $PythonExe $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "$ScriptName 실행 실패: 종료 코드 $LASTEXITCODE"
    }
}

Write-Host "분석 완료: results 폴더를 확인하십시오."

