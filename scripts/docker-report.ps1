# Запуск Docker с сохранением отчёта и графиков в папку проекта reports/
# Запускать из корня проекта: .\scripts\docker-report.ps1

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$reportsPath = Join-Path $projectRoot "reports"
New-Item -ItemType Directory -Force -Path $reportsPath | Out-Null
docker run --rm -v "${reportsPath}:/app/reports" hw1-recsys --report
Write-Host "Отчёт и графики сохранены в: $reportsPath"
