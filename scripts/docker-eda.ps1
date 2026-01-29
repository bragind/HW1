# Запуск Docker (только EDA) с сохранением графиков в папку проекта reports/figures/
# Запускать из корня проекта: .\scripts\docker-eda.ps1

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$reportsPath = Join-Path $projectRoot "reports"
New-Item -ItemType Directory -Force -Path $reportsPath | Out-Null
docker run --rm -v "${reportsPath}:/app/reports" hw1-recsys --eda
Write-Host "Графики сохранены в: $reportsPath\figures"
