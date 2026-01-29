# Запуск Docker: EDA + оценка моделей и гибрида + генерация отчёта reports/REPORT.md
# Запускать из корня проекта: .\scripts\docker-report.ps1
# Данные должны быть в data/raw/ (ratings.csv, books.csv, tags.csv, book_tags.csv)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$dataPath = Join-Path $projectRoot "data"
$reportsPath = Join-Path $projectRoot "reports"
New-Item -ItemType Directory -Force -Path $reportsPath | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $dataPath "raw") | Out-Null
docker run --rm -v "${dataPath}:/app/data" -v "${reportsPath}:/app/reports" hw1-recsys --report
Write-Host "Report and figures saved to: $reportsPath"
