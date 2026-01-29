#!/usr/bin/env bash
# Запуск Docker с сохранением отчёта и графиков в папку проекта reports/
# Запускать из корня проекта: ./scripts/docker-report.sh

cd "$(dirname "$0")/.."
mkdir -p reports
docker run --rm -v "$(pwd)/reports:/app/reports" hw1-recsys --report
echo "Отчёт и графики сохранены в: $(pwd)/reports"
