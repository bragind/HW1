#!/usr/bin/env bash
# Запуск Docker: EDA + оценка моделей и гибрида + генерация отчёта reports/REPORT.md
# Запускать из корня проекта: ./scripts/docker-report.sh
# Данные должны быть в data/raw/ (ratings.csv, books.csv, tags.csv, book_tags.csv)

cd "$(dirname "$0")/.."
mkdir -p data/raw reports
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/reports:/app/reports" hw1-recsys --report
echo "Отчёт и графики сохранены в: $(pwd)/reports"
