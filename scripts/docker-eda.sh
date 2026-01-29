#!/usr/bin/env bash
# Запуск Docker (только EDA) с сохранением графиков в папку проекта reports/figures/
# Запускать из корня проекта: ./scripts/docker-eda.sh

cd "$(dirname "$0")/.."
mkdir -p reports
docker run --rm -v "$(pwd)/reports:/app/reports" hw1-recsys --eda
echo "Графики сохранены в: $(pwd)/reports/figures"
