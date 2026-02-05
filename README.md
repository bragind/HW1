# Гибридная система рекомендаций книг

Комплексная гибридная система рекомендаций книг на датасете [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k): классические подходы (Popularity, Content-Based, Item-CF, SVD) объединены с опциональной нейросетевой моделью Two-Tower.

## Цель проекта

Разработать гибридную систему рекомендаций книг

## Структура проекта

```
HW1/
├── src/
│   └── recsys/
│       ├── __init__.py
│       ├── config.py       # Пути к данным, гиперпараметры, веса гибрида
│       ├── data.py         # Загрузка CSV и сборка профилей книг (теги)
│       ├── eda.py          # EDA: распределение оценок, активность, популярность, теги
│       ├── features.py     # Расширенные признаки пользователей и книг, сегментация (cold/regular/active)
│       ├── models.py       # Модели: Popularity, Content-Based (TF-IDF), Item-CF, SVD
│       ├── hybrid.py      # Гибрид: взвешенное объединение моделей, пул кандидатов, баланс
│       ├── metrics.py      # Метрики: Precision@K, Recall@K, nDCG@K
│       ├── evaluation.py  # Оценка моделей и гибрида, анализ по сегментам, оптимизация весов
│       ├── neural.py       # (опционально) Two-Tower модель для продвинутой части
│       ├── pipeline.py    # Сквозной пайплайн: данные → модели → гибрид → оценка
│       ├── report.py      # Генерация отчёта REPORT.md
│       └── cli.py         # CLI: run_all, run_eda_only, run_report
├── data/
│   ├── raw/                # исходные данные (ratings.csv, books.csv, tags.csv, book_tags.csv)
│   └── processed/          # обработанные данные/артефакты (при необходимости)
├── reports/
│   ├── figures/            # графики EDA (ratings_distribution.png, user_activity.png, book_popularity.png, top_tags.png)
│   └── REPORT.md           # отчёт с выводами (формируется командой python main.py --report и при запуске контейнера)
├── logs/                   # логи запусков при python main.py --log (run_YYYY-MM-DD_HH-MM-SS.log)
├── scripts/                # скрипты для Docker: docker-report.ps1, docker-eda.ps1, docker-report.sh, docker-eda.sh
├── main.py                 # Точка входа: python main.py [--eda | --report | --log]
├── requirements.txt        # Зависимости Python
├── README.md               # Этот файл
├── Dockerfile
├── notebooks/              # Директория для ноутбуков (EDA, прототипы)
│   └── README.md
└── (остальные файлы проекта)
```

## Запуск
### Вариант 1: виртуальное окружение (рекомендуется)

Из корня проекта:

**Windows (PowerShell или cmd):**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

После этого папка `venv/` будет содержать изолированное окружение; зависимости установятся только в него. Выйти из окружения: `deactivate`.


### Вариант 2: Docker

**Требование:** установленный [Docker](https://docs.docker.com/get-docker/).

Из корня проекта (где лежит `Dockerfile`):

**1. Собрать образ:**
```bash
docker build -t hw1-recsys .
```

**2. Запустить контейнер:**

**Команда для EDA + отчёт (результат в папке проекта):**

- **PowerShell (Windows):**
```powershell
mkdir -Force reports
docker run --rm -v "${PWD}/reports:/app/reports" hw1-recsys --report
```
- **cmd (Windows):**
```cmd
mkdir reports
docker run --rm -v "%CD%\reports:/app/reports" hw1-recsys --report
```
- **Linux / macOS:**
```bash
mkdir -p reports
docker run --rm -v "$(pwd)/reports:/app/reports" hw1-recsys --report
```

После выполнения отчёт будет в `reports/REPORT.md`, графики — в `reports/figures/`.

## Логика решения

1. **Данные:** ratings (user_id, book_id, rating), books (метаданные), tags и book_tags (теги книг). Один раз делается разбиение train/test по строкам ratings.

2. **Popularity:** по обучающей выборке считаются средний рейтинг и число оценок по каждой книге; топ-N — книги с числом оценок ≥ порога, отсортированные по среднему рейтингу.

3. **Content-Based:** для каждой книги строится текстовый профиль (original_title + теги), векторизация TF-IDF, поиск N ближайших книг по косинусной близости. Рекомендации для пользователя — похожие на последнюю оценённую им книгу (из train).

4. **Item-Based CF:** матрица взаимодействий user×book (неявный feedback: 1 при rating ≥ 4), косинусная схожесть между книгами по столбцам. Предсказание — взвешенное среднее «оценок» пользователя по похожим книгам; топ-N — книги с максимальным предсказанием среди непрочитанных.

5. **SVD:** библиотека Surprise, обучение на train. Топ-N — книги с наибольшим предсказанным рейтингом. RMSE считается на отложенной тестовой выборке.

6. **Оценка:** релевантные книги — с оценкой ≥ 4 в тесте. По пользователям из теста, у которых есть релевантные книги, считаются средние Precision@K, Recall@K, nDCG@K для каждой модели.

7. **Гибрид:** объединение списков SVD и Popularity с удалением дубликатов для устойчивости и разнообразия.

## Зависимости

- Python 3.8+
- pandas, numpy, scipy, scikit-learn, scikit-surprise, matplotlib

См. `requirements.txt`.

## Соответствие этапам задания

- **Этап 1 (EDA):** загрузка ratings, books, tags, book_tags; распределение оценок; активность пользователей; популярность книг; топ тегов; проблемы данных (разреженность, смещение популярности). Запуск: `python main.py --eda`.
- **Этапы 2–4:** Popularity (топ-N по среднему рейтингу с порогом), контентная модель (профиль = original_title + теги, TF-IDF, get_similar_books), Item-Based CF (матрица user×book, схожесть книг, предсказание по K соседям; сложность и оптимизации описаны в models.py), SVD (Surprise, train/test, RMSE, get_recommendations).
- **Этап 5:** Precision@K, Recall@K, nDCG@K на отложенной тестовой выборке; сводная таблица по всем моделям.
- **Этап 6:** гибридная стратегия (холодный старт: контент для новых книг, SVD + Popularity для известных); выводы и идеи улучшения. Отчёт: `python main.py --report` → `reports/REPORT.md`.

## Ноутбук-прототип

Исходный ноутбук `book.ipynb` находится в `notebooks/`. Запуск проекта выполняется через `main.py`.
