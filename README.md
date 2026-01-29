# Книжный рекомендательный сервис (Goodbooks-10k)

Прототип системы рекомендаций книг на датасете [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k).

## Структура проекта

```
HW1/
├── src/
│   └── recsys/
│       ├── __init__.py
│       ├── config.py       # Пути к данным и гиперпараметры (пороги, K, размер теста и т.д.)
│       ├── data.py         # Загрузка CSV (ratings, books, tags, book_tags) и сборка профилей книг
│       ├── eda.py          # Этап 1: EDA — распределение оценок, активность пользователей, популярность книг, теги
│       ├── models.py       # Модели: Popularity, Content-Based (TF-IDF), Item-CF, SVD
│       ├── metrics.py      # Метрики: Precision@K, Recall@K, nDCG@K
│       ├── evaluation.py   # Пайплайн обучения и оценки (train/test, RMSE, сводная таблица)
│       ├── report.py       # Генерация отчёта REPORT.md с выводами по этапам
│       └── cli.py          # CLI: run_all, run_eda_only, run_report
├── data/
│   ├── raw/                # исходные данные (ratings.csv, books.csv, tags.csv, book_tags.csv)
│   └── processed/          # обработанные данные/артефакты (при необходимости)
├── reports/
│   ├── figures/            # графики EDA (ratings_distribution.png, user_activity.png, book_popularity.png, top_tags.png)
│   └── REPORT.md           # отчёт с выводами (формируется командой python main.py --report)
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

### Вариант 2: без виртуального окружения

Из корня проекта:

```bash
pip install -r requirements.txt
python main.py
```

### Режимы запуска

| Команда | Действие |
|---------|----------|
| `python main.py` | Полный эксперимент: загрузка данных, train/test, обучение моделей, RMSE для SVD, Precision@K / Recall@K / nDCG@K, вывод в консоль |
| `python main.py --eda` | Только EDA (Этап 1): графики сохраняются в `reports/figures/` |
| `python main.py --report` | EDA + эксперимент + формирование отчёта `reports/REPORT.md` с выводами по всем этапам |
| `python main.py --log` | То же, что без флага, но весь вывод дополнительно пишется в файл в каталоге `logs/` |
| `python main.py --report --log` | Отчёт + сохранение лога запуска в `logs/run_YYYY-MM-DD_HH-MM-SS.log` |

**Просмотр логов:** при запуске с флагом `--log` весь вывод (консольный) дублируется в файл `logs/run_<дата>_<время>.log`. Каталог `logs/` создаётся автоматически; после завершения в консоли выводится путь к сохранённому логу.

Скрипт отчёта формирует документ с результатами EDA, сводной таблицей метрик, гибридной стратегией и выводами (какая модель лучше, сильные/слабые стороны, идеи улучшения).

### Вариант 3: Docker

**Требование:** установленный [Docker](https://docs.docker.com/get-docker/).

Из корня проекта (где лежит `Dockerfile`):

**1. Собрать образ:**
```bash
docker build -t hw1-recsys .
```

**2. Запустить контейнер:**

Без монтирования отчёт и графики остаются **внутри контейнера** и после остановки контейнера недоступны. Чтобы сохранять их **в папку проекта** на своей машине, используйте скрипты или команды с `-v` ниже.

| Что запустить | Команда |
|---------------|---------|
| Полный эксперимент (вывод только в консоль) | `docker run --rm hw1-recsys` |
| Только EDA (графики в контейнере) | `docker run --rm hw1-recsys --eda` |
| EDA + отчёт (в контейнере) | `docker run --rm hw1-recsys --report` |

**3. Сохранить отчёт и графики в папку проекта (рекомендуется для --report и --eda):**

**Вариант А — скрипты (удобнее):** из корня проекта:

- **PowerShell (Windows):**
```powershell
.\scripts\docker-report.ps1   # отчёт + графики → reports/
.\scripts\docker-eda.ps1      # только графики EDA → reports/figures/
```

- **Linux / macOS:** сначала `chmod +x scripts/*.sh`, затем:
```bash
./scripts/docker-report.sh   # отчёт + графики → reports/
./scripts/docker-eda.sh      # только графики EDA → reports/figures/
```

**Вариант Б — команда с монтированием вручную:**

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

**4. Свои данные с хоста (опционально):**

Если CSV лежат не в образе, а на хосте в `./data/raw/`:

- **PowerShell (Windows):** `%CD%` в PowerShell не работает — используйте `$PWD`:
```powershell
docker run --rm -v "${PWD}/data/raw:/app/data/raw" hw1-recsys
```
- **cmd (Windows):**
```cmd
docker run --rm -v "%CD%\data\raw:/app/data/raw" hw1-recsys
```
- **Linux/macOS:**
```bash
docker run --rm -v "$(pwd)/data/raw:/app/data/raw" hw1-recsys
```

- `--rm` — удалить контейнер после завершения.
- Имя образа `hw1-recsys` можно заменить на любое (например `book-recsys`).

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
