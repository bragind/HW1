# -*- coding: utf-8 -*-
"""
Конфигурация проекта: пути к данным и константы.
Goodbooks-10k — прототип книжного рекомендательного сервиса.

ВНИМАНИЕ: файл лежит в src/recsys, поэтому BASE_DIR поднимаем до корня проекта.
"""
import os

# -----------------------------------------------------------------------------
# Пути к данным (относительно корня проекта)
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Сырьевые данные (raw) — ожидаются в data/raw/
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')

PATH_RATINGS = os.path.join(DATA_RAW_DIR, 'ratings.csv')
PATH_BOOKS = os.path.join(DATA_RAW_DIR, 'books.csv')
PATH_TAGS = os.path.join(DATA_RAW_DIR, 'tags.csv')
PATH_BOOK_TAGS = os.path.join(DATA_RAW_DIR, 'book_tags.csv')

# -----------------------------------------------------------------------------
# Параметры моделей
# -----------------------------------------------------------------------------
# Неперсонализированная модель: минимальное число оценок для учёта книги
MIN_RATINGS_FOR_POPULARITY = 50

# Контентная модель: TF-IDF
TFIDF_MAX_FEATURES = 10_000
TFIDF_STOP_WORDS = 'english'

# Item-Based CF: неявный feedback (оценка >= порога считается положительной)
IMPLICIT_RATING_THRESHOLD = 4
ITEM_CF_K_NEIGHBORS = 20

# SVD (Surprise)
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
RATING_SCALE = (1, 5)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Оценка моделей
# -----------------------------------------------------------------------------
# Релевантная книга: пользователь поставил оценку >= порога
RELEVANCE_THRESHOLD = 4
# Метрики считаются при K
EVAL_K = 5
# Число пользователей для оценки (для ускорения; None = все)
EVAL_MAX_USERS = 500

