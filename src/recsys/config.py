# -*- coding: utf-8 -*-
"""
Конфигурация проекта RSHW2: пути к данным и константы.
Goodbooks-10k — гибридная система рекомендаций книг.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
MODELS_CACHE_DIR = os.path.join(PROJECT_ROOT, 'models_cache')

PATH_RATINGS = os.path.join(DATA_RAW_DIR, 'ratings.csv')
PATH_BOOKS = os.path.join(DATA_RAW_DIR, 'books.csv')
PATH_TAGS = os.path.join(DATA_RAW_DIR, 'tags.csv')
PATH_BOOK_TAGS = os.path.join(DATA_RAW_DIR, 'book_tags.csv')

# Модели (из HW1)
MIN_RATINGS_FOR_POPULARITY = 50
TFIDF_MAX_FEATURES = 10_000
TFIDF_STOP_WORDS = 'english'
IMPLICIT_RATING_THRESHOLD = 4
ITEM_CF_K_NEIGHBORS = 20
SVD_N_FACTORS = 100
SVD_N_EPOCHS = 20
RATING_SCALE = (1, 5)
RANDOM_STATE = 42

# Оценка
TEST_SIZE = 0.2
RELEVANCE_THRESHOLD = 4
EVAL_K = 5
EVAL_MAX_USERS = 500

# Сегментация пользователей (для гибрида)
COLD_USER_MAX_RATINGS = 5
ACTIVE_USER_PERCENTILE = 0.9

# Гибрид: веса по умолчанию (сумма = 1)
HYBRID_WEIGHTS = {
    'popularity': 0.15,
    'content': 0.20,
    'item_cf': 0.30,
    'svd': 0.35,
}

# Two-Tower (продвинутая часть)
EMBEDDING_DIM = 64
TT_EPOCHS = 10
TT_BATCH_SIZE = 256
TT_LEARNING_RATE = 1e-3
