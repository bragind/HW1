# -*- coding: utf-8 -*-
"""
Модели рекомендаций: Popularity, Content-Based (TF-IDF), Item-Based CF, SVD.
Все функции принимают/возвращают идентификаторы user_id и book_id из датасета.
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    IMPLICIT_RATING_THRESHOLD,
    ITEM_CF_K_NEIGHBORS,
    MIN_RATINGS_FOR_POPULARITY,
    RANDOM_STATE,
    RATING_SCALE,
    SVD_N_EPOCHS,
    SVD_N_FACTORS,
    TFIDF_MAX_FEATURES,
    TFIDF_STOP_WORDS,
)


# -----------------------------------------------------------------------------
# 1. Неперсонализированная модель (Popularity)
# -----------------------------------------------------------------------------

def build_popularity_model(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Строит агрегаты по книгам: средний рейтинг и число оценок.
    Используется для Top-N по среднему рейтингу с порогом min_ratings.

    Returns
    -------
    pd.DataFrame
        Индекс — book_id, колонки: avg_rating, n_ratings.
    """
    return ratings.groupby('book_id').agg(
        avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count'),
    )


def get_top_popular_books(
    popularity_stats: pd.DataFrame,
    min_ratings: int = MIN_RATINGS_FOR_POPULARITY,
    top_n: int = 10,
) -> List[int]:
    """
    Топ-N популярных книг: по среднему рейтингу среди книг с числом оценок >= min_ratings.
    """
    filtered = popularity_stats[popularity_stats['n_ratings'] >= min_ratings]
    top = filtered.sort_values('avg_rating', ascending=False).head(top_n)
    return top.index.tolist()


# -----------------------------------------------------------------------------
# 2. Контентная модель (TF-IDF по профилю книги: название + теги)
# -----------------------------------------------------------------------------

def build_content_model(books_enhanced: pd.DataFrame):
    """
    Строит TF-IDF матрицу по полю 'profile' (original_title + теги).
    Возвращает объекты для вызова get_similar_books.
    """
    vectorizer = TfidfVectorizer(
        stop_words=TFIDF_STOP_WORDS,
        max_features=TFIDF_MAX_FEATURES,
    )
    tfidf_matrix = vectorizer.fit_transform(books_enhanced['profile'])
    book_id_to_idx = {
        bid: idx for idx, bid in enumerate(books_enhanced['book_id'])
    }
    return tfidf_matrix, book_id_to_idx, books_enhanced


def get_similar_books(
    book_id: int,
    tfidf_matrix,
    book_id_to_idx: Dict[int, int],
    books_enhanced: pd.DataFrame,
    N: int = 5,
) -> List[int]:
    """
    N наиболее похожих книг по косинусной близости TF-IDF векторов (исключая саму книгу).
    """
    if book_id not in book_id_to_idx:
        return []
    idx = book_id_to_idx[book_id]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = np.argsort(-sim)[1 : N + 1]  # [0] — сама книга
    return books_enhanced.iloc[top_indices]['book_id'].tolist()


# -----------------------------------------------------------------------------
# 3. Item-Based Collaborative Filtering
# -----------------------------------------------------------------------------
#
# Вычислительная сложность Item-CF:
# - Построение матрицы схожестей книг: O(n_books^2) по памяти и O(n_books^2 * n_users)
#   при плотной косинусной близости; при sparse — O(nnz_columns^2) с оптимизациями.
# - Предсказание для одного (user, book): O(n_books) при наивном переборе соседей;
#   с отбором топ-K соседей: O(n_books + K).
# - Топ-N рекомендаций для пользователя: O(n_books * (n_books + K)) = O(n_books^2)
#   при наивном переборе всех книг.
#
# Оптимизации для больших данных:
# - Хранить item_sim в sparse формате (только топ-K соседей на книгу).
# - Предвычислять и кэшировать топ-K соседей для каждой книги.
# - Использовать приближённый поиск соседей (LSH, ANN) вместо полной матрицы.
# - Инкрементальное обновление схожестей при потоковых данных.
#

def build_item_cf_structures(ratings: pd.DataFrame) -> Dict[str, object]:
    """
    Строит матрицу взаимодействий user×book (неявный feedback: 1 если rating >= порог)
    и матрицу попарных схожестей книг (косинусная мера по столбцам).
    """
    ratings = ratings.copy()
    ratings['implicit'] = (ratings['rating'] >= IMPLICIT_RATING_THRESHOLD).astype(int)

    user_ids = ratings['user_id'].unique()
    book_ids = ratings['book_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    book_to_idx = {b: i for i, b in enumerate(book_ids)}
    idx_to_book = {i: b for b, i in book_to_idx.items()}

    n_users = len(user_ids)
    n_books = len(book_ids)

    rows = ratings['user_id'].map(user_to_idx)
    cols = ratings['book_id'].map(book_to_idx)
    data = ratings['implicit']
    rating_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_books))

    item_sim = cosine_similarity(rating_matrix.T)  # (n_books, n_books)

    return {
        'rating_matrix': rating_matrix,
        'item_sim': item_sim,
        'user_to_idx': user_to_idx,
        'book_to_idx': book_to_idx,
        'idx_to_book': idx_to_book,
        'n_users': n_users,
        'n_books': n_books,
    }


def predict_rating_item_cf(
    user_id: int,
    book_id: int,
    cf_structures: Dict[str, object],
    k: int = ITEM_CF_K_NEIGHBORS,
    default_rating: float = 3.0,
) -> float:
    """
    Предсказание оценки по Item-CF: взвешенное среднее оценок пользователя
    по K наиболее похожим книгам (по неявному feedback: 0/1).
    """
    rating_matrix = cf_structures['rating_matrix']
    item_sim = cf_structures['item_sim']
    user_to_idx = cf_structures['user_to_idx']
    book_to_idx = cf_structures['book_to_idx']
    n_books = cf_structures['n_books']
    idx_to_book = cf_structures['idx_to_book']

    if user_id not in user_to_idx or book_id not in book_to_idx:
        return default_rating

    u_idx = user_to_idx[user_id]
    b_idx = book_to_idx[book_id]
    user_vec = rating_matrix[u_idx].toarray().flatten()
    sim_scores = item_sim[b_idx].copy()
    sim_scores[b_idx] = 0  # исключаем саму книгу

    rated = user_vec > 0
    if not np.any(rated):
        return default_rating

    weighted_sum = np.dot(sim_scores, user_vec)
    sim_sum = np.sum(sim_scores)
    if sim_sum == 0:
        return float(np.mean(user_vec[rated]))
    return weighted_sum / sim_sum


def get_itemcf_recommendations(
    user_id: int,
    cf_structures: Dict[str, object],
    get_fallback_top_n,
    N: int = 5,
) -> List[int]:
    """
    Топ-N рекомендаций Item-CF: для пользователя предсказываем оценку по всем
    непрочитанным книгам и сортируем по убыванию.
    """
    user_to_idx = cf_structures['user_to_idx']
    rating_matrix = cf_structures['rating_matrix']
    n_books = cf_structures['n_books']
    idx_to_book = cf_structures['idx_to_book']

    if user_id not in user_to_idx:
        return get_fallback_top_n(N)

    u_idx = user_to_idx[user_id]
    user_rated = rating_matrix[u_idx].toarray().flatten() > 0

    scores = []
    for b_idx in range(n_books):
        if not user_rated[b_idx]:
            pred = predict_rating_item_cf(
                user_id, idx_to_book[b_idx], cf_structures, k=ITEM_CF_K_NEIGHBORS
            )
            scores.append((pred, idx_to_book[b_idx]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [book_id for _, book_id in scores[:N]]


# -----------------------------------------------------------------------------
# 4. SVD (Matrix Factorization, Surprise)
# -----------------------------------------------------------------------------

def build_svd_model(ratings: pd.DataFrame) -> Dict[str, object]:
    """
    Обучает SVD на переданных ratings (обычно — обучающая выборка).
    Тестовый набор для RMSE формируется снаружи из отложенной выборки.
    """
    from surprise import Dataset, Reader
    from surprise import SVD as SurpriseSVD

    reader = Reader(rating_scale=RATING_SCALE)
    data = Dataset.load_from_df(
        ratings[['user_id', 'book_id', 'rating']],
        reader,
    )
    trainset = data.build_full_trainset()

    model = SurpriseSVD(
        n_factors=SVD_N_FACTORS,
        n_epochs=SVD_N_EPOCHS,
        random_state=RANDOM_STATE,
    )
    model.fit(trainset)

    user_ids = ratings['user_id'].unique()
    book_ids = ratings['book_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    return {
        'model': model,
        'user_to_idx': user_to_idx,
        'book_ids': book_ids,
    }


def get_svd_recommendations(
    user_id: int,
    svd_structures: Dict[str, object],
    get_fallback_top_n,
    N: int = 5,
) -> List[int]:
    """
    Топ-N рекомендаций SVD: предсказание рейтинга по всем книгам, сортировка по убыванию.
    """
    model = svd_structures['model']
    user_to_idx = svd_structures['user_to_idx']
    book_ids = svd_structures['book_ids']

    if user_id not in user_to_idx:
        return get_fallback_top_n(N)

    predictions = [
        (model.predict(user_id, bid).est, bid)
        for bid in book_ids
    ]
    predictions.sort(key=lambda x: x[0], reverse=True)
    return [bid for _, bid in predictions[:N]]

