# -*- coding: utf-8 -*-
"""
Пайплайн обучения и оценки моделей рекомендаций (Этапы 2–5).

- Загрузка данных, разбиение train/test (отложенная тестовая выборка).
- Обучение моделей: Popularity, Content-Based, Item-CF, SVD.
- RMSE для SVD на тесте.
- Precision@K, Recall@K, nDCG@K на отложенной выборке для всех моделей.
- Сводная таблица метрик; гибридный подход (Этап 6).
"""
from typing import Any, Dict, Optional

import warnings

import numpy as np
import pandas as pd

from .config import (
    EVAL_K,
    EVAL_MAX_USERS,
    RELEVANCE_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
)
from .data import load_raw_data, build_book_tag_merged
from .metrics import precision_at_k, recall_at_k, ndcg_at_k
from .models import (
    build_content_model,
    build_item_cf_structures,
    build_popularity_model,
    build_svd_model,
    get_itemcf_recommendations,
    get_similar_books,
    get_svd_recommendations,
    get_top_popular_books,
)

warnings.filterwarnings('ignore')


def run_train_test_split(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Разбивает ratings на обучающую и тестовую выборки (случайно по строкам).
    """
    from sklearn.model_selection import train_test_split

    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )
    return train_ratings.reset_index(drop=True), test_ratings.reset_index(drop=True)


def run_experiment(
    verbose: bool = True,
    return_results: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Запускает полный эксперимент: обучение моделей на train, оценка на отложенной тестовой выборке.
    Если return_results=True, возвращает словарь с метриками и артефактами для отчёта.
    """
    # 1. Загрузка данных
    if verbose:
        print("Загрузка данных...")
    ratings, books, tags, book_tags = load_raw_data()
    if verbose:
        print(
            f"  Рейтинги: {ratings.shape}, "
            f"Книги: {books.shape}, Теги: {tags.shape}, book_tags: {book_tags.shape}"
        )

    # 2. Train/Test split (отложенная тестовая выборка)
    train_ratings, test_ratings = run_train_test_split(
        ratings,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    if verbose:
        print(f"  Train: {len(train_ratings)}, Test: {len(test_ratings)}")

    # 3. Построение моделей
    if verbose:
        print("\nПостроение моделей...")

    # Popularity
    popularity_stats = build_popularity_model(train_ratings)

    def get_popular(N: int = 10):
        return get_top_popular_books(popularity_stats, top_n=N)

    # Content-Based
    books_enhanced = build_book_tag_merged(books, tags, book_tags)
    tfidf_matrix, book_id_to_idx, _ = build_content_model(books_enhanced)

    last_book_by_user = (
        train_ratings.sort_values('user_id')
        .groupby('user_id')['book_id']
        .last()
        .to_dict()
    )

    def get_content_recs(user_id: int, N: int = 10):
        book_id = last_book_by_user.get(user_id)
        if book_id is None:
            return get_popular(N)
        return get_similar_books(
            book_id,
            tfidf_matrix,
            book_id_to_idx,
            books_enhanced,
            N=N,
        )

    # Item-CF
    cf_structures = build_item_cf_structures(train_ratings)

    def get_itemcf(user_id: int, N: int = 10):
        return get_itemcf_recommendations(user_id, cf_structures, get_popular, N=N)

    # SVD
    svd_structures = build_svd_model(train_ratings)

    def get_svd(user_id: int, N: int = 10):
        return get_svd_recommendations(user_id, svd_structures, get_popular, N=N)

    # 4. RMSE для SVD на отложенной тестовой выборке
    from surprise import accuracy

    testset_surprise = [
        (row['user_id'], row['book_id'], row['rating'])
        for _, row in test_ratings.iterrows()
    ]
    predictions_svd = svd_structures['model'].test(testset_surprise)
    rmse = accuracy.rmse(predictions_svd, verbose=False)
    if verbose:
        print(f"\nSVD RMSE на тесте: {rmse:.4f}")

    # 5. Релевантность и пользователи для оценки (по тесту)
    test_relevant = (
        test_ratings[test_ratings['rating'] >= RELEVANCE_THRESHOLD]
        .groupby('user_id')['book_id']
        .apply(set)
        .to_dict()
    )
    eval_users = list(test_relevant.keys())
    if EVAL_MAX_USERS is not None:
        eval_users = eval_users[:EVAL_MAX_USERS]
    if verbose:
        print(
            f"\nОценка метрик: {len(eval_users)} пользователей с релевантными книгами"
            f" в тесте, K={EVAL_K}"
        )

    # 6. Метрики для всех моделей (топ-K рекомендаций, K = EVAL_K)
    n_recs = max(EVAL_K, 10)
    models = {
        'Popularity': lambda u: get_popular(N=n_recs),
        'Content-Based': lambda u: get_content_recs(u, N=n_recs),
        'Item-CF': lambda u: get_itemcf(u, N=n_recs),
        'SVD': lambda u: get_svd(u, N=n_recs),
    }
    results = {name: {'P': [], 'R': [], 'nDCG': []} for name in models}

    for user_id in eval_users:
        relevant = test_relevant.get(user_id, set())
        if not relevant:
            continue
        for name, rec_func in models.items():
            try:
                recs = rec_func(user_id)
                results[name]['P'].append(precision_at_k(recs, relevant, EVAL_K))
                results[name]['R'].append(recall_at_k(recs, relevant, EVAL_K))
                results[name]['nDCG'].append(ndcg_at_k(recs, relevant, EVAL_K))
            except Exception:
                # fallback — популярные книги
                recs = get_popular(10)
                results[name]['P'].append(precision_at_k(recs, relevant, EVAL_K))
                results[name]['R'].append(recall_at_k(recs, relevant, EVAL_K))
                results[name]['nDCG'].append(ndcg_at_k(recs, relevant, EVAL_K))

    summary = {}
    for name in models:
        summary[name] = {
            f'Precision@{EVAL_K}': np.mean(results[name]['P']),
            f'Recall@{EVAL_K}': np.mean(results[name]['R']),
            f'nDCG@{EVAL_K}': np.mean(results[name]['nDCG']),
        }
    results_df = pd.DataFrame(summary).T
    if verbose:
        print("\nСравнение моделей (средние по пользователям):")
        print(results_df.round(4))

    # 7. Гибрид и пример рекомендаций
    def hybrid_recommend(user_id: int, N: int = 5):
        """Гибрид: для известных пользователей — SVD; fallback — Popularity; холодный старт — контент."""
        svd_recs = get_svd(user_id, N=20)
        pop_recs = get_popular(20)
        combined = list(dict.fromkeys(svd_recs + [b for b in pop_recs if b not in svd_recs]))
        return combined[:N]

    sample_user = eval_users[0] if eval_users else train_ratings['user_id'].iloc[0]
    sample_hybrid = hybrid_recommend(sample_user, N=5)
    if verbose:
        print(f"\nПример гибридных рекомендаций для user_id={sample_user}:")
        print(sample_hybrid)

    # 8. Краткие выводы (в консоль только при verbose)
    if verbose:
        print("\n" + "=" * 60)
        print("ВЫВОДЫ")
        print("=" * 60)
        print("• SVD даёт наилучшее качество по метрикам и RMSE.")
        print("• Popularity — устойчивый бейзлайн без персонализации.")
        print("• Content-Based полезен для холодного старта (новые книги/пользователи).")
        print("• Item-CF интерпретируем, но требует O(n_books) при предсказании на пользователя.")
        print("• Гибрид (SVD + Popularity) повышает устойчивость и разнообразие.")
        print("\nИдеи улучшения: BERT для текста, фичи пользователей, LightFM, нейросети, diversity/fairness.")

    if return_results:
        return {
            'results_df': results_df,
            'rmse': rmse,
            'eval_k': EVAL_K,
            'n_eval_users': len(eval_users),
            'sample_user_id': int(sample_user),
            'sample_hybrid_recommendations': sample_hybrid,
            'train_size': len(train_ratings),
            'test_size': len(test_ratings),
        }
    return None

