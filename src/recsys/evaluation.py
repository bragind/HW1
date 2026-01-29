# -*- coding: utf-8 -*-
"""
Оценка и оптимизация гибридной системы.

Этап 3: Углублённый сравнительный анализ.
- Сравнение гибридной системы с отдельными моделями.
- Анализ эффективности для разных сегментов пользователей.
- Оптимизация весов в гибридной модели.
"""
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import (
    EVAL_K,
    EVAL_MAX_USERS,
    RELEVANCE_THRESHOLD,
    RANDOM_STATE,
    TEST_SIZE,
)
try:
    from .config import HYBRID_WEIGHTS
except ImportError:
    HYBRID_WEIGHTS = {'popularity': 0.15, 'content': 0.20, 'item_cf': 0.30, 'svd': 0.35}
from .data import load_raw_data, build_book_tag_merged
from .features import build_user_features, build_book_features, get_user_segment
from .hybrid import hybrid_recommend
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


def run_train_test_split(
    ratings: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разбивает ratings на train и test."""
    from sklearn.model_selection import train_test_split
    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
    )
    return train_ratings.reset_index(drop=True), test_ratings.reset_index(drop=True)


def _build_test_relevant(test_ratings: pd.DataFrame) -> Dict[int, Set[int]]:
    """Релевантные книги по пользователям (rating >= порог)."""
    return (
        test_ratings[test_ratings['rating'] >= RELEVANCE_THRESHOLD]
        .groupby('user_id')['book_id']
        .apply(set)
        .to_dict()
    )


def _build_user_rated_on_train(train_ratings: pd.DataFrame) -> Dict[int, Set[int]]:
    """Для каждого user_id — множество book_id, которые он уже оценил в train."""
    return (
        train_ratings.groupby('user_id')['book_id']
        .apply(set)
        .to_dict()
    )


def evaluate_recommender(
    rec_func: Callable[[int], List[int]],
    eval_users: List[int],
    test_relevant: Dict[int, Set[int]],
    k: int = EVAL_K,
) -> Dict[str, float]:
    """Считает средние Precision@K, Recall@K, nDCG@K для rec_func."""
    p_list, r_list, ndcg_list = [], [], []
    for user_id in eval_users:
        relevant = test_relevant.get(user_id, set())
        if not relevant:
            continue
        try:
            recs = rec_func(user_id)
        except Exception:
            recs = []
        p_list.append(precision_at_k(recs, relevant, k))
        r_list.append(recall_at_k(recs, relevant, k))
        ndcg_list.append(ndcg_at_k(recs, relevant, k))
    if not p_list:
        return {'Precision@K': 0.0, 'Recall@K': 0.0, 'nDCG@K': 0.0}
    return {
        f'Precision@{k}': np.mean(p_list),
        f'Recall@{k}': np.mean(r_list),
        f'nDCG@{k}': np.mean(ndcg_list),
    }


def run_full_evaluation(
    verbose: bool = True,
    return_results: bool = False,
    eval_max_users: Optional[int] = EVAL_MAX_USERS,
    get_neural: Optional[Callable[[int, int], List[int]]] = None,
    neural_weight: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Полный цикл: загрузка данных, train/test, построение моделей и гибрида,
    оценка метрик по всем моделям и по сегментам пользователей.
    Данные читаются из путей в config (data/raw/).
    При передаче get_neural и neural_weight > 0 нейросетевая модель включается в гибрид.
    """
    ratings, books, tags, book_tags = load_raw_data()
    train_ratings, test_ratings = run_train_test_split(ratings)
    test_relevant = _build_test_relevant(test_ratings)
    user_rated_train = _build_user_rated_on_train(train_ratings)
    eval_users = [u for u in test_relevant if test_relevant[u]]
    if eval_max_users is not None:
        eval_users = eval_users[:eval_max_users]
    k = EVAL_K
    n_recs = max(k, 20)

    # Расширенные признаки
    user_features = build_user_features(train_ratings)
    book_features = build_book_features(train_ratings, books, book_tags, tags)

    # Модели
    popularity_stats = build_popularity_model(train_ratings)
    def get_popular(N: int = 10):
        return get_top_popular_books(popularity_stats, top_n=N)

    books_enhanced = build_book_tag_merged(books, tags, book_tags)
    tfidf_matrix, book_id_to_idx, _ = build_content_model(books_enhanced)
    last_book_by_user = (
        train_ratings.sort_values('user_id')
        .groupby('user_id')['book_id']
        .last()
        .to_dict()
    )

    def get_content(user_id: int, N: int = 10):
        book_id = last_book_by_user.get(user_id)
        if book_id is None:
            return get_popular(N)
        return get_similar_books(
            book_id, tfidf_matrix, book_id_to_idx, books_enhanced, N=N,
        )

    cf_structures = build_item_cf_structures(train_ratings)
    def get_itemcf(user_id: int, N: int = 10):
        return get_itemcf_recommendations(user_id, cf_structures, get_popular, N=N)

    svd_structures = build_svd_model(train_ratings)
    def get_svd(user_id: int, N: int = 10):
        return get_svd_recommendations(user_id, svd_structures, get_popular, N=N)

    # Рекомендаторы для оценки (принимают только user_id)
    def rec_popular(uid):
        return get_popular(n_recs)
    def rec_content(uid):
        return get_content(uid, n_recs)
    def rec_itemcf(uid):
        return get_itemcf(uid, n_recs)
    def rec_svd(uid):
        return get_svd(uid, n_recs)
    def rec_hybrid(uid):
        return hybrid_recommend(
            uid,
            get_popular=get_popular,
            get_content=get_content,
            get_itemcf=get_itemcf,
            get_svd=get_svd,
            user_rated_books=user_rated_train.get(uid, set()),
            user_features=user_features,
            weights=HYBRID_WEIGHTS,
            n_recommend=n_recs,
            get_neural=get_neural,
            neural_weight=neural_weight,
        )

    models = {
        'Popularity': rec_popular,
        'Content-Based': rec_content,
        'Item-CF': rec_itemcf,
        'SVD': rec_svd,
        'Hybrid': rec_hybrid,
    }

    results_all = {}
    for name, rec_func in models.items():
        results_all[name] = evaluate_recommender(
            rec_func, eval_users, test_relevant, k=k,
        )

    results_df = pd.DataFrame(results_all).T
    if verbose:
        print('Сравнение моделей (средние по пользователям):')
        print(results_df.round(4))

    # Анализ по сегментам
    segments = ['cold', 'regular', 'active']
    by_segment: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in segments}
    for seg in segments:
        users_seg = [u for u in eval_users if get_user_segment(u, user_features) == seg]
        if not users_seg:
            continue
        for name, rec_func in models.items():
            by_segment[seg][name] = evaluate_recommender(
                rec_func, users_seg, test_relevant, k=k,
            )
    if verbose:
        print('\nМетрики по сегментам пользователей:')
        for seg in segments:
            if by_segment[seg]:
                print(f'  {seg}:', pd.DataFrame(by_segment[seg]).T.round(4).to_string())

    # Пример гибридных рекомендаций для отчёта
    sample_user = eval_users[0] if eval_users else None
    sample_hybrid = rec_hybrid(sample_user)[:5] if sample_user else []
    if verbose and sample_user:
        print(f'\nПример гибридных рекомендаций для user_id={sample_user}: {sample_hybrid[:5]}')

    out = {
        'results_df': results_df,
        'by_segment': by_segment,
        'eval_k': k,
        'n_eval_users': len(eval_users),
        'train_size': len(train_ratings),
        'test_size': len(test_ratings),
        'sample_user': sample_user,
        'sample_hybrid_recommendations': sample_hybrid,
    }
    if return_results:
        return out
    return None


def optimize_hybrid_weights_grid(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    test_relevant: Dict[int, Set[int]],
    user_rated_train: Dict[int, Set[int]],
    user_features: pd.DataFrame,
    get_popular: Callable[[int], List[int]],
    get_content: Callable[[int, int], List[int]],
    get_itemcf: Callable[[int, int], List[int]],
    get_svd: Callable[[int, int], List[int]],
    eval_users: List[int],
    k: int = EVAL_K,
    n_recommend: int = 10,
    grid_steps: int = 3,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Поиск весов гибрида по сетке (упрощённо: перебор долей popularity/content/item_cf/svd).
    Возвращает лучшие веса и таблицу результатов по вариантам.
    """
    from itertools import product
    step = 1.0 / grid_steps
    values = [i * step for i in range(grid_steps + 1)]
    best_score = -1.0
    best_weights: Dict[str, float] = {}
    rows = []

    for wpop, wcont, wicf in product(values, values, values):
        wsvd = 1.0 - wpop - wcont - wicf
        if wsvd < 0:
            continue
        weights = {'popularity': wpop, 'content': wcont, 'item_cf': wicf, 'svd': wsvd}
        def rec_hybrid(uid):
            return hybrid_recommend(
                uid,
                get_popular=get_popular,
                get_content=get_content,
                get_itemcf=get_itemcf,
                get_svd=get_svd,
                user_rated_books=user_rated_train.get(uid, set()),
                user_features=user_features,
                weights=weights,
                n_recommend=n_recommend,
            )
        res = evaluate_recommender(rec_hybrid, eval_users, test_relevant, k=k)
        score = res[f'nDCG@{k}']
        rows.append({**weights, f'nDCG@{k}': score})
        if score > best_score:
            best_score = score
            best_weights = weights

    return best_weights, pd.DataFrame(rows)
