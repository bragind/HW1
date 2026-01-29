# -*- coding: utf-8 -*-
"""
Этап 2: Гибридная система рекомендаций.

- Взвешенное объединение предсказаний моделей (Popularity, Content, Item-CF, SVD, опционально Neural).
- Разделение пользователей по типам (холодные/активные): для холодных увеличен вес популярности.
- Пул кандидатов из всех моделей, баланс разнообразия и релевантности, фильтрация уже прочитанных.
"""
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

try:
    from .config import HYBRID_WEIGHTS
except ImportError:
    HYBRID_WEIGHTS = {'popularity': 0.15, 'content': 0.20, 'item_cf': 0.30, 'svd': 0.35}
from .features import get_user_segment


# Для холодных пользователей увеличиваем вес популярности, уменьшая остальные пропорционально
COLD_USER_POPULARITY_BOOST = 0.5  # доля веса популярности для холодных


def _reciprocal_rank_score(rank: int) -> float:
    """Скор по обратному рангу (1-based)."""
    if rank <= 0:
        return 0.0
    return 1.0 / rank


def _get_weights_for_user(
    user_id: int,
    user_features: Optional[pd.DataFrame],
    base_weights: Dict[str, float],
    cold_popularity_boost: float = COLD_USER_POPULARITY_BOOST,
) -> Dict[str, float]:
    """
    Возвращает веса для пользователя. Для холодных увеличивает вес популярности.
    """
    weights = dict(base_weights)
    if user_features is None or user_features.empty:
        return weights
    seg = get_user_segment(user_id, user_features)
    if seg == 'cold':
        w_pop_new = min(1.0, base_weights.get('popularity', 0) + cold_popularity_boost)
        remainder = 1.0 - w_pop_new
        others = {k: v for k, v in base_weights.items() if k != 'popularity'}
        others_sum = sum(others.values())
        if others_sum > 0 and remainder > 0:
            for k in others:
                weights[k] = others[k] * (remainder / others_sum)
        weights['popularity'] = w_pop_new
    return weights


def hybrid_recommend(
    user_id: int,
    get_popular: Callable[[int], List[int]],
    get_content: Callable[[int, int], List[int]],
    get_itemcf: Callable[[int, int], List[int]],
    get_svd: Callable[[int, int], List[int]],
    user_rated_books: Set[int],
    user_features: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
    n_recommend: int = 10,
    pool_size_per_model: Optional[int] = None,
    get_neural: Optional[Callable[[int, int], List[int]]] = None,
    neural_weight: float = 0.0,
) -> List[int]:
    """
    Гибридная рекомендация: объединение моделей взвешенным усреднением по reciprocal rank.

    Параметры
    ---------
    user_id : int
        Идентификатор пользователя.
    get_popular : callable(N) -> list[book_id]
        Топ-N популярных книг (без user_id).
    get_content, get_itemcf, get_svd : callable(user_id, N) -> list[book_id]
        Персонализированные рекомендации от каждой модели.
    user_rated_books : set
        Книги, уже оценённые пользователем (исключаются из выдачи).
    user_features : DataFrame, optional
        Таблица признаков пользователей для сегментации (холодный/активный).
    weights : dict, optional
        Веса моделей (popularity, content, item_cf, svd). По умолчанию из config.
    n_recommend : int
        Сколько рекомендаций вернуть.
    pool_size_per_model : int, optional
        Сколько кандидатов брать от каждой модели (по умолчанию ~2*n_recommend).
    get_neural : callable(user_id, N), optional
        Рекомендации от нейросетевой модели (Two-Tower и т.п.).
    neural_weight : float
        Вес нейросетевой модели (добавляется к весам, затем веса нормируются).

    Returns
    -------
    list[int]
        Список book_id, отсортированный по комбинированному скору.
    """
    base_weights = weights if weights is not None else dict(HYBRID_WEIGHTS)
    if get_neural is not None and neural_weight > 0:
        total = sum(base_weights.values()) + neural_weight
        base_weights = {k: v / total for k, v in base_weights.items()}
        base_weights['neural'] = neural_weight / total
    w = _get_weights_for_user(user_id, user_features, base_weights)
    pool_n = pool_size_per_model or max(30, 2 * n_recommend)

    # Собираем рекомендации от каждой модели
    rec_popular = get_popular(pool_n)
    rec_content = get_content(user_id, pool_n)
    rec_itemcf = get_itemcf(user_id, pool_n)
    rec_svd = get_svd(user_id, pool_n)
    model_lists = {
        'popularity': rec_popular,
        'content': rec_content,
        'item_cf': rec_itemcf,
        'svd': rec_svd,
    }
    if get_neural is not None and 'neural' in w:
        model_lists['neural'] = get_neural(user_id, pool_n)

    # Взвешенный reciprocal rank fusion
    scores: Dict[int, float] = {}
    for model_name, rec_list in model_lists.items():
        weight = w.get(model_name, 0.0)
        if weight <= 0:
            continue
        for rank_1based, book_id in enumerate(rec_list, start=1):
            if book_id in user_rated_books:
                continue
            rr = _reciprocal_rank_score(rank_1based)
            scores[book_id] = scores.get(book_id, 0.0) + weight * rr

    # Сортируем по убыванию скора, возвращаем топ-n_recommend
    sorted_books = sorted(scores.keys(), key=lambda b: scores[b], reverse=True)
    return sorted_books[:n_recommend]
