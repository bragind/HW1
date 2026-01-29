# -*- coding: utf-8 -*-
"""
Этап 1: Расширенная подготовка данных — признаки пользователей, книг и сегментация.

- Признаки пользователей: средний рейтинг, количество оценок, активность (перцентиль).
- Признаки книг: популярность, разнообразие оценок (std), тематические категории (топ тегов).
- Сегментация пользователей: cold / regular / active для стратегии гибрида.
"""
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .config import COLD_USER_MAX_RATINGS, ACTIVE_USER_PERCENTILE
except ImportError:
    COLD_USER_MAX_RATINGS = 5
    ACTIVE_USER_PERCENTILE = 0.9


def build_user_features(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт расширенные признаки пользователей по обучающим оценкам.

    Признаки:
    - user_id
    - avg_rating — средний рейтинг пользователя
    - n_ratings — количество оценок (активность)
    - activity_percentile — перцентиль по числу оценок (0..1) для сегментации
    """
    user_agg = ratings.groupby('user_id').agg(
        avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count'),
    ).reset_index()

    n_ratings = user_agg['n_ratings']
    user_agg['activity_percentile'] = n_ratings.rank(pct=True, method='average')
    return user_agg


def build_book_features(
    ratings: pd.DataFrame,
    books: pd.DataFrame,
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
) -> pd.DataFrame:
    """
    Создаёт расширенные признаки книг.

    Признаки:
    - book_id
    - popularity — число оценок (логарифм для устойчивости)
    - rating_mean, rating_std — средний рейтинг и разнообразие оценок
    - top_tags — объединённые топ тегов книги для тематических категорий
    """
    book_agg = ratings.groupby('book_id').agg(
        rating_mean=('rating', 'mean'),
        rating_std=('rating', 'std'),
        n_ratings=('rating', 'count'),
    ).reset_index()
    book_agg['popularity'] = np.log1p(book_agg['n_ratings'] + 1)
    book_agg['rating_std'] = book_agg['rating_std'].fillna(0)

    # Тематические категории: топ тегов по книге (goodreads_book_id в датасете = id книги)
    tagged = book_tags.merge(tags[['tag_id', 'tag_name']], on='tag_id')
    top_per_book = (
        tagged.sort_values(['goodreads_book_id', 'count'], ascending=[True, False])
        .groupby('goodreads_book_id')
        .head(5)
    )
    top_tags_agg = (
        top_per_book.groupby('goodreads_book_id')['tag_name']
        .apply(lambda x: '|'.join(x.astype(str)))
        .reset_index()
    )
    top_tags_agg.columns = ['goodreads_book_id', 'top_tags']
    top_tags_agg = top_tags_agg.rename(columns={'goodreads_book_id': 'book_id'})
    book_agg = book_agg.merge(top_tags_agg, on='book_id', how='left')
    if 'top_tags' not in book_agg.columns:
        book_agg['top_tags'] = ''
    else:
        book_agg['top_tags'] = book_agg['top_tags'].fillna('')
    return book_agg


def get_user_segment(
    user_id: int,
    user_features: pd.DataFrame,
    cold_max_ratings: Optional[int] = None,
    active_percentile: Optional[float] = None,
) -> str:
    """
    Определяет сегмент пользователя для стратегии гибрида.

    - cold: n_ratings <= cold_max_ratings (новые/малоактивные)
    - active: activity_percentile >= active_percentile (топ по активности)
    - regular: остальные
    """
    cold_max_ratings = cold_max_ratings if cold_max_ratings is not None else COLD_USER_MAX_RATINGS
    active_percentile = active_percentile if active_percentile is not None else ACTIVE_USER_PERCENTILE
    row = user_features[user_features['user_id'] == user_id]
    if row.empty:
        return 'cold'
    row = row.iloc[0]
    n_ratings = int(row['n_ratings'])
    if n_ratings <= cold_max_ratings:
        return 'cold'
    if row['activity_percentile'] >= active_percentile:
        return 'active'
    return 'regular'
