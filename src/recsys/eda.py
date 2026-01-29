# -*- coding: utf-8 -*-
"""
Этап 1: Разведочный анализ данных (EDA).

Загружает ratings, books, tags, book_tags.
Строит графики и формирует выводы:
- Распределение оценок (смещение в сторону высоких?)
- Активность пользователей: количество пользователей vs количество оценок
- Популярность книг: количество книг vs количество оценок (длинный хвост)
- Самые частые теги (book_tags + tags)
- Проблемы данных: разреженность, смещение популярности.
"""
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .config import FIGURES_DIR, REPORTS_DIR
from .data import load_raw_data


def _ensure_dirs() -> None:
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


def run_ratings_distribution(
    ratings: pd.DataFrame,
    figures_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Распределение оценок. Есть ли смещение в сторону высоких оценок?
    """
    figures_dir = figures_dir or FIGURES_DIR
    _ensure_dirs()

    dist = ratings['rating'].value_counts().sort_index()
    mean_rating = float(ratings['rating'].mean())
    high_share = float((ratings['rating'] >= 4).mean())

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        dist.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel('Оценка')
        ax.set_ylabel('Количество')
        ax.set_title(f'Распределение оценок (среднее = {mean_rating:.2f}; доля ≥4: {high_share:.0%})')
        plt.tight_layout()
        plt.savefig(Path(figures_dir) / 'ratings_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    return {
        'distribution': dist.to_dict(),
        'mean_rating': mean_rating,
        'share_high_ratings_ge4': high_share,
        'conclusion': 'Смещение в сторону высоких оценок есть' if mean_rating > 3.5 else 'Смещение умеренное',
    }


def run_user_activity(
    ratings: pd.DataFrame,
    figures_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Анализ активности пользователей.
    График: количество пользователей vs количество оценок (распределение числа оценок на пользователя).
    """
    figures_dir = figures_dir or FIGURES_DIR
    _ensure_dirs()

    per_user = ratings.groupby('user_id').size()
    active_threshold = per_user.quantile(0.9)
    cold_users = (per_user <= 5).sum()
    active_users = (per_user >= active_threshold).sum()

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        per_user.hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(5, color='red', linestyle='--', label=f'холодные (≤5 оценок): {cold_users} польз.')
        ax.axvline(active_threshold, color='green', linestyle='--', label=f'активные (≥P90): {active_users} польз.')
        ax.set_xlabel('Количество оценок на пользователя')
        ax.set_ylabel('Количество пользователей')
        ax.set_title('Взаимоотношение количества пользователей и количества оценок')
        ax.legend()
        plt.tight_layout()
        plt.savefig(Path(figures_dir) / 'user_activity.png', dpi=100, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    return {
        'n_users': int(per_user.shape[0]),
        'ratings_per_user_median': float(per_user.median()),
        'ratings_per_user_mean': float(per_user.mean()),
        'cold_users_le5': int(cold_users),
        'active_users_p90': int(active_users),
        'conclusion': f'Проблема холодного старта: {cold_users} пользователей с ≤5 оценок.',
    }


def run_book_popularity(
    ratings: pd.DataFrame,
    figures_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Анализ популярности книг.
    График: количество книг vs количество оценок (длинный хвост).
    """
    figures_dir = figures_dir or FIGURES_DIR
    _ensure_dirs()

    per_book = ratings.groupby('book_id').size()
    tail_10 = (per_book <= per_book.quantile(0.1)).sum()
    head_10 = (per_book >= per_book.quantile(0.9)).sum()

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        per_book.hist(bins=80, ax=ax, color='coral', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Количество оценок на книгу')
        ax.set_ylabel('Количество книг')
        ax.set_title('Количество книг vs количество оценок (длинный хвост)')
        plt.tight_layout()
        plt.savefig(Path(figures_dir) / 'book_popularity.png', dpi=100, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    return {
        'n_books': int(per_book.shape[0]),
        'ratings_per_book_median': float(per_book.median()),
        'ratings_per_book_mean': float(per_book.mean()),
        'long_tail_bottom_10pct_books': int(tail_10),
        'head_top_10pct_books': int(head_10),
        'conclusion': 'Сильное смещение популярности: мало книг с большим числом оценок, длинный хвост малооценённых.',
    }


def run_top_tags(
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
    top_n: int = 20,
    figures_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Визуализация самых частых тегов (по числу присвоений в book_tags).
    """
    figures_dir = figures_dir or FIGURES_DIR
    _ensure_dirs()

    tag_counts = book_tags.groupby('tag_id')['count'].sum().reset_index()
    tag_counts = tag_counts.merge(tags[['tag_id', 'tag_name']], on='tag_id')
    tag_counts = tag_counts.sort_values('count', ascending=False).head(top_n)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(tag_counts)), tag_counts['count'], color='seagreen', alpha=0.8)
        ax.set_yticks(range(len(tag_counts)))
        ax.set_yticklabels(tag_counts['tag_name'].str[:40], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Количество присвоений (count)')
        ax.set_title(f'Топ-{top_n} самых частых тегов книг (book_tags)')
        plt.tight_layout()
        plt.savefig(Path(figures_dir) / 'top_tags.png', dpi=100, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    return {
        'top_tags': tag_counts[['tag_name', 'count']].to_dict('records'),
        'conclusion': 'Частые теги помогают контентной модели и интерпретации.',
    }


def run_eda(
    figures_dir: Optional[str] = None,
    load_data: bool = True,
    ratings: Optional[pd.DataFrame] = None,
    books: Optional[pd.DataFrame] = None,
    tags: Optional[pd.DataFrame] = None,
    book_tags: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Полный EDA: загрузка данных (если не переданы), все графики и сводка проблем данных.
    """
    if load_data and ratings is None:
        ratings, books, tags, book_tags = load_raw_data()
    elif ratings is None:
        ratings, books, tags, book_tags = load_raw_data()

    out = {
        'ratings_shape': ratings.shape,
        'books_shape': books.shape if books is not None else None,
        'tags_shape': tags.shape if tags is not None else None,
        'book_tags_shape': book_tags.shape if book_tags is not None else None,
    }

    r_dist = run_ratings_distribution(ratings, figures_dir=figures_dir)
    out['ratings_distribution'] = r_dist

    u_act = run_user_activity(ratings, figures_dir=figures_dir)
    out['user_activity'] = u_act

    b_pop = run_book_popularity(ratings, figures_dir=figures_dir)
    out['book_popularity'] = b_pop

    if book_tags is not None and tags is not None:
        top_t = run_top_tags(book_tags, tags, figures_dir=figures_dir)
        out['top_tags'] = top_t

    # Разреженность и смещение популярности
    n_cells = int(ratings['user_id'].nunique()) * int(ratings['book_id'].nunique())
    sparsity = 1.0 - (len(ratings) / n_cells) if n_cells else 0
    out['data_issues'] = {
        'sparsity': sparsity,
        'sparsity_note': f'Матрица user×book заполнена на {(1 - sparsity) * 100:.4f}% — высокая разреженность.',
        'popularity_bias_note': 'Смещение популярности: большинство оценок приходится на популярные книги и активных пользователей.',
    }

    return out
