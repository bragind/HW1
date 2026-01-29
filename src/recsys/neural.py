# -*- coding: utf-8 -*-
"""
Продвинутая часть: нейросетевая архитектура Two-Tower.

Две башни (пользователь и книга) с эмбеддингами, скор = скалярное произведение.
Обучение на положительных парах (user, book) с рейтингом >= порога.
Интеграция в гибрид: get_neural(user_id, N) возвращает топ-N рекомендаций.
"""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    IMPLICIT_RATING_THRESHOLD,
    EMBEDDING_DIM,
    TT_EPOCHS,
    TT_BATCH_SIZE,
    TT_LEARNING_RATE,
    RANDOM_STATE,
)


def _build_mappings(
    ratings: pd.DataFrame,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], int, int]:
    """Строит user_to_idx, book_to_idx, idx_to_book и размеры."""
    user_ids = ratings['user_id'].unique()
    book_ids = ratings['book_id'].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    book_to_idx = {b: i for i, b in enumerate(book_ids)}
    idx_to_book = {i: b for b, i in book_to_idx.items()}
    n_users = len(user_ids)
    n_books = len(book_ids)
    return user_to_idx, book_to_idx, idx_to_book, n_users, n_books


def build_two_tower_recommender(
    train_ratings: pd.DataFrame,
    embedding_dim: int = EMBEDDING_DIM,
    epochs: int = TT_EPOCHS,
    batch_size: int = TT_BATCH_SIZE,
    lr: float = TT_LEARNING_RATE,
    verbose: bool = True,
) -> Callable[[int, int], List[int]]:
    """
    Обучает Two-Tower модель на train_ratings и возвращает функцию рекомендаций.

    Положительные пары: рейтинг >= IMPLICIT_RATING_THRESHOLD.
    Отрицательные: случайная выборка непрочитанных книг (per user).

    Returns
    -------
    get_neural(user_id, N) -> list[book_id]
        Топ-N рекомендаций для пользователя (исключая уже оценённые в train).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("Two-Tower требует torch. Установите: pip install torch")

    user_to_idx, book_to_idx, idx_to_book, n_users, n_books = _build_mappings(train_ratings)

    # Положительные пары
    pos = train_ratings[train_ratings['rating'] >= IMPLICIT_RATING_THRESHOLD][
        ['user_id', 'book_id']
    ].drop_duplicates()
    pos['user_idx'] = pos['user_id'].map(user_to_idx)
    pos['book_idx'] = pos['book_id'].map(book_to_idx)
    pos = pos.dropna(subset=['user_idx', 'book_idx'])
    pos = pos.astype({'user_idx': np.int64, 'book_idx': np.int64})

    if pos.empty:
        if verbose:
            print("Нет положительных пар для Two-Tower, возвращаем заглушку.")
        return _fallback_recommender(train_ratings)

    # Отрицательные пары: для каждой положительной пары — случайная книга, которую пользователь не оценивал
    user_rated = train_ratings.groupby('user_id')['book_id'].apply(set).to_dict()
    all_books = set(book_to_idx.keys())

    def neg_sample(user_id: int) -> int:
        rated = user_rated.get(user_id, set())
        cand = list(all_books - rated)
        if not cand:
            return np.random.choice(list(book_to_idx.keys()))
        return np.random.choice(cand)

    np.random.seed(RANDOM_STATE)
    neg_books = pos['user_id'].map(neg_sample)
    pos['book_neg_idx'] = neg_books.map(book_to_idx)

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, embedding_dim)
            self.book_emb = nn.Embedding(n_books, embedding_dim)
            nn.init.xavier_uniform_(self.user_emb.weight)
            nn.init.xavier_uniform_(self.book_emb.weight)

        def forward(self, u_idx, b_idx):
            u = self.user_emb(u_idx)   # (B, dim)
            b = self.book_emb(b_idx)   # (B, dim)
            return (u * b).sum(dim=1)   # (B,)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoTower().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Обучение: BPR-подобный loss (скор позитива выше скора негатива)
    pos_u = torch.LongTensor(pos['user_idx'].values)
    pos_b = torch.LongTensor(pos['book_idx'].values)
    neg_b = torch.LongTensor(pos['book_neg_idx'].values)

    n_pos = len(pos_u)
    for epoch in range(epochs):
        perm = np.random.permutation(n_pos)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_pos, batch_size):
            end = min(start + batch_size, n_pos)
            idx = perm[start:end]
            u = pos_u[idx].to(device)
            b_pos = pos_b[idx].to(device)
            b_neg = neg_b[idx].to(device)

            score_pos = model(u, b_pos)
            score_neg = model(u, b_neg)
            loss = -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-8).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if verbose and (epoch + 1) % 2 == 0:
            print(f"  Two-Tower epoch {epoch + 1}/{epochs} loss={total_loss / max(n_batches, 1):.4f}")

    # Кэш: пользователь -> множество прочитанных книг (по train)
    user_rated_train = train_ratings.groupby('user_id')['book_id'].apply(set).to_dict()

    def get_neural(user_id: int, N: int = 10) -> List[int]:
        if user_id not in user_to_idx:
            return []
        u_idx = user_to_idx[user_id]
        rated = user_rated_train.get(user_id, set())
        model.eval()
        with torch.no_grad():
            u_t = torch.LongTensor([u_idx]).to(device).expand(n_books)
            b_all = torch.arange(n_books, dtype=torch.long, device=device)
            sc = model(u_t, b_all).cpu().numpy()
        order = np.argsort(-sc)
        out = [idx_to_book[int(i)] for i in order if idx_to_book[int(i)] not in rated][:N]
        return out

    return get_neural


def _fallback_recommender(train_ratings: pd.DataFrame) -> Callable[[int, int], List[int]]:
    """Заглушка: топ популярных книг по числу оценок (если Two-Tower не обучился)."""
    top_books = (
        train_ratings.groupby('book_id').size()
        .sort_values(ascending=False)
        .head(500)
        .index.tolist()
    )
    user_rated = train_ratings.groupby('user_id')['book_id'].apply(set).to_dict()

    def get_fallback(user_id: int, N: int = 10) -> List[int]:
        rated = user_rated.get(user_id, set())
        return [b for b in top_books if b not in rated][:N]
    return get_fallback
