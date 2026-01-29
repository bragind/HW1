# -*- coding: utf-8 -*-
"""
Метрики качества рекомендаций: Precision@K, Recall@K, nDCG@K.
Релевантность задаётся порогом оценки (например, rating >= 4).
"""
from typing import List, Set

import numpy as np


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Precision@K: доля релевантных среди первых K рекомендаций.
    """
    rec_k = set(recommended[:k])
    if not relevant:
        return 0.0
    return len(rec_k & relevant) / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Recall@K: доля релевантных книг, попавших в топ-K рекомендаций.
    """
    rec_k = set(recommended[:k])
    if not relevant:
        return 0.0
    return len(rec_k & relevant) / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    nDCG@K: учитывает порядок рекомендаций.
    """
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    dcg = sum(
        (1 if item in relevant else 0) / np.log2(i + 2)
        for i, item in enumerate(rec_k)
    )
    ideal_len = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
    if idcg <= 0:
        return 0.0
    return dcg / idcg

