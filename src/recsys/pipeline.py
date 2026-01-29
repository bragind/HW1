# -*- coding: utf-8 -*-
"""
Сквозной пайплайн: от загрузки данных до выдачи рекомендаций и оценки.

Этапы:
1. Загрузка данных (data/raw/)
2. Разбиение train/test, предобработка
3. Построение расширенных признаков (пользователи, книги)
4. Обучение моделей (Popularity, Content, Item-CF, SVD)
5. Опционально: обучение нейросетевой модели (Two-Tower)
6. Гибридная система и генерация рекомендаций
7. Оценка качества (Precision@K, Recall@K, nDCG@K) и анализ по сегментам

Повторный запуск на новых данных: достаточно положить новые CSV в data/raw/
и запустить пайплайн снова.
"""
from typing import Any, Callable, Dict, List, Optional

from .evaluation import run_full_evaluation


def run_pipeline(
    use_neural: bool = False,
    eval_max_users: Optional[int] = 500,
    verbose: bool = True,
    neural_weight: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """
    Запускает сквозной пайплайн: данные → предобработка → модели → гибрид → оценка.

    Параметры
    ---------
    use_neural : bool
        Строить и включать в гибрид Two-Tower модель (требуется torch).
    eval_max_users : int, optional
        Максимум пользователей для расчёта метрик (ускорение оценки).
    verbose : bool
        Выводить прогресс и таблицы метрик.
    neural_weight : float
        Вес нейросетевой модели в гибриде (при use_neural=True).

    Returns
    -------
    dict или None
        Результаты оценки (results_df, by_segment, ...) при return_results=True внутри.
    """
    get_neural: Optional[Callable[[int, int], List[int]]] = None
    nw = neural_weight if use_neural else 0.0

    if use_neural:
        try:
            from .neural import build_two_tower_recommender
            from .data import load_raw_data
            from .evaluation import run_train_test_split
            from .config import RANDOM_STATE, TEST_SIZE

            ratings, books, tags, book_tags = load_raw_data()
            train_ratings, _ = run_train_test_split(
                ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            get_neural = build_two_tower_recommender(train_ratings, verbose=verbose)
        except ImportError as e:
            if verbose:
                print(f'Нейросеть отключена (нет torch): {e}')
            nw = 0.0

    return run_full_evaluation(
        verbose=verbose,
        return_results=True,
        eval_max_users=eval_max_users,
        get_neural=get_neural,
        neural_weight=nw,
    )
