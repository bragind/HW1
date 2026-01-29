# -*- coding: utf-8 -*-
"""
Этап 6: Генерация отчёта с выводами о проделанной работе.

Скрипт формирует отчёт (REPORT.md) в каталоге reports/:
- Этап 1: результаты EDA и проблемы данных
- Этапы 2–4: описание моделей
- Этап 5: сводная таблица метрик (Precision@K, Recall@K, nDCG@K, RMSE)
- Этап 6: гибридная стратегия, выводы, идеи улучшения
"""
from pathlib import Path
from typing import Any, Dict, Optional

from .config import REPORTS_DIR, FIGURES_DIR
from .eda import run_eda
from .evaluation import run_experiment


def _section(title: str, level: int = 1) -> str:
    return "\n" + "#" * level + " " + title + "\n\n"


def _write_eda_section(eda_results: Dict[str, Any]) -> str:
    out = []
    out.append(_section("Этап 1: Знакомство с данными и EDA", 1))

    out.append("## Загруженные данные\n\n")
    out.append(f"- ratings: {eda_results.get('ratings_shape', (0, 0))}\n")
    out.append(f"- books: {eda_results.get('books_shape', (0, 0))}\n")
    out.append(f"- tags: {eda_results.get('tags_shape', (0, 0))}\n")
    out.append(f"- book_tags: {eda_results.get('book_tags_shape', (0, 0))}\n\n")

    # Относительный путь от reports/REPORT.md к reports/figures/
    fig = "figures"

    r = eda_results.get("ratings_distribution", {})
    out.append("## Распределение оценок\n\n")
    out.append(f"- Средняя оценка: {r.get('mean_rating', 0):.2f}\n")
    out.append(f"- Доля оценок ≥ 4: {r.get('share_high_ratings_ge4', 0):.2%}\n")
    out.append(f"- **Вывод:** {r.get('conclusion', '')}\n\n")
    out.append(f"![Распределение оценок]({fig}/ratings_distribution.png)\n\n")

    u = eda_results.get("user_activity", {})
    out.append("## Активность пользователей\n\n")
    out.append(f"- Число пользователей: {u.get('n_users', 0)}\n")
    out.append(f"- Медиана оценок на пользователя: {u.get('ratings_per_user_median', 0):.0f}\n")
    out.append(f"- Холодные пользователи (≤5 оценок): {u.get('cold_users_le5', 0)}\n")
    out.append(f"- **Вывод:** {u.get('conclusion', '')}\n\n")
    out.append(f"![Активность пользователей]({fig}/user_activity.png)\n\n")

    b = eda_results.get("book_popularity", {})
    out.append("## Популярность книг\n\n")
    out.append(f"- Число книг: {b.get('n_books', 0)}\n")
    out.append(f"- Медиана оценок на книгу: {b.get('ratings_per_book_median', 0):.0f}\n")
    out.append(f"- **Вывод:** {b.get('conclusion', '')}\n\n")
    out.append(f"![Популярность книг]({fig}/book_popularity.png)\n\n")

    t = eda_results.get("top_tags", {})
    out.append("## Самые частые теги\n\n")
    out.append(f"- **Вывод:** {t.get('conclusion', '')}\n\n")
    out.append(f"![Топ тегов]({fig}/top_tags.png)\n\n")

    issues = eda_results.get("data_issues", {})
    out.append("## Основные проблемы данных (разреженность, смещение популярности)\n\n")
    out.append(f"- **Разреженность:** {issues.get('sparsity_note', '')}\n")
    out.append(f"- **Смещение популярности:** {issues.get('popularity_bias_note', '')}\n\n")

    return "".join(out)


def _write_models_section() -> str:
    out = []
    out.append(_section("Этапы 2–4: Модели рекомендаций", 1))
    out.append("## Неперсонализированная модель (Popularity)\n\n")
    out.append("Топ-N по среднему рейтингу с порогом минимального количества оценок.\n\n")
    out.append("## Контентная модель\n\n")
    out.append("Профиль книги: original_title + теги из book_tags. Векторизация TF-IDF, ")
    out.append("get_similar_books(book_id, N=5) по косинусной близости.\n\n")
    out.append("## Item-Based CF\n\n")
    out.append("Матрица user×book (неявный feedback: 1 при rating ≥ 4). ")
    out.append("Попарная схожесть книг (косинус). Предсказание — взвешенное среднее по K соседям. ")
    out.append("Сложность: O(n_books²) по памяти для матрицы схожестей; оптимизации: sparse топ-K соседей, LSH/ANN.\n\n")
    out.append("## SVD (Matrix Factorization)\n\n")
    out.append("Библиотека Surprise, train/test split, RMSE на тесте. ")
    out.append("get_recommendations(user_id, N=5) — топ-N по предсказанному рейтингу.\n\n")
    return "".join(out)


def _write_metrics_section(exp_results: Dict[str, Any]) -> str:
    out = []
    out.append(_section("Этап 5: Оценка и сравнение моделей", 1))
    out.append("Оценка на отложенной тестовой выборке (не участвовавшей в обучении). ")
    out.append(f"Релевантные книги: оценка ≥ 4. K = {exp_results.get('eval_k', 5)}.\n\n")
    out.append(f"- Размер обучающей выборки: {exp_results.get('train_size', 0)}\n")
    out.append(f"- Размер тестовой выборки: {exp_results.get('test_size', 0)}\n")
    out.append(f"- Число пользователей для расчёта метрик: {exp_results.get('n_eval_users', 0)}\n\n")
    out.append("**RMSE (SVD на тесте):** " + f"{exp_results.get('rmse', 0):.4f}\n\n")
    out.append("### Сводная таблица метрик\n\n")
    results_df = exp_results.get("results_df")
    if results_df is not None:
        try:
            table_str = results_df.round(4).to_markdown()
        except ImportError:
            table_str = results_df.round(4).to_string()
        out.append(table_str)
        out.append("\n\n")
    return "".join(out)


def _write_conclusions_section(exp_results: Dict[str, Any]) -> str:
    out = []
    out.append(_section("Этап 6: Гибридизация и выводы", 1))

    out.append("## Гибридная стратегия (холодный старт и устойчивость)\n\n")
    out.append("Для **новых книг** (мало оценок) используем контентные рекомендации (TF-IDF по тегам и названию). ")
    out.append("Для **известных пользователей** — SVD как основу; при отсутствии предсказания — Popularity. ")
    out.append("Гибрид: объединение списков SVD и Popularity с удалением дубликатов повышает устойчивость и разнообразие.\n\n")

    sample_user = exp_results.get("sample_user_id")
    sample_hybrid = exp_results.get("sample_hybrid_recommendations", [])
    out.append(f"Пример гибридных рекомендаций для user_id={sample_user}: {sample_hybrid}\n\n")

    out.append("## Выводы по работе\n\n")
    out.append("### Какая модель показала наилучшее качество и почему?\n\n")
    out.append("SVD (матричная факторизация) обычно даёт наилучшие Precision@K, Recall@K и nDCG@K, ")
    out.append("так как обучается предсказывать рейтинги и учитывает латентные факторы пользователей и книг. ")
    out.append("RMSE на тесте отражает точность предсказания оценок.\n\n")

    out.append("### Сильные и слабые стороны моделей\n\n")
    out.append("- **Popularity:** сильная — устойчивость, нет холодного старта по книгам; слабая — нет персонализации.\n")
    out.append("- **Content-Based:** сильная — холодный старт для новых книг, интерпретируемость; слабая — не использует коллаборативные сигналы.\n")
    out.append("- **Item-Based CF:** сильная — интерпретируемость («похожие книги»); слабая — сложность O(n_books²), холодный старт для новых книг.\n")
    out.append("- **SVD:** сильная — высокое качество, масштабируемость; слабая — холодный старт для новых пользователей/книг, чёрный ящик.\n\n")

    out.append("### Как можно улучшить систему в дальнейшем?\n\n")
    out.append("- Использовать более сложные эмбеддинги для текста (Word2Vec, BERT) в контентной модели.\n")
    out.append("- Добавить фичи пользователей (демография, история) в гибрид или двухбашенные модели.\n")
    out.append("- Реализовать нейросетевые подходы (NCF, two-tower, трансформеры).\n")
    out.append("- Учитывать разнообразие и справедливость рекомендаций (diversity, fairness).\n")
    out.append("- Использовать LightFM или аналоги для объединения контента и коллаборативных сигналов.\n\n")

    return "".join(out)


def generate_report(
    report_path: Optional[str] = None,
    figures_dir: Optional[str] = None,
    run_eda_first: bool = True,
    verbose: bool = True,
) -> str:
    """
    Запускает EDA (опционально), эксперимент и формирует отчёт REPORT.md.

    Returns
    -------
    str
        Путь к сохранённому файлу отчёта.
    """
    report_path = report_path or str(Path(REPORTS_DIR) / "REPORT.md")
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append("# Отчёт по проекту: рекомендательная система книг (Goodbooks-10k)\n\n")
    parts.append("Проделанная работа по этапам: EDA, базовые и контентные модели, ")
    parts.append("коллаборативная фильтрация, матричные разложения, оценка и гибридизация.\n\n")

    if run_eda_first:
        if verbose:
            print("Запуск EDA...")
        eda_results = run_eda(figures_dir=figures_dir or FIGURES_DIR)
        parts.append(_write_eda_section(eda_results))
    else:
        eda_results = {}

    if verbose:
        print("Запуск эксперимента (обучение и оценка моделей)...")
    exp_results = run_experiment(verbose=verbose, return_results=True)
    if not exp_results:
        exp_results = {}

    parts.append(_write_models_section())
    parts.append(_write_metrics_section(exp_results))
    parts.append(_write_conclusions_section(exp_results))

    report_content = "".join(parts)
    Path(report_path).write_text(report_content, encoding="utf-8")

    if verbose:
        print(f"Отчёт сохранён: {report_path}")
    return report_path
