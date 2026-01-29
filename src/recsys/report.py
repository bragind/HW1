# -*- coding: utf-8 -*-
"""
Генерация отчёта с визуализацией (REPORT.md в reports/).

- Этап 1: EDA и проблемы данных (графики в figures/)
- Модели: Popularity, Content-Based, Item-CF, SVD, Гибрид
- Сводная таблица метрик (Precision@K, Recall@K, nDCG@K)
- Анализ по сегментам пользователей
- Выводы и гибридная стратегия
"""
import base64
from pathlib import Path
from typing import Any, Dict, Optional

from .config import REPORTS_DIR, FIGURES_DIR
from .eda import run_eda
from .evaluation import run_full_evaluation


def _section(title: str, level: int = 1) -> str:
    return "\n" + "#" * level + " " + title + "\n\n"


def _embed_figure_as_base64(figures_dir: Path, filename: str) -> str:
    """Читает PNG из figures_dir и возвращает data URI для вставки в Markdown."""
    path = figures_dir / filename
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _write_eda_section(eda_results: Dict[str, Any], figures_dir: Optional[Path] = None) -> str:
    out = []
    out.append(_section("Этап 1: Знакомство с данными и EDA", 1))

    out.append("## Загруженные данные\n\n")
    out.append(f"- ratings: {eda_results.get('ratings_shape', (0, 0))}\n")
    out.append(f"- books: {eda_results.get('books_shape', (0, 0))}\n")
    out.append(f"- tags: {eda_results.get('tags_shape', (0, 0))}\n")
    out.append(f"- book_tags: {eda_results.get('book_tags_shape', (0, 0))}\n\n")

    fig_dir = figures_dir or Path(FIGURES_DIR)

    def _img(alt: str, filename: str) -> str:
        uri = _embed_figure_as_base64(fig_dir, filename)
        if uri:
            return f"![{alt}]({uri})\n\n"
        return f"![{alt}](figures/{filename})\n\n"

    r = eda_results.get("ratings_distribution", {})
    out.append("## Распределение оценок\n\n")
    out.append(f"- Средняя оценка: {r.get('mean_rating', 0):.2f}\n")
    out.append(f"- Доля оценок ≥ 4: {r.get('share_high_ratings_ge4', 0):.2%}\n")
    out.append(f"- **Вывод:** {r.get('conclusion', '')}\n\n")
    out.append(_img("Распределение оценок", "ratings_distribution.png"))

    u = eda_results.get("user_activity", {})
    out.append("## Активность пользователей\n\n")
    out.append(f"- Число пользователей: {u.get('n_users', 0)}\n")
    out.append(f"- Медиана оценок на пользователя: {u.get('ratings_per_user_median', 0):.0f}\n")
    out.append(f"- Холодные пользователи (≤5 оценок): {u.get('cold_users_le5', 0)}\n")
    out.append(f"- **Вывод:** {u.get('conclusion', '')}\n\n")
    out.append(_img("Активность пользователей", "user_activity.png"))

    b = eda_results.get("book_popularity", {})
    out.append("## Популярность книг\n\n")
    out.append(f"- Число книг: {b.get('n_books', 0)}\n")
    out.append(f"- Медиана оценок на книгу: {b.get('ratings_per_book_median', 0):.0f}\n")
    out.append(f"- **Вывод:** {b.get('conclusion', '')}\n\n")
    out.append(_img("Популярность книг", "book_popularity.png"))

    t = eda_results.get("top_tags", {})
    out.append("## Самые частые теги\n\n")
    out.append(f"- **Вывод:** {t.get('conclusion', '')}\n\n")
    out.append(_img("Топ тегов", "top_tags.png"))

    issues = eda_results.get("data_issues", {})
    out.append("## Основные проблемы данных\n\n")
    out.append(f"- **Разреженность:** {issues.get('sparsity_note', '')}\n")
    out.append(f"- **Смещение популярности:** {issues.get('popularity_bias_note', '')}\n\n")

    return "".join(out)


def _write_models_section() -> str:
    out = []
    out.append(_section("Модели рекомендаций и гибрид", 1))
    out.append("## Popularity\n\n")
    out.append("Топ-N по среднему рейтингу с порогом минимального количества оценок.\n\n")
    out.append("## Content-Based (TF-IDF)\n\n")
    out.append("Профиль книги: original_title + теги. Векторизация TF-IDF, похожие книги по косинусной близости.\n\n")
    out.append("## Item-Based CF\n\n")
    out.append("Матрица user×book (неявный feedback при rating ≥ 4). Попарная схожесть книг, предсказание — взвешенное среднее по K соседям.\n\n")
    out.append("## SVD (Matrix Factorization)\n\n")
    out.append("Библиотека Surprise, train/test split. Топ-N по предсказанному рейтингу.\n\n")
    out.append("## Гибридная система\n\n")
    out.append("Взвешенное объединение рекомендаций всех моделей. Для холодных пользователей увеличен вес популярности. ")
    out.append("Баланс разнообразия и релевантности, фильтрация уже прочитанных книг.\n\n")
    return "".join(out)


def _write_metrics_section(exp_results: Dict[str, Any]) -> str:
    out = []
    out.append(_section("Оценка и сравнение моделей", 1))
    out.append("Оценка на отложенной тестовой выборке. Релевантные книги: оценка ≥ 4.\n\n")
    out.append(f"- K для метрик: {exp_results.get('eval_k', 5)}\n")
    out.append(f"- Размер обучающей выборки: {exp_results.get('train_size', 0)}\n")
    out.append(f"- Размер тестовой выборки: {exp_results.get('test_size', 0)}\n")
    out.append(f"- Число пользователей для расчёта метрик: {exp_results.get('n_eval_users', 0)}\n\n")
    out.append("### Сводная таблица метрик\n\n")
    results_df = exp_results.get("results_df")
    if results_df is not None:
        try:
            table_str = results_df.round(4).to_markdown()
        except (ImportError, AttributeError):
            table_str = results_df.round(4).to_string()
        out.append(table_str)
        out.append("\n\n")
    sample_user = exp_results.get("sample_user")
    sample_hybrid = exp_results.get("sample_hybrid_recommendations", [])
    if sample_user is not None and sample_hybrid:
        out.append("### Пример гибридных рекомендаций\n\n")
        out.append(f"Для user_id={sample_user}: {sample_hybrid}\n\n")
    return "".join(out)


def _write_segments_section(exp_results: Dict[str, Any]) -> str:
    import pandas as pd
    out = []
    out.append(_section("Анализ по сегментам пользователей", 1))
    out.append("Метрики по сегментам: cold (≤5 оценок), regular, active (≥P90).\n\n")
    by_segment = exp_results.get("by_segment", {})
    for seg, metrics in by_segment.items():
        if not metrics:
            continue
        out.append(f"### {seg}\n\n")
        df = pd.DataFrame(metrics).T
        try:
            out.append(df.round(4).to_markdown() + "\n\n")
        except (ImportError, AttributeError):
            out.append(df.round(4).to_string() + "\n\n")
    return "".join(out)


def _write_conclusions_section() -> str:
    out = []
    out.append(_section("Выводы", 1))
    out.append("## Гибридная стратегия\n\n")
    out.append("Гибрид объединяет Popularity, Content-Based, Item-CF и SVD с весами. ")
    out.append("Для холодных пользователей увеличен вес популярности. ")
    out.append("Пул кандидатов формируется из рекомендаций всех моделей с балансом разнообразия и релевантности.\n\n")
    out.append("## Сильные и слабые стороны\n\n")
    out.append("- **Popularity:** устойчивость, нет персонализации.\n")
    out.append("- **Content-Based:** холодный старт для новых книг; не использует коллаборативные сигналы.\n")
    out.append("- **Item-CF:** интерпретируемость; сложность по памяти.\n")
    out.append("- **SVD:** высокое качество; холодный старт для новых пользователей/книг.\n")
    out.append("- **Гибрид:** баланс персонализации и разнообразия, устойчивость к холодному старту.\n\n")
    out.append("## Идеи улучшения\n\n")
    out.append("- Two-Tower или Wide & Deep для нейросетевых эмбеддингов.\n")
    out.append("- Оптимизация весов гибрида по сетке (evaluation.optimize_hybrid_weights_grid).\n")
    out.append("- Более богатые признаки пользователей и книг.\n\n")
    return "".join(out)


def generate_report(
    report_path: Optional[str] = None,
    figures_dir: Optional[str] = None,
    run_eda_first: bool = True,
    verbose: bool = True,
    eval_max_users: Optional[int] = 500,
) -> str:
    """
    Запускает EDA (опционально), полную оценку моделей и гибрида, формирует REPORT.md.

    Returns
    -------
    str
        Путь к сохранённому файлу отчёта.
    """
    report_path = report_path or str(Path(REPORTS_DIR) / "REPORT.md")
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append("# Отчёт: гибридная система рекомендаций книг (RSHW2)\n\n")
    parts.append("Проделанная работа: EDA, расширенные признаки, модели (Popularity, Content, Item-CF, SVD), ")
    parts.append("гибридная система, оценка метрик и анализ по сегментам пользователей.\n\n")

    figs_dir = Path(figures_dir or FIGURES_DIR)
    if run_eda_first:
        if verbose:
            print("Запуск EDA...")
        eda_results = run_eda(figures_dir=str(figs_dir))
        parts.append(_write_eda_section(eda_results, figures_dir=figs_dir))
    else:
        eda_results = {}
        parts.append(_write_eda_section(eda_results, figures_dir=figs_dir))

    if verbose:
        print("Запуск оценки моделей и гибрида...")
    exp_results = run_full_evaluation(verbose=verbose, return_results=True, eval_max_users=eval_max_users)
    if not exp_results:
        exp_results = {}

    parts.append(_write_models_section())
    parts.append(_write_metrics_section(exp_results))
    parts.append(_write_segments_section(exp_results))
    parts.append(_write_conclusions_section())

    report_content = "".join(parts)
    Path(report_path).write_text(report_content, encoding="utf-8")

    if verbose:
        print(f"Отчёт сохранён: {report_path}")
    return report_path
