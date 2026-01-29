# -*- coding: utf-8 -*-
"""CLI-обёртки для запуска EDA, эксперимента и отчёта."""
from .eda import run_eda
from .evaluation import run_full_evaluation
from .report import generate_report


def run_all() -> None:
    """Полный цикл: обучение, оценка моделей и гибрида, вывод в консоль."""
    run_full_evaluation(verbose=True, return_results=False)


def run_eda_only() -> None:
    """Только EDA: графики в reports/figures/."""
    run_eda()


def run_report(eval_max_users: int = 500) -> str:
    """EDA + оценка моделей и гибрида + формирование REPORT.md в reports/. Возвращает путь к отчёту."""
    return generate_report(
        run_eda_first=True,
        verbose=True,
        eval_max_users=eval_max_users,
    )
