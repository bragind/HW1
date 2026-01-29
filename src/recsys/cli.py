# -*- coding: utf-8 -*-
"""
CLI-обёртки для запуска экспериментов и отчётов.
"""
from .eda import run_eda
from .evaluation import run_experiment
from .report import generate_report


def run_all() -> None:
    """
    Запустить полный цикл: обучение, оценка, вывод в консоль.
    """
    run_experiment(verbose=True, return_results=False)


def run_eda_only() -> None:
    """
    Запустить только EDA (Этап 1): графики в reports/figures/, вывод в консоль.
    """
    run_eda()


def run_report() -> str:
    """
    Сформировать отчёт: EDA + эксперимент + REPORT.md в reports/.
    Возвращает путь к сохранённому отчёту.
    """
    return generate_report(run_eda_first=True, verbose=True)

