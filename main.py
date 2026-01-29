"""Тонкий CLI-скрипт для запуска эксперимента из пакета src.recsys.

Использование:
  python main.py           — полный эксперимент (обучение, оценка, вывод)
  python main.py --eda     — только EDA (графики в reports/figures/)
  python main.py --report  — EDA + эксперимент + формирование отчёта reports/REPORT.md
  python main.py --log     — то же, что без флага, но с записью вывода в logs/
  python main.py --report --log  — отчёт + логи в файл
"""

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.recsys.cli import run_all, run_eda_only, run_report


@contextmanager
def tee_to_log(log_dir: str = "logs"):
    """
    Дублирует stdout и stderr в файл в каталоге logs/ (имя с датой и временем).
    После выхода из контекста вывод снова идёт только в консоль.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_path / f"run_{stamp}.log"

    class Tee:
        def __init__(self, stream, f):
            self._stream = stream
            self._file = f

        def write(self, data):
            self._stream.write(data)
            self._stream.flush()
            if self._file:
                self._file.write(data)
                self._file.flush()

        def flush(self):
            self._stream.flush()
            if self._file:
                self._file.flush()

    with open(log_file, "w", encoding="utf-8") as f:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = Tee(orig_stdout, f)
        sys.stderr = Tee(orig_stderr, f)
        try:
            yield log_file
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    print(f"\nЛог сохранён: {log_file.resolve()}")


if __name__ == '__main__':
    use_log = '--log' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--log']

    def run():
        if '--report' in args:
            run_report()
        elif '--eda' in args:
            run_eda_only()
        else:
            run_all()

    if use_log:
        with tee_to_log():
            run()
    else:
        run()
