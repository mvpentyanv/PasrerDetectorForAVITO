# Проверка submission.csv перед отправкой. Попыток на лидерборде всего семь,
# тратить их на битый файл обидно, поэтому проверяю всё, что можно проверить локально.
# Запуск: `python -m src.check_submission [путь]`, при ошибках код возврата 1.
import sys
from pathlib import Path

import pandas as pd

from src import config

MAX_TIE_SHARE = 0.01   # предупреждаю, если в самой большой группе равных score больше 1% кук


# Возвращает список ошибок (пустой, если всё хорошо), предупреждения просто печатает.
def check_submission(path: Path = config.SUBMISSION_PATH) -> list[str]:
    errors = []
    sub = pd.read_csv(path)
    sample = pd.read_csv(config.SAMPLE_SUB_PATH, nrows=0)
    test = pd.read_csv(config.TEST_PATH, usecols=["cookie_id"])

    # заголовок и порядок колонок должны быть ровно как в образце
    if list(sub.columns) != list(sample.columns):
        errors.append(f"колонки {list(sub.columns)} != {list(sample.columns)} (как в sample_submission)")
        return errors

    # набор кук должен совпадать с test один в один: без пропусков, дублей и лишних
    ids, expected = sub.cookie_id, set(test.cookie_id)
    if ids.isna().any():
        errors.append(f"пустые cookie_id: {ids.isna().sum()}")
    if ids.duplicated().any():
        errors.append(f"дубликаты cookie_id: {ids.duplicated().sum()}")
    missing, extra = expected - set(ids), set(ids) - expected
    if missing:
        errors.append(f"нет {len(missing)} cookie_id из test.csv")
    if extra:
        errors.append(f"лишние cookie_id (нет в test.csv): {len(extra)}")

    score = pd.to_numeric(sub.score, errors="coerce")
    if score.isna().any():
        errors.append(f"score не число или NaN: {score.isna().sum()}")
    elif not score.between(0, 1).all():
        errors.append(f"score вне [0, 1]: {(~score.between(0, 1)).sum()}")

    # ties не ошибка, но метрика отмечает одинаковые score только целой группой,
    # и большая группа может заметно просадить precision
    if not errors:
        biggest_tie = score.value_counts().iloc[0]
        n_unique = score.nunique()
        print(f"строк: {len(sub)}, уникальных score: {n_unique}, самая большая группа равных score: {biggest_tie}")
        if biggest_tie / len(sub) > MAX_TIE_SHARE:
            print(f"ПРЕДУПРЕЖДЕНИЕ: много одинаковых score ({biggest_tie} кук) — такие куки метрика "
                  f"отмечает только все вместе")
    return errors


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else config.SUBMISSION_PATH
    errs = check_submission(path)
    for e in errs:
        print("ОШИБКА:", e)
    print("submission OK" if not errs else "submission НЕ прошёл проверку")
    sys.exit(1 if errs else 0)
