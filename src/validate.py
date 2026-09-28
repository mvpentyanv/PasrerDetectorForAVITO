# Схема валидации.
#
# Основная: StratifiedKFold(5, shuffle, SEED), метрика по OOF-предсказаниям плюс среднее и std по фолдам.
# Контрольная: time-holdout, учусь на окнах до 17 апреля и проверяю на последних трёх днях train.
#
# Почему основная не по времени, хотя тест лежит позже трейна: в EDA сдвига не нашлось
# (adversarial ROC-AUC train против test 0.50, доля ботов стабильна по дням). Тогда стратифицированные
# фолды честнее: они используют всех 899 ботов и дают оценку разброса. Holdout оставляю как проверку,
# похожую на лидерборд, но решений по нему не принимаю: там всего 160 ботов и метрика сильно шумит.
#
# Модель передаётся функцией fit_predict(X_tr, y_tr, X_va) -> score. Всё, что зависит от таргета,
# происходит внутри неё и видит только обучающую часть фолда. Так утечку через кодирование
# или подбор просто некуда пронести.
from __future__ import annotations

import csv
import time
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config
from src.metrics import evaluate

FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame], np.ndarray]

HOLDOUT_START = pd.Timestamp("2026-04-17", tz="UTC")  # последние 3 дня train
EXPERIMENTS_PATH = config.ROOT / "reports" / "experiments.csv"


def stratified_folds(y, n_splits: int = config.N_FOLDS, seed: int = config.SEED):
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def time_holdout_split(window_start: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    is_valid = (window_start >= HOLDOUT_START).to_numpy()
    return np.flatnonzero(~is_valid), np.flatnonzero(is_valid)


def cross_validate(fit_predict: FitPredict, X: pd.DataFrame, y, window_start: pd.Series | None = None,
                   verbose: bool = True) -> dict:
    y = np.asarray(y, dtype=int)
    oof = np.full(len(y), np.nan)
    fold_scores = []
    t0 = time.time()
    for k, (tr, va) in enumerate(stratified_folds(y)):
        oof[va] = fit_predict(X.iloc[tr], y[tr], X.iloc[va])
        fold_scores.append(evaluate(y[va], oof[va]))
        if verbose:
            print(f"  fold {k}: " + ", ".join(f"{m}={v:.4f}" for m, v in fold_scores[-1].items()))

    # главная цифра это метрика по всему OOF сразу; среднее и std по фолдам нужны, чтобы видеть шум
    folds = pd.DataFrame(fold_scores)
    res = {"oof": oof, "folds": folds, **{f"oof_{m}": v for m, v in evaluate(y, oof).items()},
           **{f"{m}_mean": v for m, v in folds.mean().items()},
           **{f"{m}_std": v for m, v in folds.std(ddof=0).items()}}

    if window_start is not None:
        tr, va = time_holdout_split(window_start)
        pred = fit_predict(X.iloc[tr], y[tr], X.iloc[va])
        res["holdout_pred"] = pred
        res.update({f"holdout_{m}": v for m, v in evaluate(y[va], pred).items()})

    res["seconds"] = time.time() - t0
    if verbose:
        print(summary_line(res))
    return res


def summary_line(res: dict) -> str:
    s = (f"OOF P@R70={res['oof_p_at_r70']:.4f} (folds {res['p_at_r70_mean']:.4f}±{res['p_at_r70_std']:.4f}) | "
         f"PR-AUC={res['oof_pr_auc']:.4f} | ROC-AUC={res['oof_roc_auc']:.4f}")
    if "holdout_p_at_r70" in res:
        s += f" | holdout P@R70={res['holdout_p_at_r70']:.4f}"
    return s


# Каждый прогон дописываю строкой в reports/experiments.csv, чтобы история экспериментов
# не жила только в выводе терминала. Из этого файла собрана таблица в README.
def log_experiment(name: str, features: str, res: dict, path=EXPERIMENTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": name, "features": features,
        "oof_p_at_r70": res["oof_p_at_r70"], "p_at_r70_mean": res["p_at_r70_mean"], "p_at_r70_std": res["p_at_r70_std"],
        "oof_pr_auc": res["oof_pr_auc"], "oof_roc_auc": res["oof_roc_auc"],
        "holdout_p_at_r70": res.get("holdout_p_at_r70", np.nan),
    }
    row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
