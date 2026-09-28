# Метрика задачи: Precision при Recall >= 0.7, плюс PR-AUC и ROC-AUC для диагностики.
#
# Реализую строго по определению, а не через sklearn precision_recall_curve:
#   1. пороги это уникальные значения score по убыванию;
#   2. при пороге t отмечаются все куки со score >= t, то есть группа одинаковых score
#      попадает в отметку целиком и разрезать её нельзя;
#   3. ответ это максимальный precision среди порогов, где recall >= 0.7
#      (именно максимум, а не первая точка, где recall достиг 0.7: precision вдоль кривой не монотонен).
#
# Тесты лежат ниже в этом же файле: `pytest src/metrics.py` или `python -m src.metrics`.
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import TARGET_RECALL


# Для каждого порога (уникальный score по убыванию) считает precision и recall.
def precision_recall_by_threshold(y_true, score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.shape != score.shape:
        raise ValueError("y_true и score разной длины")
    if np.isnan(score).any():
        raise ValueError("score содержит NaN")

    # np.unique сортирует по возрастанию, inverse даёт номер группы одинаковых score для каждой куки
    thresholds, group = np.unique(score, return_inverse=True)
    pos_in_group = np.bincount(group, weights=y_true, minlength=len(thresholds))
    size_of_group = np.bincount(group, minlength=len(thresholds))

    # иду от самого большого score к меньшему и накапливаю группы целиком, так ties не делятся
    tp = np.cumsum(pos_in_group[::-1])
    predicted = np.cumsum(size_of_group[::-1])
    fp = predicted - tp

    n_pos = y_true.sum()
    precision = tp / (tp + fp)
    recall = tp / n_pos if n_pos > 0 else np.full_like(tp, np.nan, dtype=float)
    return thresholds[::-1], precision, recall


def precision_at_recall(y_true, score, min_recall: float = TARGET_RECALL) -> float:
    if np.asarray(y_true).sum() == 0:
        return float("nan")
    _, precision, recall = precision_recall_by_threshold(y_true, score)
    ok = recall >= min_recall
    # пустым ok быть не может: самый низкий порог отмечает всех и даёт recall = 1
    return float(precision[ok].max())


def evaluate(y_true, score) -> dict[str, float]:
    return {
        "p_at_r70": precision_at_recall(y_true, score),
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
    }


# ----------------------------------------------------------------------------- тесты

def test_toy_with_ties():
    # score:  0.9 | 0.8 0.8 0.8 | 0.5 0.5 | 0.1
    # y:       1  |  1   0   1  |  1   0  |  0      (4 позитива)
    y = [1, 1, 0, 1, 1, 0, 0]
    s = [0.9, 0.8, 0.8, 0.8, 0.5, 0.5, 0.1]
    thr, p, r = precision_recall_by_threshold(y, s)
    np.testing.assert_allclose(thr, [0.9, 0.8, 0.5, 0.1])
    np.testing.assert_allclose(p, [1 / 1, 3 / 4, 4 / 6, 4 / 7])
    np.testing.assert_allclose(r, [1 / 4, 3 / 4, 4 / 4, 4 / 4])
    # recall >= 0.7 у порогов 0.8, 0.5, 0.1, лучший precision среди них 0.75
    assert precision_at_recall(y, s) == 0.75


def test_ties_cannot_be_split():
    # если бы группу 0.5 можно было разрезать и сначала взять позитивы, получилось бы 1.0
    assert precision_at_recall([1, 1, 0, 0], [0.5] * 4) == 0.5
    assert precision_at_recall([0, 0, 1, 1], [0.5] * 4) == 0.5  # и порядок строк ни на что не влияет


def test_max_not_first_feasible_point():
    # пример из официального metric.py: первая точка с recall >= 0.7 даёт 0.75, правильный ответ 0.80
    assert precision_at_recall([1, 0, 1, 1, 1], [5, 4, 3, 2, 1]) == 0.8


# Главная проверка: на случайных данных с кучей одинаковых score моя реализация
# должна совпадать с официальной до последнего знака.
def test_matches_official_on_random_ties():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metric import precision_at_recall as official

    rng = np.random.default_rng(0)
    for _ in range(300):
        n = rng.integers(5, 400)
        y = rng.random(n) < rng.uniform(0.05, 0.6)
        if y.sum() == 0:
            continue
        # грубое округление даёт много одинаковых score
        s = np.round(rng.random(n) + y * rng.uniform(0, 1), rng.integers(0, 3))
        assert precision_at_recall(y, s) == official(y, s)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
