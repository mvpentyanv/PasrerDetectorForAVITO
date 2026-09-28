# Финальная модель (эксперимент E9) и предсказание теста.
#
# Это ранговый бленд LightGBM (среднее по пяти сидам) и CatBoost на полном наборе признаков.
# Выбрал его по OOF P@R>=0.7 (0.822): лучший результат среди кандидатов при сопоставимом std по фолдам.
# Разница с соседними вариантами в пределах шума, но два разных бустинга вместе обычно
# устойчивее на новых данных, чем один.
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import config, models

TIE_EPS = 1e-6


# Смешиваю ранги, а не вероятности: у LightGBM и CatBoost разная калибровка,
# и простое среднее вероятностей фактически дало бы одной модели больший вес.
def _rank01(score: np.ndarray) -> np.ndarray:
    return rankdata(score) / len(score)


# Тот же интерфейс fit_predict, что и у остальных моделей, поэтому финал проверяется той же CV.
def final_fit_predict(X_tr: pd.DataFrame, y_tr, X_va: pd.DataFrame) -> np.ndarray:
    p_lgb = np.mean([models.lgbm(X_tr, y_tr, X_va, seed=s) for s in config.SEEDS], axis=0)
    p_cat = models.catboost(X_tr, y_tr, X_va, seed=config.SEED)
    rank_mean = (_rank01(p_lgb) + _rank01(p_cat)) / 2
    # У среднего двух рангов бывают совпадения (10+20 и 15+15), а метрика склеивает равные score в группу.
    # Крошечная добавка сырых вероятностей разбивает такие ничьи. Шаг ранга при n до 5000 не меньше 1e-4,
    # а добавка не больше 1e-6, так что порядок уже различающихся кук не меняется. Итог остаётся в [0, 1].
    return (1 - TIE_EPS) * rank_mean + TIE_EPS * (p_lgb + p_cat) / 2


# Обучение на всём train и score для теста в формате sample_submission.
def predict_test(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    score = final_fit_predict(X_train, np.asarray(y_train), X_test)
    return pd.DataFrame({"cookie_id": test_ids.to_numpy(), "score": score})
