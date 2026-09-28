# Модели в едином формате fit_predict(X_tr, y_tr, X_va) -> score.
# Один интерфейс для всех, чтобы validate.cross_validate мог гонять любую модель одинаково,
# а всё обучение гарантированно шло только на train-части фолда.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED


# Модель из quickstart организаторов без изменений, даже seed оставил их (0), чтобы сравнение было честным.
def quickstart_rf(X_tr, y_tr, X_va) -> np.ndarray:
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0)
    return model.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]


# Линейный baseline. Масштабирование внутри пайплайна, так что оно тоже обучается только на train-части.
def logreg(X_tr, y_tr, X_va) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    return model.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]


# --------------------------------------------------------------------------- бустинги
# Бустинг выбрал потому, что он сам работает с NaN (у меня это "не применимо", например курсор на мобильных),
# с категориальными признаками и с взаимодействиями вроде "курсор важен только на web".
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostClassifier

# Параметры намеренно консервативные: мелкие деревья и сэмплирование строк и колонок, потому что ботов всего 899.
# Optuna на отдельном разбиении прироста не дала (эксперимент E10), поэтому остались эти.
# Классы не перевзвешиваю: метрика смотрит только на порядок кук, калибровка тут не важна.
LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1,
)
CAT_PARAMS = dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3, verbose=0, thread_count=-1)


# LightGBM сам понимает pandas category, поэтому категориальные колонки отдаю как есть.
def lgbm(X_tr, y_tr, X_va, seed: int = SEED, **params) -> np.ndarray:
    model = lgb.LGBMClassifier(**{**LGB_PARAMS, **params}, random_state=seed)
    return model.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]


# CatBoost кодирует категории через ordered target statistics: для каждой строки считает статистику
# только по предыдущим строкам, поэтому таргет текущей строки не протекает. На вход ему нужны строки.
def catboost(X_tr, y_tr, X_va, seed: int = SEED, **params) -> np.ndarray:
    cat_cols = [c for c in X_tr.columns if isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
    to_str = lambda X: X.astype({c: str for c in cat_cols})
    model = CatBoostClassifier(**{**CAT_PARAMS, **params}, random_seed=seed, cat_features=cat_cols)
    return model.fit(to_str(X_tr), y_tr).predict_proba(to_str(X_va))[:, 1]
