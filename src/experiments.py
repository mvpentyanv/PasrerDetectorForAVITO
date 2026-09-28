# Эксперименты: baselines, бустинги, вторая версия признаков, подбор параметров, важность признаков.
# Запуск: `python -m src.experiments <stage>`. Каждый прогон печатается и дописывается в reports/experiments.csv.
# Для финального сабмита этот файл не нужен, это "лаборатория", из которой выросли решения в train.py.
import sys

import numpy as np
import pandas as pd

from src import config, data, features, models, validate
from src.metrics import evaluate

# признаки строятся около 45 секунд, для серий экспериментов кэширую их на диск
CACHE = config.ROOT / "cache" / "features.pkl"
# колонки, которые целиком завязаны на user agent: с ними проверяю, не держится ли модель на "ярлыках"
UA_COLS = ["ua_kind", "ua_browser", "ua_version", "ua_os", "ua_len", "ua_freq", "n_user_agents", "ua_changed"]


def load_prepared():
    tr, te, ev = data.load_train(), data.load_test(), data.load_events()
    ev, n_dup = data.prepare_events(ev, pd.concat([tr, te]))
    return tr, te, ev, n_dup


def run(name, feat_desc, fit_predict, X, tr):
    print(f"== {name} [{feat_desc}], {X.shape[1]} признаков")
    res = validate.cross_validate(fit_predict, X, tr.target.values, tr.window_start_ts)
    validate.log_experiment(name, feat_desc, res)
    return res


def stage_baselines():
    tr, _, ev, _ = load_prepared()
    run("B0 quickstart RF", "n_events, item_nunique", models.quickstart_rf, features.quickstart_features(ev, tr), tr)
    run("B1 LogReg", "log n_events + доли event_name", models.logreg, features.basic_features(ev, tr), tr)


def load_features(rebuild: bool = False):
    tr, te = data.load_train(), data.load_test()
    if CACHE.exists() and not rebuild:
        F = pd.read_pickle(CACHE)
    else:
        _, _, ev, n_dup = load_prepared()
        F = features.build_features(ev, pd.concat([tr, te], ignore_index=True), n_dup)
        CACHE.parent.mkdir(exist_ok=True)
        F.to_pickle(CACHE)
    return tr, te, F.loc[tr.cookie_id].reset_index(drop=True), F.loc[te.cookie_id].reset_index(drop=True)


# Бленд уже посчитанных OOF: модели обучались на одних и тех же фолдах,
# поэтому их OOF можно усреднять напрямую, без повторного обучения.
def blend(name, feat_desc, results, tr, weights=None):
    from scipy.stats import rankdata
    w = weights or [1] * len(results)
    avg = lambda key: sum(wi * rankdata(r[key]) / len(r[key]) for wi, r in zip(w, results)) / sum(w)
    y = tr.target.values
    oof = avg("oof")
    folds = pd.DataFrame([evaluate(y[va], oof[va]) for _, va in validate.stratified_folds(y)])
    _, va = validate.time_holdout_split(tr.window_start_ts)
    res = {"oof": oof, **{f"oof_{m}": v for m, v in evaluate(y, oof).items()},
           **{f"{m}_mean": v for m, v in folds.mean().items()}, **{f"{m}_std": v for m, v in folds.std(ddof=0).items()},
           **{f"holdout_{m}": v for m, v in evaluate(y[va], avg("holdout_pred")).items()}}
    print(f"== {name}\n" + validate.summary_line(res))
    validate.log_experiment(name, feat_desc, res)
    return res


# Усреднение по сидам снимает часть шума от сэмплирования строк и колонок в бустинге.
def multi_seed(fit_predict, seeds=config.SEEDS):
    return lambda X_tr, y_tr, X_va: np.mean([fit_predict(X_tr, y_tr, X_va, seed=s) for s in seeds], axis=0)


def stage_features_v2():
    tr, _, X, _ = load_features(rebuild=True)
    r_lgb = run("E5 LightGBM", "full v2 (+внутрисессионный темп, смены лок/кат, контакты)", models.lgbm, X, tr)
    run("E6 LightGBM без UA", "full v2 − UA", models.lgbm, X.drop(columns=UA_COLS), tr)
    r_lgb5 = run("E7 LightGBM ×5 seeds", "full v2", multi_seed(models.lgbm), X, tr)
    r_cat = run("E8 CatBoost", "full v2", models.catboost, X, tr)
    blend("E9 blend LGB×5 + Cat (rank mean)", "full v2", [r_lgb5, r_cat], tr)


# Подбираю параметры на другом разбиении (seed 7), а оцениваю на обычных фолдах (seed 42).
# Иначе Optuna подогнала бы параметры под те же фолды, по которым я потом сужу о качестве.
TUNE_FOLD_SEED = 7
TUNED_PATH = config.ROOT / "reports" / "lgb_tuned_params.json"


# Optuna по LightGBM, 40 попыток, цель OOF P@R70. Тест тут не участвует никак.
def stage_tune(n_trials: int = 40):
    import json
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from src.metrics import precision_at_recall

    tr, _, X, _ = load_features()
    y = tr.target.values
    folds = list(StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=TUNE_FOLD_SEED).split(X, y))

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        oof = np.zeros(len(y))
        for a, b in folds:
            oof[b] = models.lgbm(X.iloc[a], y[a], X.iloc[b], **params)
        return precision_at_recall(y, oof)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials)
    print("best (tune split):", round(study.best_value, 4), study.best_params)
    TUNED_PATH.write_text(json.dumps(study.best_params, indent=2))

    tuned = lambda a, b, c, seed=config.SEED: models.lgbm(a, b, c, seed=seed, **study.best_params)
    r = run("E10 LightGBM tuned ×5 seeds", "full v2", multi_seed(tuned), X, tr)
    r_cat = run("E8 CatBoost", "full v2", models.catboost, X, tr)
    blend("E11 blend LGB tuned×5 + Cat", "full v2", [r, r_cat], tr)


# Важность признаков тремя способами, результат в reports/:
# gain из LightGBM, среднее |SHAP| и permutation importance на валидационной части фолдов.
def stage_importance():
    import lightgbm as lgb
    import shap
    from sklearn.inspection import permutation_importance
    from src.metrics import precision_at_recall

    tr, _, X, _ = load_features()
    y = tr.target.values
    params = models.LGB_PARAMS          # подобранные параметры прироста не дали (E10), беру базовые
    model = lgb.LGBMClassifier(**params, random_state=config.SEED).fit(X, y)
    gain = pd.Series(model.booster_.feature_importance("gain"), X.columns)
    sv = shap.TreeExplainer(model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    shap_imp = pd.Series(np.abs(sv).mean(axis=0), X.columns)

    # для permutation беру PR-AUC: он гладкий, а P@R70 на одном фолде слишком шумный, чтобы мерить им падение
    perm = []
    for a, b in validate.stratified_folds(y):
        m = lgb.LGBMClassifier(**params, random_state=config.SEED).fit(X.iloc[a], y[a])
        pi = permutation_importance(m, X.iloc[b], y[b], scoring="average_precision", n_repeats=3,
                                    random_state=config.SEED, n_jobs=1)
        perm.append(pi.importances_mean)
    imp = pd.DataFrame({"gain_share": gain / gain.sum(), "mean_abs_shap": shap_imp,
                        "perm_pr_auc_drop": np.mean(perm, axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(config.ROOT / "reports" / "feature_importance.csv")
    print(imp.head(30).round(4).to_string())

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(sv, X, max_display=25, show=False)
    plt.tight_layout(); plt.savefig(config.ROOT / "reports" / "shap_summary.png", dpi=110); plt.close()


# Первая серия экспериментов: LightGBM, он же без UA (проверка на шорткаты), CatBoost и их бленд.
def stage_boosting():
    tr, _, X, _ = load_features()
    r_lgb = run("E1 LightGBM", "full", models.lgbm, X, tr)
    run("E2 LightGBM без UA", "full − UA", models.lgbm, X.drop(columns=UA_COLS), tr)
    r_cat = run("E3 CatBoost", "full", models.catboost, X, tr)
    blend("E4 blend LGB+Cat (rank mean)", "full", [r_lgb, r_cat], tr)


if __name__ == "__main__":
    {"baselines": stage_baselines, "boosting": stage_boosting, "features_v2": stage_features_v2, "tune": stage_tune, "importance": stage_importance}[sys.argv[1]]()
