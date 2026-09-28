# Весь пайплайн одной командой: данные -> признаки -> CV -> submission.csv.
#
#   python main.py            # полный прогон с CV финальной модели (~10 мин на 12 ядрах)
#   python main.py --no-cv    # только обучение на всём train и submission.csv
#
# Всё считается с нуля из data/. Кэш признаков из экспериментов здесь сознательно не используется,
# чтобы результат гарантированно воспроизводился на чистой машине.
import sys

import pandas as pd

from src import config, data, features, train, validate
from src.check_submission import check_submission


def main(run_cv: bool = True) -> None:
    print("1/5 загрузка данных")
    tr, te = data.load_train(), data.load_test()
    events = data.load_events()
    meta = pd.concat([tr, te], ignore_index=True)

    print("2/5 окно [start, end), удаление дублей, нормализация платформы")
    ev, n_dup = data.prepare_events(events, meta)

    # train и test считаю одним вызовом: так у категориальных признаков общий словарь,
    # а частота user agent считается по всем кукам сразу
    print("3/5 признаки (один код для train и test)")
    F = features.build_features(ev, meta, n_dup)
    X_tr = F.loc[tr.cookie_id].reset_index(drop=True)
    X_te = F.loc[te.cookie_id].reset_index(drop=True)
    print(f"    train {X_tr.shape}, test {X_te.shape}")

    # CV именно той функции, которая потом строит сабмит, а не её приближения из экспериментов
    if run_cv:
        print("4/5 CV финальной модели (StratifiedKFold(5) + time-holdout)")
        validate.cross_validate(train.final_fit_predict, X_tr, tr.target.values, tr.window_start_ts)
    else:
        print("4/5 CV пропущен (--no-cv)")

    print("5/5 обучение на всём train и предсказание теста")
    sub = train.predict_test(X_tr, tr.target.values, X_te, te.cookie_id)
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"    сохранено: {config.SUBMISSION_PATH}")

    # не выпускаю файл, который не прошёл проверку
    errors = check_submission(config.SUBMISSION_PATH)
    if errors:
        raise SystemExit("submission не прошёл проверку: " + "; ".join(errors))
    print("submission OK")


if __name__ == "__main__":
    main(run_cv="--no-cv" not in sys.argv)
