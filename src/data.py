# Загрузка данных и подготовка событий: время, окно наблюдения, дубли, платформа.
import pandas as pd

from src import config

# В окнах дата лежит без времени, в событиях со временем, поэтому формат не фиксирую,
# а после парсинга проверяю, что ничего не превратилось в NaT.
# Все строки без таймзоны считаю UTC, так train, test и events парсятся одинаково.


def _to_utc(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, format="mixed")
    bad = s.notna() & out.isna()
    if bad.any():
        raise ValueError(f"{s.name}: не распарсилось {bad.sum()} значений")
    return out


def load_meta(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in config.META_TIME_COLS:
        df[col] = _to_utc(df[col])
    return df


def load_train() -> pd.DataFrame:
    return load_meta(config.TRAIN_PATH)


def load_test() -> pd.DataFrame:
    return load_meta(config.TEST_PATH)


def load_events() -> pd.DataFrame:
    # item_id читаю строкой: это идентификатор, считать по нему среднее бессмысленно
    ev = pd.read_csv(config.EVENTS_PATH, dtype={"item_id": "string"})
    ev["event_ts"] = _to_utc(ev["event_ts"])
    return ev


# В сырых данных 11 вариантов написания platform (web, WEB, Web, desktop и т.д.).
# Внутри одной куки они чередуются случайно и с таргетом не связаны, это просто шум логирования.
# Поэтому свожу всё к трём семействам и дальше работаю только с ними.
PLATFORM_FAMILY = {
    "web": "web", "WEB": "web", "Web": "web", "desktop": "web",
    "android": "android", "ANDROID": "android", "Android": "android",
    "ios": "ios", "IOS": "ios", "iOS": "ios", "iphone": "ios",
}

EVENT_COLS = [
    "cookie_id", "event_ts", "eid", "event_name", "platform", "user_agent",
    "item_id", "item_category", "item_location", "seller_type",
    "search_query", "search_page", "pointer_x", "pointer_y",
]


# Оставляю только события внутри окна куки: [window_start_ts, window_end_ts).
# Правую границу исключаю: окно идёт ровно с 00:00 до 00:00, и событие в полночь это уже следующие сутки
# (таких 68). Это самое важное место во всей обработке: события после окна есть только у train-кук,
# и среди них лежат все показы капчи, 88% которых у ботов. Если их не отрезать, модель выучит
# реакцию антибот-системы из будущего, а в тесте этого сигнала нет.
def filter_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["cookie_id", "window_start_ts", "window_end_ts"]
    ev = events.merge(meta[cols], on="cookie_id", how="inner")
    inside = (ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)
    return ev.loc[inside].drop(columns=cols[1:]).reset_index(drop=True)


# Полные дубли строк удаляю. Их около 1.5% и доля одинаковая у ботов и людей, так что это
# артефакт логирования, а не поведение. Но число дублей на куку считаю до удаления и отдаю как признак,
# вдруг в нём что-то есть (на деле сигнал слабый).
def drop_duplicate_events(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    dup = events.duplicated(EVENT_COLS, keep="first")
    n_dup = dup.groupby(events.cookie_id).sum().rename("n_dup_events")
    return events.loc[~dup].reset_index(drop=True), n_dup


# Полная подготовка: окно, дубли, семейство платформы, сортировка по времени.
# В исходном файле строки перемешаны, а признакам на интервалы нужен порядок внутри куки.
def prepare_events(events: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    ev = filter_window(events, meta)
    ev, n_dup = drop_duplicate_events(ev)
    ev["platform_family"] = ev.platform.map(PLATFORM_FAMILY)
    # падаю явно, если в новых данных появится незнакомое написание платформы
    if ev.platform_family.isna().any():
        raise ValueError(f"неизвестные platform: {ev.loc[ev.platform_family.isna(), 'platform'].unique()}")
    ev = ev.sort_values(["cookie_id", "event_ts"], kind="mergesort").reset_index(drop=True)
    return ev, n_dup
