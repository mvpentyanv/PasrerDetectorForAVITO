# Признаки на уровне куки.
# Все функции получают события после data.prepare_events (только окно, без дублей, отсортированы)
# и считают признаки только по событиям самой куки. Единственное исключение: частота user agent
# по всему датасету, это обычный frequency encoding без таргета.
# Train и test прогоняются через один и тот же код, чтобы признаки гарантированно совпадали по смыслу.
import numpy as np
import pandas as pd
from scipy.stats import skew

EVENT_NAMES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login", "captcha_shown",
]


def _align(f: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    return f.reindex(meta.cookie_id).reset_index(drop=True)


# Baseline 0: ровно те два признака, что в quickstart организаторов, для честного сравнения.
def quickstart_features(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    f = pd.DataFrame({"n_events": g.size(), "item_nunique": g.item_id.nunique()})
    return _align(f, meta).fillna(0)


# Baseline 1: объём и состав событий. Нужен, чтобы понять, сколько даёт "что делал" без "как делал".
def basic_features(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_NAMES, fill_value=0)
    n = counts.sum(axis=1)
    f = counts.div(n, axis=0).add_prefix("share_")
    f.insert(0, "log_n_events", np.log1p(n))
    return _align(f, meta).fillna(0)


# ============================================================================ полный набор

SESSION_GAP_S = 30 * 60           # пауза больше 30 минут считается началом новой сессии
AUTOMATION_UA = r"curl|python-requests|python-urllib3|Go-http-client|node-fetch|Scrapy"
SCREEN_W, SCREEN_H = 1920, 1080   # максимальные координаты курсора, которые встречаются в данных
# низкокардинальные категории: их бустинги обрабатывают сами, без ручного кодирования
CATEGORICAL = ["platform_family", "ua_kind", "ua_browser", "ua_os"]


# Энтропия распределения значений col внутри куки (NaN не учитываются).
# Хорошо ловит "разнообразие": человек обычно крутится вокруг пары категорий, парсер размазан по многим.
def _entropy(ev: pd.DataFrame, col: str) -> pd.Series:
    cnt = ev.groupby(["cookie_id", col]).size()
    p = cnt / cnt.groupby(level=0).transform("sum")
    return (-(p * np.log2(p))).groupby(level=0).sum()


# Длина самой длинной серии подряд идущих True внутри куки.
# Трюк: номер серии растёт на каждом False, дальше просто считаю True в каждой серии.
def _max_run(flag: pd.Series, key: pd.Series) -> pd.Series:
    run_id = (~flag).groupby(key).cumsum()
    runs = flag.groupby([key, run_id]).sum()
    return runs.groupby(level=0).max()


# Вспомогательные колонки, которые нужны сразу нескольким группам признаков.
# Здесь важно, что события уже отсортированы по куке и времени, иначе diff и shift будут мусором.
def _add_event_columns(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["t"] = ev.event_ts.astype("int64") / 1e6            # время в секундах (datetime64 в микросекундах)
    ev["dt"] = ev.groupby("cookie_id").t.diff()             # интервал до предыдущего события куки
    ev["hour"] = ev.event_ts.dt.hour
    ev["prev_event"] = ev.groupby("cookie_id").event_name.shift()
    return ev


# Темп и ритм. По EDA бот действует чаще и ровнее человека: медиана интервала 14 с против 47 с,
# основная масса интервалов у ботов в диапазоне 2-20 с.
def _tempo(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    dt = ev.dt
    flags = pd.DataFrame({
        "cookie_id": ev.cookie_id,
        # время в данных с точностью до секунды, поэтому "меньше 0.3 с" на практике значит "0 с"
        "dt_lt1": dt < 1, "dt_zero": dt == 0,
        "dt_2_20": dt.between(2, 20), "dt_gt300": dt > 300,
        "new_session": dt > SESSION_GAP_S,
        "night": ev.hour < 6,
    })
    has_dt = dt.notna()
    fg = flags[has_dt].groupby("cookie_id")
    dur = g.t.max() - g.t.min()
    # беру квантили, а не только среднее: у людей длинный хвост пауз, и среднее от него сильно плывёт
    f = pd.DataFrame({
        "n_events": g.size(),
        "duration_h": dur / 3600,
        "dt_mean": g.dt.mean(), "dt_median": g.dt.median(), "dt_std": g.dt.std(),
        "dt_min": g.dt.min(), "dt_max": g.dt.max(),
        "dt_q10": g.dt.quantile(0.1), "dt_q25": g.dt.quantile(0.25),
        "dt_q75": g.dt.quantile(0.75), "dt_q90": g.dt.quantile(0.9),
        "share_dt_lt1s": fg.dt_lt1.mean(), "share_dt_lt03s": fg.dt_zero.mean(),
        "share_dt_2_20s": fg.dt_2_20.mean(), "share_dt_gt5min": fg.dt_gt300.mean(),
        "n_sessions": 1 + flags.groupby("cookie_id").new_session.sum(),
        "night_share": flags.groupby("cookie_id").night.mean(),
        "hour_first": g.hour.first(), "hour_last": g.hour.last(), "hour_mean": g.hour.mean(),
        "n_active_hours": g.hour.nunique(),
    })
    f["log_n_events"] = np.log1p(f.n_events)
    # нижняя граница в 1 минуту, чтобы куки из одного-двух событий не давали бесконечный темп
    f["events_per_hour"] = f.n_events / np.maximum(f.duration_h, 1 / 60)
    f["dt_cv"] = f.dt_std / f.dt_mean
    f["events_per_session"] = f.n_events / f.n_sessions
    # активное время: сумма интервалов внутри сессий, без долгих перерывов между заходами
    active = ev.dt.where(ev.dt <= SESSION_GAP_S).groupby(ev.cookie_id).sum()
    f["active_min"] = active / 60
    f["events_per_active_min"] = f.n_events / np.maximum(f.active_min, 1)
    # энтропия интервалов по логарифмическим корзинам: у скрипта с фиксированной задержкой она низкая
    tmp = ev.loc[has_dt, ["cookie_id"]].assign(dt_bin=np.floor(np.log2(dt[has_dt] + 1)))
    f["dt_entropy"] = _entropy(tmp, "dt_bin")
    f["hour_entropy"] = _entropy(ev, "hour")
    return f


# Что именно делала кука: счётчики и доли типов событий, их отношения и переходы между ними.
def _event_types(ev: pd.DataFrame, top_bigrams: list[str]) -> pd.DataFrame:
    # капчу выкидываю: внутри окна она не встречается ни разу, все показы лежат после окна (см. data.py)
    names = [e for e in EVENT_NAMES if e != "captcha_shown"]
    cnt = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=names, fill_value=0)
    n = cnt.sum(axis=1)
    f = pd.concat([cnt.add_prefix("cnt_"), cnt.div(n, axis=0).add_prefix("share_")], axis=1)
    iv, sr = cnt.item_view, cnt.search_results_view
    contacts = cnt.contact_phone_show + cnt.contact_chat_open + cnt.contact_message_sent
    f["n_contacts"] = contacts
    # отношения на один просмотр объявления: человек листает фото и добавляет в избранное,
    # парсер чаще просто открывает карточки и страницы продавцов. Деление на 0 даёт NaN, а не inf.
    f["ratio_contacts_per_item"] = contacts / iv.replace(0, np.nan)
    f["ratio_item_per_search"] = iv / sr.replace(0, np.nan)
    f["ratio_photo_per_item"] = cnt.photo_swipe / iv.replace(0, np.nan)
    f["ratio_fav_per_item"] = cnt.favorite_add / iv.replace(0, np.nan)
    f["ratio_seller_per_item"] = cnt.seller_page_view / iv.replace(0, np.nan)
    f["has_login"] = (cnt.login > 0).astype(int)
    f["event_type_entropy"] = _entropy(ev, "event_name")
    f["n_event_types"] = (cnt > 0).sum(axis=1)
    # биграммы переходов A>B как доля от всех переходов куки: описывают "сценарий" навигации
    pairs = ev.prev_event + ">" + ev.event_name
    tr = pd.crosstab(ev.cookie_id, pairs).reindex(columns=top_bigrams, fill_value=0)
    f = f.join(tr.div(np.maximum(n - 1, 1), axis=0).add_prefix("bigram_"))
    f["share_same_event_repeat"] = (ev.prev_event == ev.event_name).groupby(ev.cookie_id).sum() / np.maximum(n - 1, 1)
    return f


# Объявления и поиск. Парсер обходит каталог "вширь": много городов и объявлений, глубокое листание
# выдачи, почти без возвратов к уже открытому. Человек обычно выбирает, поэтому возвращается и сравнивает.
def _items_search(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    n = g.size()
    it = ev[ev.item_id.notna()]
    gi = it.groupby("cookie_id")
    f = pd.DataFrame({
        "n_items": g.item_id.nunique(), "n_categories": g.item_category.nunique(),
        "n_locations": g.item_location.nunique(), "n_seller_types": g.seller_type.nunique(),
        "item_repeat_share": 1 - gi.item_id.nunique() / gi.size(),
        "share_pro_seller": gi.seller_type.apply(lambda s: (s == "pro").sum()) / gi.seller_type.count(),
        # высокую кардинальность категорий и городов сворачиваю в энтропию вместо кодирования самих значений
        "category_entropy": _entropy(ev, "item_category"),
        "location_entropy": _entropy(ev, "item_location"),
    })
    # нормирую на число событий, иначе признак просто дублирует объём активности
    for c in ["n_items", "n_categories", "n_locations"]:
        f[c + "_per_event"] = f[c] / n

    s = ev[ev.event_name == "search_results_view"].copy()
    key = s.cookie_id
    s["dp"] = s.groupby(key).search_page.diff()
    s["same_query"] = s.search_query.eq(s.groupby(key).search_query.shift())
    s["step1"] = s.dp.eq(1)
    # страница +1 по тому же запросу: типичный последовательный обход выдачи скриптом
    s["paging"] = s.step1 & s.same_query
    gs = s.groupby(key)
    has_prev = s.dp.notna()
    sp = pd.DataFrame({
        "page_max": gs.search_page.max(), "page_mean": gs.search_page.mean(), "page_std": gs.search_page.std(),
        "share_page_gt1": gs.search_page.apply(lambda x: (x > 1).mean()),
        "share_page_step1": s[has_prev].groupby("cookie_id").step1.mean(),
        "share_paging_same_query": s[has_prev].groupby("cookie_id").paging.mean(),
        "max_page_run": _max_run(s.step1, key),
        "n_queries": gs.search_query.nunique(),
        "query_len_mean": gs.search_query.apply(lambda x: x.str.len().mean()),
        "share_same_query": s[has_prev].groupby("cookie_id").same_query.mean(),
    })
    sp["queries_per_search"] = sp.n_queries / gs.size()
    sp["page_entropy"] = _entropy(s, "search_page")
    return f.join(sp)


# Курсор. Самая сильная группа: у ботов "виртуальная" мышь с узким разбросом по X
# (std 203 против 553 у людей), смещённая влево и часто упирающаяся в края экрана.
# Курсор есть только на web. Для android и ios оставляю NaN, а не 0: это "не применимо",
# и бустинг отправляет такие куки в отдельную ветку, не путая их с нулевыми координатами.
def _pointer(ev: pd.DataFrame) -> pd.DataFrame:
    web = ev[ev.platform_family == "web"]
    has = web.pointer_x.notna() & web.pointer_y.notna()
    p = web[has].copy()
    key = p.cookie_id
    p["jump"] = np.hypot(p.groupby(key).pointer_x.diff(), p.groupby(key).pointer_y.diff())
    p["zero"] = (p.pointer_x == 0) & (p.pointer_y == 0)
    p["edge"] = p.pointer_x.isin([0, SCREEN_W]) | p.pointer_y.isin([0, SCREEN_H])
    p["xy"] = p.pointer_x.astype(int).astype(str) + "_" + p.pointer_y.astype(int).astype(str)
    gp = p.groupby(key)
    f = pd.DataFrame({
        "ptr_share": has.groupby(web.cookie_id).mean(),
        "ptr_n": gp.size(),
        "ptr_x_mean": gp.pointer_x.mean(), "ptr_y_mean": gp.pointer_y.mean(),
        "ptr_x_std": gp.pointer_x.std(), "ptr_y_std": gp.pointer_y.std(),
        "ptr_x_min": gp.pointer_x.min(), "ptr_x_max": gp.pointer_x.max(),
        "ptr_y_min": gp.pointer_y.min(), "ptr_y_max": gp.pointer_y.max(),
        "ptr_unique_share": gp.xy.nunique() / gp.size(),
        "ptr_zero_share": gp.zero.mean(), "ptr_edge_share": gp.edge.mean(),
        "ptr_jump_mean": gp.jump.mean(), "ptr_jump_std": gp.jump.std(),
    })
    f["ptr_x_range"] = f.ptr_x_max - f.ptr_x_min
    f["ptr_y_range"] = f.ptr_y_max - f.ptr_y_min
    return f


# Строку user agent как есть не использую: это 148 уникальных значений, и модель легко выучила бы
# конкретные строки. Вместо этого разбираю её на несколько низкокардинальных полей.
def _parse_ua(ua: pd.Series) -> pd.DataFrame:
    kind = np.select(
        [ua.str.contains(AUTOMATION_UA), ua.str.contains("Headless"), ua.str.startswith("Avito/")],
        ["script", "headless", "app"], default="browser")
    browser = ua.str.extract(r"(Edg|OPR|YaBrowser|Firefox|Chrome|HeadlessChrome|Version/[\d.]+ Mobile/\S+ Safari|Safari)")[0]
    browser = browser.str.replace(r"^Version.*", "Safari", regex=True).fillna("other")
    version = ua.str.extract(r"(?:Chrome|Firefox|Edg|OPR|YaBrowser|Version|Avito)/(\d+)")[0].astype(float)
    os_ = np.select(
        [ua.str.contains("Android"), ua.str.contains("iPhone|iPad|iOS"), ua.str.contains("Windows"),
         ua.str.contains("Mac OS X"), ua.str.contains("Linux|X11")],
        ["android", "ios", "windows", "mac", "linux"], default="none")
    return pd.DataFrame({"ua_kind": kind, "ua_browser": browser, "ua_version": version, "ua_os": os_,
                         "ua_len": ua.str.len()}, index=ua.index)


# Устройство и сама кука.
def _device(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    # у 7% кук UA меняется внутри окна, поэтому беру самый частый
    main_ua = g.user_agent.agg(lambda s: s.mode().iat[0])
    f = pd.DataFrame({
        "platform_family": g.platform_family.first(),       # проверял в EDA: у куки семейство всегда одно
        "n_user_agents": g.user_agent.nunique(),
    })
    f["ua_changed"] = (f.n_user_agents > 1).astype(int)
    f = f.join(_parse_ua(main_ua))
    # frequency encoding по train+test. Таргет не участвует, так что утечки нет,
    # а редких UA в данных нет (минимум 40 кук на значение), переобучиться не на чем.
    f["ua_freq"] = main_ua.map(main_ua.value_counts(normalize=True))
    # возраст куки на начало окна: у ботов куки заметно моложе (медиана 19 дней против 62)
    m = meta.set_index("cookie_id")
    age = (m.window_start_ts - m.cookie_created_at).dt.total_seconds() / 86400
    f["cookie_age_d"] = age
    f["log_cookie_age_d"] = np.log1p(age)
    f["cookie_age_lt1d"] = (age < 1).astype(int)
    f["cookie_created_hour"] = m.cookie_created_at.dt.hour
    return f


# Пропуски не заполняю: они структурные и зависят от типа события.
# Считаю только те, где поле по смыслу должно быть, то есть у событий с объявлением.
def _missing(ev: pd.DataFrame) -> pd.DataFrame:
    item_ev = ev[ev.item_id.notna()]
    gi = item_ev.groupby("cookie_id")
    return pd.DataFrame({
        "na_share_category": gi.item_category.apply(lambda s: s.isna().mean()),
        "na_share_location": gi.item_location.apply(lambda s: s.isna().mean()),
        "na_share_seller_type": gi.seller_type.apply(lambda s: s.isna().mean()),
    })


# Интервалы только внутри сессий и форма их распределения.
# Паузы между заходами у ботов и людей похожи, а вот внутри захода бот быстрее и равномернее.
# На мобильных, где нет курсора, это главный сигнал. Группа добавлена во второй итерации, дала около +0.01.
def _session_timing(ev: pd.DataFrame) -> pd.DataFrame:
    key = ev.cookie_id
    ins = ev.dt.where(ev.dt <= SESSION_GAP_S)
    ldt = np.log1p(ins)
    f = pd.DataFrame({
        "in_dt_median": ins.groupby(key).median(), "in_dt_mean": ins.groupby(key).mean(),
        "in_dt_std": ins.groupby(key).std(),
        "log_dt_mean": ldt.groupby(key).mean(), "log_dt_std": ldt.groupby(key).std(),
        # асимметрию считаю только от трёх интервалов, на меньшем числе она ничего не значит
        "log_dt_skew": ldt.groupby(key).apply(lambda s: skew(s.dropna()) if s.notna().sum() > 2 else np.nan),
    })
    f["in_dt_cv"] = f.in_dt_std / f.in_dt_mean
    return f


# Как часто соседние события относятся к разным городам и категориям.
# Энтропия говорит, сколько всего было разного, а этот признак ещё и про то, как часто кука "прыгает".
def _switches(ev: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col, name in [("item_location", "loc_change_share"), ("item_category", "cat_change_share")]:
        e = ev[ev[col].notna()]
        changed = e[col].ne(e.groupby("cookie_id")[col].shift())
        out[name] = changed.groupby(e.cookie_id).mean()
    return pd.DataFrame(out)


# Выгрузка контактов продавцов, один из сценариев парсинга:
# доля показов телефона у pro-продавцов и переход "страница продавца, потом телефон".
def _contacts(ev: pd.DataFrame) -> pd.DataFrame:
    phone = ev[ev.event_name == "contact_phone_show"]
    seller_then_phone = (ev.prev_event == "seller_page_view") & (ev.event_name == "contact_phone_show")
    return pd.DataFrame({
        "phone_pro_share": phone.seller_type.eq("pro").groupby(phone.cookie_id).mean(),
        "seller_then_phone": seller_then_phone.groupby(ev.cookie_id).sum(),
    })


# Список самых частых переходов между типами событий. Он нужен только чтобы выбрать колонки
# для биграмм, таргет тут не участвует. Из 81 возможного перехода топ-25 покрывают 87% всех,
# остальные слишком редкие, чтобы давать устойчивый сигнал.
def top_bigrams(ev: pd.DataFrame, n: int = 25) -> list[str]:
    ev = _add_event_columns(ev) if "prev_event" not in ev else ev
    pairs = (ev.prev_event + ">" + ev.event_name).dropna()
    return pairs.value_counts().head(n).index.tolist()


# Полный набор признаков (151 штука) для всех кук из meta, индекс по cookie_id.
# Сам cookie_id в признаки не попадает ни в каком виде.
def build_features(ev: pd.DataFrame, meta: pd.DataFrame, n_dup: pd.Series) -> pd.DataFrame:
    ev = _add_event_columns(ev[ev.cookie_id.isin(meta.cookie_id)])
    parts = [_tempo(ev), _session_timing(ev), _event_types(ev, top_bigrams(ev)), _items_search(ev),
             _switches(ev), _contacts(ev), _pointer(ev), _device(ev, meta), _missing(ev)]
    f = pd.concat(parts, axis=1).reindex(meta.cookie_id)
    f["n_dup_events"] = n_dup.reindex(f.index).fillna(0)
    f["dup_share"] = f.n_dup_events / (f.n_events + f.n_dup_events)
    # общий набор категорий на train и test сразу, чтобы коды категорий совпадали между выборками
    for c in CATEGORICAL:
        f[c] = pd.Categorical(f[c])
    return f
