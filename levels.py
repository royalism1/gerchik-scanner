"""Уровни по Герчику для одной акции.

Вход: дневные бары (DataFrame с колонками Open, High, Low, Close, Volume, индекс — даты),
последний бар — день отбора D. Уровни строятся ТОЛЬКО по барам до D (бар D их не создаёт),
по дневке за ~6 месяцев и недельке (6 мес. основное окно, год — для исторических/зеркальных).
"""
import math
import numpy as np
import pandas as pd


# ---------- вспомогательные ----------

def true_range(df):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr


def atr14(df):
    return float(true_range(df).iloc[-14:].mean())


def atr_gerchik(df, a14, L=None):
    """ATR по Герчику: среднее за 5 последних дней без паранормальных (слишком больших и слишком маленьких) баров.
    L — раздел levels конфига. Без новых ключей — прежний расчёт (TR, пороги 1,5/0,5, бар D включён).
    paranormal_tr_atr / small_bar_atr — пороги «слишком большой» / «слишком маленький» бар (материал владельца: 2 и 1/3);
    atr_mode="hl" — по материалу: среднее High−Low последних 5 дней ДО D (бар D не берётся);
    если нормальных баров < 3 — окно расширяется до 10 дней."""
    L = L or {}
    big = L.get("paranormal_tr_atr", 1.5)
    small = L.get("small_bar_atr", 0.5)
    if L.get("atr_mode", "tr") == "hl":
        hl = (df["High"] - df["Low"]).iloc[:-1]
        for w in (5, 10):
            x = hl.iloc[-w:]
            ok = x[(x < big * a14) & (x > small * a14)]
            if len(ok) >= 3:
                return float(ok.iloc[-5:].mean())
        return a14
    tr = true_range(df).iloc[-5:]
    if "small_bar_atr" in L:
        ok = tr[(tr < big * a14) & (tr > small * a14)]
    else:
        ok = tr[(tr <= 1.5 * a14) & (tr >= 0.5 * a14)]
    return float(ok.mean()) if len(ok) >= 3 else a14


def weekly(df):
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna()


def pivots(high, low, k, last_idx):
    """Фракталы: экстремум среди k баров слева и справа. Для свежих баров справа хватает 1 бара.
    last_idx — индекс бара D (он сам не может быть источником уровня)."""
    out = []
    n = last_idx  # кандидаты i < n
    for i in range(k, n):
        r = min(i + k, n - 1)  # правая граница без бара D
        if r <= i:
            continue
        lh = high[i - k:i]
        rh = high[i + 1:r + 1]
        if high[i] >= lh.max() and high[i] > rh.max():
            out.append(("h", i, float(high[i])))
        ll = low[i - k:i]
        rl = low[i + 1:r + 1]
        if low[i] <= ll.min() and low[i] < rl.min():
            out.append(("l", i, float(low[i])))
    return out


def round_step(price):
    if price < 10:
        return 0.5
    if price < 50:
        return 1.0
    if price < 200:
        return 5.0
    if price < 1000:
        return 10.0
    return 50.0


def is_round(price, atr, tol_atr):
    st = round_step(price)
    near = round(price / st) * st
    return abs(price - near) <= max(tol_atr * atr, 0.01), near


def level_strength(base, mirror, lp, rnd, gap, weekly_conf, sawing, L):
    """Итоговая сила уровня 1–7 (B, инженер 24.10): бонусы круглая/геп/неделька вместе ≤ strength_bonus_max
    (уровень >4 этим правилом не опускается ниже 4); без зеркальности и ЛП — не выше strength_cap_no_mirror_lp;
    программа ставит не выше strength_max_program. Без ключей в конфиге — прежний расчёт."""
    bonus = int(rnd) + int(gap) + int(weekly_conf)
    s = base + bonus - (2 if sawing else 0)
    if s > 4:
        s = max(4, min(s, base + L.get("strength_bonus_max", 3) - (2 if sawing else 0)))
    if not (mirror or lp):
        s = min(s, L.get("strength_cap_no_mirror_lp", 7))
    s = min(s, L.get("strength_max_program", 7))
    return max(1, min(7, s))


def strict_limit_ok(H, Lo, O, C, lvl, D, atr):
    """Строгий лимитный уровень (EN-4): ≥3 удара в одну цену (хай или верх тела для сопротивления, лой или низ тела
    для поддержки) в пределах max(0,01$; 0,02·ATR), и после первого удара цена ни разу не прошла уровень даже на цент."""
    tol = max(0.01, 0.02 * atr)
    for side in ("h", "l"):
        hits = [i for i in range(D) if
                (side == "h" and (abs(H[i] - lvl) <= tol or abs(max(O[i], C[i]) - lvl) <= tol)) or
                (side == "l" and (abs(Lo[i] - lvl) <= tol or abs(min(O[i], C[i]) - lvl) <= tol))]
        if len(hits) < 3:
            continue
        f = hits[0]
        if side == "h" and all(H[i] <= lvl + max(0.01, tol) for i in range(f, D)):
            return True
        if side == "l" and all(Lo[i] >= lvl - max(0.01, tol) for i in range(f, D)):
            return True
    return False


# ---------- поиск уровней ----------

def find_levels(df, cfg):
    """Возвращает список уровней с типами, силой и признаками. df — дневки до D включительно."""
    L = cfg["levels"]
    n_all = len(df)
    win = min(L["window_daily_bars"] + 1, n_all)
    d = df.iloc[-win:]
    H, Lo, C, O = d["High"].values, d["Low"].values, d["Close"].values, d["Open"].values
    n = len(d)
    D = n - 1
    a14 = atr14(df)
    tr_all = true_range(df)
    atr_roll = tr_all.rolling(14).mean().iloc[-win:].values
    tol = L["cluster_tol_atr"] * a14

    cands = []  # (price, source, idx)
    # пивоты дневки
    dpiv = pivots(H, Lo, L["pivot_k_daily"], D)
    ph = [(i, p) for kind, i, p in dpiv if kind == "h"]
    pl = [(i, p) for kind, i, p in dpiv if kind == "l"]
    for kind, i, p in dpiv:
        cands.append((p, "pivot_" + kind, i))
    # лимитный игрок: 2+ бара подряд с одинаковыми хаями/лоями
    lp_tol = L["limit_player_tol_atr"] * a14
    kk = L["pivot_k_daily"]
    for i in range(1, D):
        seg = slice(max(0, i - 1 - kk), min(D, i + 1 + kk))
        if abs(H[i] - H[i - 1]) <= lp_tol and max(H[i], H[i - 1]) >= H[seg].max():
            cands.append((max(H[i], H[i - 1]), "limit_h", i))
        if abs(Lo[i] - Lo[i - 1]) <= lp_tol and min(Lo[i], Lo[i - 1]) <= Lo[seg].min():
            cands.append((min(Lo[i], Lo[i - 1]), "limit_l", i))
    # паранормальные бары и гэпы
    for i in range(1, D):
        ar = atr_roll[i] if not math.isnan(atr_roll[i]) else a14
        if H[i] - Lo[i] >= L["paranormal_tr_atr"] * ar:
            cands.append((H[i], "para_h", i))
            cands.append((Lo[i], "para_l", i))
        if Lo[i] - H[i - 1] >= L["gap_min_atr"] * ar:
            cands.append((H[i - 1], "gap", i)); cands.append((Lo[i], "gap", i))
        if Lo[i - 1] - H[i] >= L["gap_min_atr"] * ar:
            cands.append((Lo[i - 1], "gap", i)); cands.append((H[i], "gap", i))
    # недельные пивоты (год)
    w = weekly(df.iloc[:-1])  # без бара D
    w = w.iloc[-L["history_weekly_bars"]:]
    wpiv = pivots(w["High"].values, w["Low"].values, L["pivot_k_weekly"], len(w))
    for kind, i, p in wpiv:
        cands.append((p, "weekly_" + kind, -1))

    if not cands:
        return [], a14
    cands.sort(key=lambda x: x[0])
    clusters, cur = [], [cands[0]]
    for c in cands[1:]:
        if c[0] - cur[-1][0] <= tol:
            cur.append(c)
        else:
            clusters.append(cur); cur = [c]
    clusters.append(cur)

    # крупные развороты (излом тренда): экстремум в ±major_k барах и потом движение ≥ X ATR
    mk = L["major_pivot_k"]
    major = []
    for i in range(1, D):
        lo_, hi_ = max(0, i - mk), min(D - 1, i + mk)
        if H[i] >= H[lo_:hi_ + 1].max():
            after = Lo[i + 1:D]
            if len(after) and H[i] - after.min() >= L["major_pivot_move_atr"] * a14:
                major.append(("h", i, H[i]))
        if Lo[i] <= Lo[lo_:hi_ + 1].min():
            after = H[i + 1:D]
            if len(after) and after.max() - Lo[i] >= L["major_pivot_move_atr"] * a14:
                major.append(("l", i, Lo[i]))

    wk_prices = [p for _, _, p in wpiv]
    hi_year = float(df["High"].iloc[-253:-1].max()) if n_all > 2 else float(H.max())
    lo_year = float(df["Low"].iloc[-253:-1].min()) if n_all > 2 else float(Lo.min())
    ttol = L["touch_tol_atr"] * a14

    levels = []
    for cl in clusters:
        srcs = {s for _, s, _ in cl}
        daily_srcs = {s for s in srcs if not s.startswith("weekly")}
        prices = [p for p, s, _ in cl if not s.startswith("weekly")] or [p for p, _, _ in cl]
        # цена уровня: самая частая/крайняя из пивотов, иначе медиана
        piv_prices = [p for p, s, _ in cl if s.startswith("pivot") or s.startswith("limit")]
        lvl = float(np.median(piv_prices if piv_prices else prices))

        # реакции считаются только по ЛОКАЛЬНЫМ ЭКСТРЕМУМАМ (пивотам) у уровня, а не по любым барам
        res_t = [i for i, p in ph if abs(p - lvl) <= ttol]           # хай-разворот у уровня (сопротивление)
        sup_t = [i for i, p in pl if abs(p - lvl) <= ttol]           # лой-разворот у уровня (поддержка)
        lp_up = [i for i, p in ph if p > lvl + ttol and p - lvl <= 0.6 * a14 and C[i] < lvl]  # прокол вверх, закрытие под
        lp_dn = [i for i, p in pl if p < lvl - ttol and lvl - p <= 0.6 * a14 and C[i] > lvl]  # прокол вниз, закрытие над
        # ЛП засчитывается, только если уровень был ДО прокола
        first_ref = min(res_t + sup_t) if (res_t or sup_t) else None
        lp_up = [i for i in lp_up if first_ref is not None and i > first_ref]
        lp_dn = [i for i in lp_dn if first_ref is not None and i > first_ref]
        crosses = sum(1 for i in range(1, D) if (C[i - 1] - lvl) * (C[i] - lvl) < 0)
        touches = [i for i in range(D) if H[i] >= lvl - ttol and Lo[i] <= lvl + ttol]
        reactions = sorted(set(res_t + sup_t + lp_up + lp_dn))
        mirror = bool((res_t or lp_up) and (sup_t or lp_dn))
        lp = bool(lp_up or lp_dn)
        wk_hits = sum(1 for p in wk_prices if abs(p - lvl) <= L["weekly_confirm_tol_atr"] * a14)
        historical = (len(reactions) >= 2 and reactions[-1] - reactions[0] >= L["historical_min_gap_bars"]) or wk_hits >= 2
        trend_break = any(abs(p - lvl) <= tol for _, _, p in major)
        limit = any(s.startswith("limit") for s in srcs)
        if limit and L.get("limit_strict"):
            limit = strict_limit_ok(H, Lo, O, C, lvl, D, a14)
        para = any(s.startswith("para") for s in srcs)
        gap = "gap" in srcs
        run, best, tset = 0, 0, set(touches)
        for i in range(D):
            run = run + 1 if i in tset else 0
            best = max(best, run)
        congestion = best >= 3
        weekly_conf = wk_hits >= 1
        only_weekly = not daily_srcs
        rnd, _ = is_round(lvl, a14, L["round_tol_atr"])
        sawing = crosses >= L["sawing_crosses"]
        if crosses >= 2 * L["sawing_crosses"]:
            continue  # это не уровень, а зона проторговки — цена ходит сквозь
        ext_year = "максимум года" if lvl >= hi_year - tol else ("минимум года" if lvl <= lo_year + tol else "")

        types = []
        if trend_break: types.append("излом тренда")
        if mirror: types.append("зеркальный")
        if historical: types.append("исторический")
        if lp: types.append("ЛП")
        if limit: types.append("лимитный игрок")
        if para: types.append("паранормальный бар")
        if gap: types.append("граница гепа")
        if congestion and not types: types.append("проторговка")
        if not types:
            types.append("воздушный")

        if mirror and lp and historical:
            base = 7
        elif mirror and lp:
            base = 6
        elif trend_break or mirror:
            base = 5
        elif lp or limit or (para and len(reactions) >= 1):
            base = 4
        elif len(reactions) >= 2 or historical:
            base = 3
        else:
            base = 2
        s = level_strength(base, mirror, lp, rnd, gap, weekly_conf and not only_weekly, sawing, L)

        marks = []
        if rnd: marks.append("круглая цена")
        if weekly_conf: marks.append("видна на недельке")
        if sawing: marks.append("пилится")
        if congestion: marks.append("проторговка")
        if ext_year: marks.append(ext_year)
        last_touch = max(touches) if touches else None
        levels.append({
            "price": round(lvl, 2),
            "type": " + ".join(types),
            "strength": s,
            "base_strength": base,
            "tf": "неделя" if only_weekly else ("неделя+день" if weekly_conf else "день"),
            "marks": marks,
            "reactions": len(reactions),
            "lp_count": len(lp_up) + len(lp_dn),
            "last_touch_bars_ago": (D - last_touch) if last_touch is not None else None,
            "crosses": crosses,
        })
    # убрать дубли по цене (оставить сильнейший)
    levels.sort(key=lambda x: (x["price"], -x["strength"]))
    uniq = []
    for lv in levels:
        if uniq and abs(lv["price"] - uniq[-1]["price"]) <= tol:
            if lv["strength"] > uniq[-1]["strength"]:
                uniq[-1] = lv
        else:
            uniq.append(lv)
    return uniq, a14


# ---------- бар у уровня ----------

def near_levels(df, levels, a14, cfg):
    """Уровни силой ≥ min_strength, у которых стоит бар D. Возвращает список с признаками."""
    N, Z = cfg["near_level"], cfg["zones"]
    atr = atr_gerchik(df, a14, cfg["levels"])
    last = df.iloc[-1]
    Hd, Ld, Cd, Od = float(last["High"]), float(last["Low"]), float(last["Close"]), float(last["Open"])
    win = min(cfg["levels"]["window_daily_bars"] + 1, len(df))
    d = df.iloc[-win:]
    H, Lo, C = d["High"].values, d["Low"].values, d["Close"].values
    D = len(d) - 1
    ttol = N["touch_tol_atr"] * atr
    out = []
    strong = [lv for lv in levels if lv["strength"] >= N["min_strength"] and lv.get("base_strength", 0) >= 3]
    for lv in strong:
        p = lv["price"]
        dist = abs(Cd - p) / atr
        touched = Ld <= p + ttol and Hd >= p - ttol
        if dist > N["band_max_close_atr"]:
            continue
        in_thr = dist <= N["in_close_atr"] or (touched and dist <= N["in_touch_close_atr"])
        pos = "сверху" if p >= Cd else "снизу"  # где уровень относительно закрытия
        # сколько дней назад цена была ЗА уровнем (по ходу пробоя)
        days_beyond = None
        for back in range(1, D + 1):
            i = D - back
            if (pos == "сверху" and H[i] > p + ttol) or (pos == "снизу" and Lo[i] < p - ttol):
                days_beyond = back
                break
        # предыдущее касание (ретест)
        retest = None
        for back in range(1, D + 1):
            i = D - back
            if H[i] >= p - ttol and Lo[i] <= p + ttol:
                retest = back
                break
        marks = list(lv["marks"])
        if days_beyond is None:
            marks.append("свободная зона > 6 мес.")
        elif days_beyond >= Z["free_zone_days"]:
            marks.append(f"свободная зона {days_beyond} дн.")
        else:
            marks.append(f"зараженная зона {days_beyond} дн.")
        if retest is not None:
            if retest <= Z["near_retest_days"]:
                marks.append(f"ближний ретест {retest} дн.")
            elif retest >= Z["far_retest_days"]:
                marks.append(f"дальний ретест {retest} дн.")
        out.append({**lv, "marks": marks, "position": pos, "dist_close_atr": round(dist, 3),
                    "dist_open_atr": round(abs(Od - p) / atr, 3), "touched": bool(touched),
                    "in_threshold": bool(in_thr), "days_beyond": days_beyond, "retest_days": retest})
    out.sort(key=lambda x: (-x["strength"], x["dist_close_atr"]))
    return out, atr


def box_features(df, atr, p, max_h_atr=1.5, edge_atr=0.15):
    """Узкая проторговка (EN-9): наибольшее N ≥ 3 последних баров (включая D), у которых High−Low всей коробки
    ≤ max_h_atr·ATR и уровень p у края коробки (≤ edge_atr·ATR)."""
    H, Lo = df["High"].values, df["Low"].values
    best = None
    for N in range(3, min(60, len(df)) + 1):
        hi, lo = H[-N:].max(), Lo[-N:].min()
        if hi - lo > max_h_atr * atr:
            break
        if min(abs(p - hi), abs(p - lo)) <= edge_atr * atr:
            best = (N, (hi - lo) / atr, hi, lo)
    if not best:
        return {"box_days": 0, "box_height_atr": None, "box_edge_touches": 0}
    N, h, hi, lo = best
    edge = hi if abs(p - hi) <= abs(p - lo) else lo
    touches = int(sum(1 for i in range(1, N + 1) if (abs(H[-i] - edge) <= 0.1 * atr if edge == hi else abs(Lo[-i] - edge) <= 0.1 * atr)))
    return {"box_days": N, "box_height_atr": round(h, 2), "box_edge_touches": touches}


def bar_features(df, a14, atr, level, levels, cfg=None):
    """Признаки последнего бара и подхода к уровню — для предпосылок."""
    last = df.iloc[-1]; prev = df.iloc[-2]
    Hd, Ld, Cd, Od = map(float, (last["High"], last["Low"], last["Close"], last["Open"]))
    rng = max(Hd - Ld, 1e-9)
    p = level["price"]
    closes = df["Close"].values
    # движение к уровню без откатов
    run = 0
    toward_up = p >= Cd
    for i in range(len(closes) - 1, 0, -1):
        if (closes[i] > closes[i - 1]) == toward_up:
            run += 1
        else:
            break
    run_move = abs(closes[-1] - closes[-1 - run]) / atr if run else 0.0
    tr = true_range(df)
    LL = (cfg or {}).get("levels", {})
    para_last3 = int((tr.iloc[-3:] >= LL.get("paranormal_tr_atr", 1.5) * a14).sum())
    # EN-6: сколько ATR пройдено за день D в сторону уровня; исключение — новый 20-дневный экстремум с закрытием у него
    pc = float(prev["Close"])
    day_move = ((Cd - pc) if toward_up else (pc - Cd)) / atr
    hh20 = float(df["High"].iloc[-21:-1].max()); ll20 = float(df["Low"].iloc[-21:-1].min())
    cpos = (Cd - Ld) / rng
    ext_close = (Hd >= hh20 and cpos >= 0.8) if toward_up else (Ld <= ll20 and cpos <= 0.2)
    # маленькие бары: ≥3 из последних 4 с High−Low ≤ 0,7 ATR
    hl4 = (df["High"] - df["Low"]).values[-4:]
    small_bars = int((hl4 <= 0.7 * atr).sum()) >= 3
    # поджатие: лои растут к сопротивлению / хаи падают к поддержке (3+ бара)
    lows, highs = df["Low"].values[-4:], df["High"].values[-4:]
    squeeze = bool(np.all(np.diff(lows) > 0)) if toward_up else bool(np.all(np.diff(highs) < 0))
    # следующие уровни (запас хода)
    above = sorted([lv for lv in levels if lv["price"] > p + 0.1 * atr and lv["strength"] >= 3], key=lambda x: x["price"])
    below = sorted([lv for lv in levels if lv["price"] < p - 0.1 * atr and lv["strength"] >= 3], key=lambda x: -x["price"])
    nxt_up = above[0] if above else None
    nxt_dn = below[0] if below else None
    ma20 = float(df["Close"].iloc[-20:].mean()); ma50 = float(df["Close"].iloc[-50:].mean())
    vol20 = float(df["Volume"].iloc[-21:-1].mean())
    return {
        "close_pos_in_bar": round((Cd - Ld) / rng, 2),
        "range_atr": round((Hd - Ld) / atr, 2),
        "chg_pct": round((Cd / float(prev["Close"]) - 1) * 100, 2),
        "gap_atr": round((Od - float(prev["Close"])) / atr, 2),
        "rel_vol": round(float(last["Volume"]) / vol20, 2) if vol20 else None,
        "run_bars_to_level": run,
        "run_move_atr": round(run_move, 2),
        "paranormal_last3": para_last3,
        "day_move_atr": round(day_move, 2),
        "ext_close": bool(ext_close),
        "small_bars": bool(small_bars),
        **box_features(df, atr, p),
        "squeeze": squeeze,
        "next_up": {"price": nxt_up["price"], "type": nxt_up["type"], "strength": nxt_up["strength"],
                    "room_atr": round((nxt_up["price"] - p) / atr, 2)} if nxt_up else None,
        "next_down": {"price": nxt_dn["price"], "type": nxt_dn["type"], "strength": nxt_dn["strength"],
                      "room_atr": round((p - nxt_dn["price"]) / atr, 2)} if nxt_dn else None,
        "trend": {"above_ma20": Cd > ma20, "above_ma50": Cd > ma50, "ma20_over_ma50": ma20 > ma50},
    }
