"""Ночной сканер: все акции NASDAQ+NYSE по фильтрам → сильные уровни → бар у уровня.

Запуск:  python scan.py                 # за последнюю завершённую сессию
         python scan.py --date 2026-10-22   # как будто сейчас вечер этого дня (прогон по истории)
Результат: results/<D>/summary.json (кандидаты у уровней), results/<D>/bars/<T>.json (свечи),
           results/eval/<предыдущий D>.json (как сходили вчерашние кандидаты), results/latest.json.
"""
import argparse, io, json, os, re, sys, time, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from levels import find_levels, near_levels, bar_features, atr14, atr_gerchik, true_range
from assess import assess

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
OUT = os.path.join(ROOT, "results")
NY = ZoneInfo("America/New_York")
SECTOR_ETF = {"XLK": "Технологии", "XLF": "Финансы", "XLE": "Энергетика", "XLV": "Здравоохранение",
              "XLI": "Промышленность", "XLY": "Потреб. циклич.", "XLP": "Потреб. базовые", "XLU": "Коммунальные",
              "XLB": "Материалы", "XLRE": "Недвижимость", "XLC": "Связь"}
BAD_NAME = re.compile(r"warrant|right|unit|preferred|depositary|\bnotes?\b|debenture|\betf\b|\bfund\b|"
                      r"acquisition corp|\btrust\b.*units|%|subordinated|\bseries [a-z]\b", re.I)


def log(*a):
    print(*a, flush=True)


# ---------- вселенная ----------

def universe():
    base = "https://www.nasdaqtrader.com/dynamic/SymDir/"
    rows = []
    t = requests.get(base + "nasdaqlisted.txt", timeout=60).text
    df = pd.read_csv(io.StringIO(t), sep="|")
    df = df[df["Symbol"].notna() & ~df["Symbol"].astype(str).str.startswith("File Creation")]
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N")]
    for s, nm in zip(df["Symbol"], df["Security Name"]):
        rows.append((str(s), str(nm), "NASDAQ"))
    t = requests.get(base + "otherlisted.txt", timeout=60).text
    df = pd.read_csv(io.StringIO(t), sep="|")
    df = df[df["ACT Symbol"].notna() & ~df["ACT Symbol"].astype(str).str.startswith("File Creation")]
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N") & (df["Exchange"] == "N")]
    for s, nm in zip(df["ACT Symbol"], df["Security Name"]):
        rows.append((str(s), str(nm), "NYSE"))
    out = []
    for s, nm, ex in rows:
        if "$" in s or BAD_NAME.search(nm):
            continue
        out.append({"ticker": s.replace(".", "-"), "name": nm, "exchange": ex})
    return out


# ---------- данные ----------

def download(tickers, period, chunk=150, tries=3):
    data = {}
    for k in range(0, len(tickers), chunk):
        part = tickers[k:k + chunk]
        for attempt in range(tries):
            try:
                raw = yf.download(part, period=period, interval="1d", group_by="ticker", auto_adjust=False,
                                  threads=True, progress=False)
                break
            except Exception as e:
                log("download error", e); time.sleep(5 * (attempt + 1)); raw = None
        if raw is None or raw.empty:
            continue
        for t in part:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) > 30:
                    d.index = pd.to_datetime(d.index).tz_localize(None)
                    data[t] = d
            except Exception:
                pass
        log(f"  загружено {min(k + chunk, len(tickers))}/{len(tickers)}")
    return data


def session_date(as_of):
    if as_of:
        return pd.Timestamp(as_of)
    now = dt.datetime.now(NY)
    d = now.date() if now.hour * 60 + now.minute >= 16 * 60 + 10 else now.date() - dt.timedelta(days=1)
    return pd.Timestamp(d)


# ---------- главное ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD — прогон как будто вечер этого дня")
    ap.add_argument("--limit", type=int, default=0, help="только N тикеров (для проверки)")
    ap.add_argument("--skip-if-done", action="store_true", help="не пересчитывать, если день уже посчитан")
    ap.add_argument("--intraday", action="store_true",
                    help="разовый срез во время сессии: D = сегодня (незавершённый бар), запись в results/intraday/<D>, без eval и latest")
    a = ap.parse_args()
    U, LV = CFG["universe"], CFG

    target = pd.Timestamp(dt.datetime.now(NY).date()) if a.intraday else session_date(a.date)
    spy = download(["SPY"], "2y")["SPY"]
    spy = spy[spy.index <= target]
    D = spy.index[-1]
    Ds = D.strftime("%Y-%m-%d")
    log("Сессия D =", Ds)
    if a.skip_if_done and os.path.exists(os.path.join(OUT, Ds, "summary.json")):
        log("Этот день уже посчитан — пропуск."); return

    uni = universe()
    if a.limit:
        uni = uni[:a.limit]
    meta = {u["ticker"]: u for u in uni}
    log("Вселенная после отсева не-обыкновенных:", len(uni))

    # шаг 1: быстрые фильтры по 6 месяцам
    quick = download(list(meta), "2y" if a.date else "6mo")
    passed = {}
    for t, d in quick.items():
        d = d[d.index <= D]
        if len(d) < 60 or d.index[-1] != D:
            continue
        c, v = float(d["Close"].iloc[-1]), float(d["Volume"].iloc[-2 if a.intraday else -1])  # внутри сессии объём дня неполный — берём вчерашний
        v90 = float(d["Volume"].iloc[-90:].mean())
        a14 = atr14(d)
        if c > U["min_price"] and v > U["min_volume_day"] and v90 > U["min_avg_volume_90d"] and a14 > U["min_atr14"]:
            passed[t] = d
    log("Прошли фильтры:", len(passed))

    # шаг 2: год+ истории для прошедших
    full = download(list(passed), "2y") if not a.date else {t: quick[t] for t in passed}
    rows, bars_out = [], {}
    for t, d in full.items():
        d = d[d.index <= D]
        if len(d) < 60 or d.index[-1] != D:
            continue
        try:
            lv, a14 = find_levels(d, CFG)
            near, atr = near_levels(d, lv, a14, CFG)
        except Exception as e:
            log("ошибка", t, e); continue
        if not near:
            continue
        main_lv = near[0]
        feats = bar_features(d, a14, atr, main_lv, lv, CFG)
        rows.append({"ticker": t, "name": meta[t]["name"], "exchange": meta[t]["exchange"],
                     "close": round(float(d["Close"].iloc[-1]), 2), "atr14": round(a14, 2), "atr": round(atr, 2),
                     "atr_method": round(atr_gerchik(d, a14), 2),  # ATR метода проверки (конституция п.3) — не меняется
                     "level": main_lv, "other_near": near[1:3], **feats})
        rows[-1]["assess"] = assess(rows[-1], CFG)
        o = CFG["output"]
        dd = d.iloc[-o["bars_daily"]:]
        w = d.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna().iloc[-o["bars_weekly"]:]
        bars_out[t] = {"daily": [[i.strftime("%Y-%m-%d"), *[round(float(x), 2) for x in r]] for i, r in
                                 zip(dd.index, dd[["Open", "High", "Low", "Close", "Volume"]].values)],
                       "weekly": [[i.strftime("%Y-%m-%d"), *[round(float(x), 2) for x in r]] for i, r in
                                  zip(w.index, w[["Open", "High", "Low", "Close"]].values)],
                       "levels": [x for x in lv if x["strength"] >= 3]}

    rows.sort(key=lambda r: (not r["level"]["in_threshold"], -r["level"]["strength"], r["level"]["dist_close_atr"]))
    in_thr = [r for r in rows if r["level"]["in_threshold"]]
    log(f"У сильного уровня: в полосе {len(rows)}, в пороге {len(in_thr)}")

    # фон: SPY, QQQ, сектора
    bg_t = ["SPY", "QQQ"] + list(SECTOR_ETF)
    bgd = download(bg_t, "6mo" if not a.date else "2y")
    bg = []
    for t in bg_t:
        d = bgd.get(t)
        if d is None:
            continue
        d = d[d.index <= D]
        if len(d) < 25:
            continue
        c = d["Close"].values
        bg.append({"ticker": t, "name": SECTOR_ETF.get(t, t), "close": round(float(c[-1]), 2),
                   "chg_pct": round((c[-1] / c[-2] - 1) * 100, 2), "chg5_pct": round((c[-1] / c[-6] - 1) * 100, 2),
                   "above_ma20": bool(c[-1] > c[-20:].mean())})

    day_dir = os.path.join(OUT, "intraday", Ds) if a.intraday else os.path.join(OUT, Ds)
    os.makedirs(os.path.join(day_dir, "bars"), exist_ok=True)
    summary = {"date": Ds, "made_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "config_version": CFG["version"], "universe": len(uni), "passed_filters": len(passed),
               "near_band": len(rows), "in_threshold": len(in_thr), "background": bg, "candidates": rows}
    json.dump(summary, open(os.path.join(day_dir, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for t, b in bars_out.items():
        json.dump(b, open(os.path.join(day_dir, "bars", f"{t}.json"), "w", encoding="utf-8"), ensure_ascii=False)

    if a.intraday:
        log("Срез внутри сессии сохранён:", day_dir); return
    evaluate_previous(D, full)
    json.dump({"date": Ds, "summary": f"results/{Ds}/summary.json"},
              open(os.path.join(OUT, "latest.json"), "w"), ensure_ascii=False)
    log("Готово.")


def evaluate_previous(D, data):
    """Как сходили кандидаты предыдущего дня: движение вверх/вниз от их закрытия в ATR.
    Сторону (лонг/шорт) знают агенты — здесь только сырые цифры для статистики по корзинам."""
    days = sorted(x for x in os.listdir(OUT) if re.match(r"\d{4}-\d{2}-\d{2}$", x) and x < D.strftime("%Y-%m-%d"))
    if not days:
        return
    prev = days[-1]
    sm = json.load(open(os.path.join(OUT, prev, "summary.json"), encoding="utf-8"))
    res, nsess = [], None
    for r in sm["candidates"]:
        d = data.get(r["ticker"])
        if d is None:
            continue
        nxt = d[d.index > pd.Timestamp(prev)]
        if nxt.empty or nxt.index[0] > D:
            continue
        b = nxt.iloc[0]; c0, atr = r["close"], r.get("atr_method", r["atr"])  # проверка — всегда в ATR метода
        nsess = nxt.index[0].strftime("%Y-%m-%d")
        res.append({"ticker": r["ticker"], "level": r["level"]["price"], "type": r["level"]["type"],
                    "strength": r["level"]["strength"], "position": r["level"]["position"],
                    "dist_close_atr": r["level"]["dist_close_atr"], "in_threshold": r["level"]["in_threshold"],
                    "stage": (r.get("assess") or {}).get("stage"),
                    "best": {k: ((r.get("assess") or {}).get("best") or {}).get(k) for k in ("scenario", "side", "n")},
                    "up_atr": round((float(b["High"]) - c0) / atr, 2), "down_atr": round((c0 - float(b["Low"])) / atr, 2),
                    "close_atr": round((float(b["Close"]) - c0) / atr, 2), "gap_atr": round((float(b["Open"]) - c0) / atr, 2),
                    "broke_level": bool((r["level"]["position"] == "сверху" and float(b["High"]) > r["level"]["price"]) or
                                        (r["level"]["position"] == "снизу" and float(b["Low"]) < r["level"]["price"]))})
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    json.dump({"date": prev, "next_session": nsess, "items": res},
              open(os.path.join(OUT, "eval", f"{prev}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("Оценка", prev, ":", len(res))


if __name__ == "__main__":
    main()
