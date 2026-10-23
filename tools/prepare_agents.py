"""Подготовка данных сканера для ночных агентов.

python tools/prepare_agents.py <D> quick [G]
    → work/<D>/meta.json (фон + воронка отсева), work/<D>/quick_<i>.json (компактные строки БЕЗ свечей
      по всем, кто прошёл отсев кодом: зона, запас хода, ≥ N предпосылок), work/<D>/rejected_code.json.
python tools/prepare_agents.py <D> deep <G> <T1,T2,...>
    → work/<D>/deep_<i>.json (полные строки + дневки 60 + недельки 26) только по выбранным тикерам.
"""
import json, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from assess import assess

D, mode = sys.argv[1], sys.argv[2]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 4
cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
sm = json.load(open(os.path.join(ROOT, "results", D, "summary.json"), encoding="utf-8"))
out = os.path.join(ROOT, "work", D)
os.makedirs(out, exist_ok=True)
cands = [c for c in sm["candidates"] if c["level"]["in_threshold"]]
for c in cands:
    if "assess" not in c:
        c["assess"] = assess(c, cfg)

LV = ["price", "type", "strength", "base_strength", "tf", "marks", "position", "dist_close_atr", "dist_open_atr",
      "days_beyond", "retest_days"]


def compact(c):
    a = c["assess"]
    return {"ticker": c["ticker"], "close": c["close"], "atr": c["atr"], "chg_pct": c["chg_pct"], "rel_vol": c["rel_vol"],
            "gap_atr": c["gap_atr"], "close_pos_in_bar": c["close_pos_in_bar"], "range_atr": c["range_atr"],
            "level": {k: c["level"].get(k) for k in LV}, "trend": c["trend"],
            "scenarios": [{k: s[k] for k in ["scenario", "side", "room_atr", "premises", "n"]}
                          for s in a["scenarios"] if s["viable"] and s["n"] >= cfg["assess"]["min_premises"]]}


if mode == "quick":
    passed = [c for c in cands if c["assess"]["pass"]]
    passed.sort(key=lambda c: -c["assess"]["best"]["n"])
    for g in range(G):
        json.dump([compact(c) for c in passed[g::G]], open(os.path.join(out, f"quick_{g + 1}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
    rej = [{"ticker": c["ticker"], "stage": c["assess"]["stage"], "reason": c["assess"]["reject_reason"]}
           for c in cands if not c["assess"]["pass"]]
    json.dump(rej, open(os.path.join(out, "rejected_code.json"), "w", encoding="utf-8"), ensure_ascii=False)
    funnel = {"universe": sm["universe"], "passed_filters": sm["passed_filters"], "near_band": sm["near_band"],
              "in_threshold": len(cands),
              "rejected_zone_room": sum(1 for r in rej if r["stage"] == "zone_room"),
              "rejected_premises": sum(1 for r in rej if r["stage"] == "premises"),
              "to_agents": len(passed)}
    meta = {"date": sm["date"], "made_at": sm["made_at"], "config_version": cfg["version"], "funnel": funnel,
            "quick_groups": G, "background": sm["background"]}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(funnel, ensure_ascii=False))
elif mode == "deep":
    want = [t.strip() for t in sys.argv[4].split(",") if t.strip()]
    by = {c["ticker"]: c for c in cands}
    rows = []
    for t in want:
        c = by.get(t)
        if not c:
            print("нет кандидата", t); continue
        b = json.load(open(os.path.join(ROOT, "results", D, "bars", t + ".json"), encoding="utf-8"))
        r = {k: v for k, v in c.items() if k not in ("assess",)}
        r["scenarios"] = compact(c)["scenarios"]
        r["bars_daily"] = [x[:5] for x in b["daily"]]
        r["bars_weekly"] = b["weekly"]
        rows.append(r)
    for g in range(G):
        json.dump(rows[g::G], open(os.path.join(out, f"deep_{g + 1}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"deep: {len(rows)} тикеров в {G} файлах")
