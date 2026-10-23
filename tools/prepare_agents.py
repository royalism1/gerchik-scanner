"""Подготовка данных сканера для ночных агентов.
python tools/prepare_agents.py <D> [N] [n_groups]
Пишет work/<D>/meta.json (фон, счётчики), work/<D>/group_<i>.json (кандидаты + свечи для аналитика i),
work/<D>/all_ranked.json (все в пороге, ранжированные, без свечей)."""
import json, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1]
cfg = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
N = int(sys.argv[2]) if len(sys.argv) > 2 else cfg["output"]["max_for_agents"]
G = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sm = json.load(open(os.path.join(ROOT, "results", D, "summary.json"), encoding="utf-8"))
def score(c):
    l = c["level"]; m = " ".join(l["marks"]); s = l["base_strength"] if "base_strength" in l else l["strength"]
    s += 0.5 * (l["strength"] - s)                  # бонусы (круглая, неделька, гэп) — вполсилы
    if "пилится" in m: s -= 2
    if "свободная зона" in m: s += 1
    if l.get("days_beyond") is not None and l["days_beyond"] <= 2: s -= 0.5
    if (c.get("rel_vol") or 0) >= 1.5: s += 0.5
    return s - l["dist_close_atr"]

ranked = sorted([c for c in sm["candidates"] if c["level"]["in_threshold"]], key=score, reverse=True)
out = os.path.join(ROOT, "work", D); os.makedirs(out, exist_ok=True)
drop = {"other_near"}
def slim(c):
    r = {k: v for k, v in c.items() if k not in drop}
    r["other_near"] = [{k: o.get(k) for k in ["price", "type", "strength", "position", "dist_close_atr"]} for o in c["other_near"]]
    return r
json.dump([{"rank": i + 1, "ticker": c["ticker"], "score": round(score(c), 2), "level": c["level"]["price"],
            "type": c["level"]["type"], "strength": c["level"]["strength"], "position": c["level"]["position"]}
           for i, c in enumerate(ranked)], open(os.path.join(out, "all_ranked.json"), "w", encoding="utf-8"), ensure_ascii=False)
top = ranked[:N]
for g in range(G):
    part = top[g::G]
    rows = []
    for c in part:
        b = json.load(open(os.path.join(ROOT, "results", D, "bars", c["ticker"] + ".json"), encoding="utf-8"))
        rows.append({**slim(c), "bars_daily": [x[:5] for x in b["daily"]], "bars_weekly": b["weekly"]})
    json.dump(rows, open(os.path.join(out, f"group_{g + 1}.json"), "w", encoding="utf-8"), ensure_ascii=False)
meta = {k: sm[k] for k in ["date", "made_at", "config_version", "universe", "passed_filters", "near_band", "in_threshold"]}
meta.update({"sent_to_agents": len(top), "groups": G, "background": sm["background"]})
json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: meta[k] for k in ["universe", "passed_filters", "in_threshold", "sent_to_agents"]}))
