"""Бесплатный отсев кодом (до агентов): заражённая зона, запас хода, измеримые предпосылки.

Для кандидата у сильного уровня рассматриваются два сценария:
  пробой — движение сквозь уровень (уровень сверху → лонг, снизу → шорт);
  отбой/ЛП — движение от уровня обратно (уровень сверху → шорт, снизу → лонг).
Сценарий допустим, только если есть запас хода ≥ min_room_atr до следующего уровня,
а для пробоя ещё и свободная зона за уровнем (цена не была за ним ≥ free_zone_days дней).
Затем по допустимому сценарию считаются измеримые предпосылки; нужно ≥ min_premises.
"""


def _has(marks, word):
    return any(word in m for m in marks)


def assess(c, cfg):
    A = cfg["assess"]
    lv = c["level"]
    pos = lv["position"]                     # "сверху" / "снизу" от закрытия
    marks = lv.get("marks", [])
    db = lv.get("days_beyond")               # None — за уровнем не была > 6 мес.
    rt = lv.get("retest_days")
    tr = c.get("trend", {})
    up_room = (c.get("next_up") or {}).get("room_atr")      # от уровня вверх
    dn_room = (c.get("next_down") or {}).get("room_atr")    # от уровня вниз
    free = db is None or db >= A["free_zone_days"]
    cp = c.get("close_pos_in_bar", 0.5)
    trend_up = tr.get("above_ma20") and tr.get("ma20_over_ma50")
    trend_dn = (not tr.get("above_ma20")) and (not tr.get("ma20_over_ma50"))
    out = []

    # ---- пробой ----
    b_side = "long" if pos == "сверху" else "short"
    b_room = up_room if pos == "сверху" else dn_room
    b_ok_room = b_room is None or b_room >= A["min_room_atr"]
    reasons = []
    if not free:
        reasons.append(f"заражённая зона за уровнем ({db} дн.)")
    if not b_ok_room:
        reasons.append(f"мало запаса хода ({b_room} ATR)")
    prem = []
    if free:
        prem.append("свободная зона" + (" (исторический экстремум)" if db is None else f" {db} дн."))
    if (b_side == "long" and cp >= A["close_extreme"]) or (b_side == "short" and cp <= 1 - A["close_extreme"]):
        prem.append("закрытие у " + ("хая" if b_side == "long" else "лоя") + " дня")
    if lv["dist_close_atr"] <= A["close_at_level_atr"]:
        prem.append("закрытие у уровня")
    if rt is not None and rt <= A["near_retest_days"]:
        prem.append(f"ближний ретест {rt} дн.")
    if c.get("squeeze"):
        prem.append("поджатие к уровню")
    if (b_side == "long" and trend_up) or (b_side == "short" and trend_dn):
        prem.append("по тренду (MA20/MA50)")
    if (c.get("rel_vol") or 0) >= A["rel_vol"]:
        prem.append(f"повышенный объём ×{c['rel_vol']}")
    if b_room is None:
        prem.append("впереди пусто (нет уровней)")
    out.append({"scenario": "пробой", "side": b_side, "room_atr": b_room, "viable": free and b_ok_room,
                "reject": reasons, "premises": prem, "n": len(prem)})

    # ---- отбой / ЛП ----
    r_side = "short" if pos == "сверху" else "long"
    r_room = dn_room if pos == "сверху" else up_room
    r_ok_room = r_room is None or r_room >= A["min_room_atr"]
    reasons = [] if r_ok_room else [f"мало запаса хода ({r_room} ATR)"]
    prem = []
    if not free:
        prem.append(f"заражённая зона за уровнем {db} дн.")
    if c.get("run_move_atr", 0) >= A["long_run_atr"] and c.get("run_bars_to_level", 0) >= A["long_run_bars"]:
        prem.append(f"длинное безоткатное движение ({c['run_bars_to_level']} бар., {c['run_move_atr']} ATR)")
    if c.get("paranormal_last3", 0) >= 1:
        prem.append("подход на паранормальных барах")
    if lv.get("dist_open_atr", 0) >= A["far_open_atr"]:
        prem.append("открытие далеко от уровня")
    if lv["dist_close_atr"] >= A["far_close_atr"]:
        prem.append("закрытие далеко от уровня")
    if rt is None or rt >= A["far_retest_days"]:
        prem.append("дальний ретест" + ("" if rt is None else f" {rt} дн."))
    if _has(marks, "максимум года") or _has(marks, "минимум года"):
        prem.append("экстремум года")
    if (r_side == "long" and cp >= A["close_extreme"]) or (r_side == "short" and cp <= 1 - A["close_extreme"]):
        prem.append("закрытие в сторону отбоя (у " + ("хая" if r_side == "long" else "лоя") + ")")
    if (r_side == "long" and trend_up) or (r_side == "short" and trend_dn):
        prem.append("по тренду (MA20/MA50)")
    out.append({"scenario": "отбой/ЛП", "side": r_side, "room_atr": r_room, "viable": r_ok_room,
                "reject": reasons, "premises": prem, "n": len(prem)})

    good = [s for s in out if s["viable"] and s["n"] >= A["min_premises"]]
    good.sort(key=lambda s: -s["n"])
    if good:
        stage, why = "agents", ""
    elif not any(s["viable"] for s in out):
        stage, why = "zone_room", "; ".join(r for s in out for r in s["reject"])
    else:
        stage, why = "premises", f"меньше {A['min_premises']} предпосылок (" + \
            ", ".join(f"{s['scenario']}: {s['n']}" for s in out if s["viable"]) + ")"
    return {"scenarios": out, "pass": bool(good), "best": good[0] if good else None,
            "stage": stage, "reject_reason": why}
