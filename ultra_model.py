"""가까운 6시간용 '초단기예보 → 안양대' 모델 (data/ultra_model.json).

학습: 입력 = 초단기예보(기온·습도·풍속) + 예보 시차 + 시간·월 + 발표 정시 관측, 정답 = 안양대 500m 격자.
지금 모델(AWS 관측으로 학습)에 예보를 넣으면 입력이 어긋나 예보 그대로보다 나빴던 문제를 고친 모델.
2022~23 학습 → 2024~25 시험: 종합 74.27점 / ±1°C 70.6% (예보 그대로 68.63점 / 64.6%).
사이트에 쓰는 파일은 2022~2025년 전부로 다시 학습한 것. lightgbm 없이 나무를 직접 따라가 계산한다.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from functools import lru_cache

import config

PATH = config.BASE_DIR / "data" / "ultra_model.json"


@lru_cache(maxsize=1)
def model() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def _value(node, x) -> float:
    while "v" not in node:
        v = x[node["f"]]
        if not math.isfinite(v):
            node = node["l"] if node["d"] else node["r"]
        else:
            node = node["l"] if v <= node["t"] else node["r"]
    return node["v"]


def predict(fcst: dict, valid: datetime, issue_hour: datetime, obs: dict) -> dict:
    """fcst = 그 시각 초단기예보 {temp, humidity, wind}, issue_hour = 발표 정시(HH30 발표 → HH:00),
    obs = 발표 정시의 관측(만안구 초단기실황) {temp, humidity, wind} → 안양대 {temp, humidity, wind}."""
    m = model()
    lead = (valid - issue_hour).total_seconds() / 3600
    x = [fcst["temp"], fcst["humidity"], fcst["wind"], lead,
         math.sin(2 * math.pi * valid.hour / 24), math.cos(2 * math.pi * valid.hour / 24),
         math.sin(2 * math.pi * valid.month / 12), math.cos(2 * math.pi * valid.month / 12),
         obs["temp"], obs["humidity"], obs["wind"]]
    d = {t: sum(_value(tree, x) for tree in m["trees"][t]) for t in ("uni_ta", "uni_hm", "uni_ws")}
    return {"temp": round(fcst["temp"] + d["uni_ta"], 2),
            "humidity": round(min(100.0, max(0.0, fcst["humidity"] + d["uni_hm"])), 2),
            "wind": round(max(0.0, fcst["wind"] + d["uni_ws"]), 2)}


def issue_hour(base: str) -> datetime:
    """초단기예보 발표 표시 'YYYYMMDD HH30' → 발표 정시 datetime(HH:00)."""
    return datetime.strptime(base.split()[0] + base.split()[1][:2], "%Y%m%d%H")
