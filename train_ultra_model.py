"""가까운 6시간용 '초단기예보 → 안양대' 모델 학습 → data/ultra_model.json (사이트가 바로 씀).

    python train_ultra_model.py --csv-dir <초단기예보 CSV 폴더>

- CSV: 기상청 기상자료개방포털 '동네예보(초단기예보)' 안양5동(격자 59,123) 기온·습도·풍속.
  파일 이름에 기온/습도/풍속이 들어가면 기간별 여러 개를 자동으로 합친다 (드라이브 '오늘춥나_초단기모델' 폴더에 보관).
- 정답: data/train_data.csv(2023~2025) + data/grid/grid_2022_*.csv·data/22train.csv(2022)의 안양대 500m 격자.
- 입력: 초단기예보 + 예보 시차(발표 정시 기준) + 시간·월 + 발표 정시 만안 관측(운영에서는 초단기실황).
- 비교 시험은 Colab 노트북 '6_ultra_forecast_model'(드라이브)에서: 2022~23 학습 → 2024~25 시험
  종합 74.27점 / ±1°C 70.6% (예보 그대로 68.63점 / 64.6%, 이전 방식 53.37점 / 48.1%).
"""
import argparse, glob, json, os, re, unicodedata
import numpy as np, pandas as pd, lightgbm as lgb

REPO = os.path.dirname(os.path.abspath(__file__))
KIND = {"기온": "f_ta", "습도": "f_hm", "풍속": "f_ws"}


def load_ultra(path, col):
    rows, base = [], None
    with open(path, encoding="utf-8-sig", errors="ignore") as fh:
        for line in fh:
            m = re.search(r"Start\s*:\s*(\d{8})", line)
            if m:
                base = pd.Timestamp(m.group(1)); continue
            p = [x.strip() for x in line.split(",")]
            if len(p) < 4 or not p[0].isdigit() or base is None:
                continue
            hhmm = p[1].zfill(4)
            issue_utc = base.replace(day=int(p[0])) + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[2:]))
            rows.append((issue_utc, int(p[2].replace("+", "")), float(p[3])))
    df = pd.DataFrame(rows, columns=["issue_utc", "lead", col])
    df["issue"] = df.issue_utc + pd.Timedelta(hours=9)
    df["valid"] = df.issue.dt.floor("h") + pd.to_timedelta(df.lead, unit="h")
    return df[["issue", "valid", "lead", col]]


def kma_feels(month, ta, rh, ws):
    month, ta, rh, ws = (np.asarray(x, float) for x in (month, ta, rh, ws))
    v = (ws * 3.6) ** 0.16
    win = np.where((ta <= 10) & (ws >= 1.3), 13.12 + 0.6215*ta - 11.37*v + 0.3965*v*ta, ta)
    tw = (ta*np.arctan(0.151977*np.sqrt(rh+8.313659)) + np.arctan(ta+rh) - np.arctan(rh-1.67633)
          + 0.00391838*rh**1.5*np.arctan(0.023101*rh) - 4.686035)
    sm = -0.2442 + 0.55399*tw + 0.45535*ta - 0.0022*tw**2 + 0.00278*tw*ta + 3.0
    return np.where((month >= 5) & (month <= 9), sm, win)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "ultra_model.json"))
    args = ap.parse_args()
    files = {c: [] for c in KIND.values()}
    for n in sorted(os.listdir(args.csv_dir)):
        name = unicodedata.normalize("NFC", n)
        col = next((c for k, c in KIND.items() if k in name), None)
        if col and name.lower().endswith(".csv"):
            files[col].append(os.path.join(args.csv_dir, n))
    missing = [k for k, c in KIND.items() if not files[c]]
    if missing:
        raise SystemExit(f"이름에 {missing} 이(가) 들어간 CSV가 {args.csv_dir} 에 없음")
    d = {c: pd.concat([load_ultra(p, c) for p in ps]).drop_duplicates(["issue", "lead"]) for c, ps in files.items()}
    fc = d["f_ta"].merge(d["f_hm"], on=["issue", "valid", "lead"]).merge(d["f_ws"], on=["issue", "valid", "lead"])
    bad = (fc.f_ta < -50) | (fc.f_hm < 0) | (fc.f_ws < 0)
    fc = fc[~bad].reset_index(drop=True)
    print(f"예보 {len(fc):,}줄 (결측 {int(bad.sum())}), {fc.issue.min()} ~ {fc.issue.max()}")

    train = pd.read_csv(f"{REPO}/data/train_data.csv", parse_dates=["tm"])
    g22 = pd.concat([pd.read_csv(f, encoding="utf-8-sig", parse_dates=["tm"])
                     for f in sorted(glob.glob(f"{REPO}/data/grid/grid_2022_*.csv"))])
    a22 = pd.read_csv(f"{REPO}/data/22train.csv", encoding="cp949")
    a22 = pd.DataFrame({"tm": pd.to_datetime(a22["일시"]), "aws_ta": a22["기온(°C)"],
                        "aws_hm": a22["습도(%)"], "aws_ws": a22["풍속(m/s)"]})
    t22 = g22.rename(columns={"ta": "uni_ta", "hm": "uni_hm", "ws_10m": "uni_ws"})[["tm", "uni_ta", "uni_hm", "uni_ws"]] \
             .merge(a22, on="tm").dropna()
    train = pd.concat([t22, train[t22.columns]], ignore_index=True).drop_duplicates("tm").sort_values("tm")
    print("정답 해별:", train.tm.dt.year.value_counts().sort_index().to_dict())
    obs = train.set_index("tm")[["aws_ta", "aws_hm", "aws_ws"]].rename(columns=lambda c: "o_" + c[4:])
    data = fc.merge(train[["tm", "uni_ta", "uni_hm", "uni_ws"]], left_on="valid", right_on="tm").drop(columns="tm")
    data["obs_time"] = data.issue.dt.floor("h")
    data = data.merge(obs, left_on="obs_time", right_index=True, how="left").dropna().reset_index(drop=True)
    data["month"], data["hour"] = data.valid.dt.month, data.valid.dt.hour
    for k, p in (("hour", 24), ("month", 12)):
        data[k + "_sin"], data[k + "_cos"] = np.sin(2*np.pi*data[k]/p), np.cos(2*np.pi*data[k]/p)
    data["real"] = kma_feels(data.month, data.uni_ta, data.uni_hm, data.uni_ws)
    cnt = data.valid.dt.year.value_counts()
    years = sorted(int(y) for y, n in cnt.items() if n >= 3000)
    data = data[data.valid.dt.year.isin(years)].reset_index(drop=True)
    print("해별 줄:", data.valid.dt.year.value_counts().sort_index().to_dict())

    F_OBS = ["f_ta", "f_hm", "f_ws", "lead", "hour_sin", "hour_cos", "month_sin", "month_cos", "o_ta", "o_hm", "o_ws"]
    TARGETS = {"uni_ta": "f_ta", "uni_hm": "f_hm", "uni_ws": "f_ws"}

    def node(n):
        if "leaf_value" in n:
            return {"v": round(float(n["leaf_value"]), 6)}
        assert n["decision_type"] == "<=", n["decision_type"]
        return {"f": int(n["split_feature"]), "t": float(n["threshold"]), "d": bool(n.get("default_left", True)),
                "l": node(n["left_child"]), "r": node(n["right_child"])}


    def value(nd, x):
        while "v" not in nd:
            v = x[nd["f"]]
            nd = (nd["l"] if nd["d"] else nd["r"]) if not np.isfinite(v) else (nd["l"] if v <= nd["t"] else nd["r"])
        return nd["v"]



    trees, models = {}, {}
    for tgt, base in TARGETS.items():
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0)
        m.fit(data[F_OBS], data[tgt] - data[base])
        models[tgt] = m
        trees[tgt] = [node(t["tree_structure"]) for t in m.booster_.dump_model()["tree_info"]]

    # 내보낸 나무로 계산한 값이 lightgbm 결과와 같은지 확인
    X = data[F_OBS].sample(300, random_state=1).to_numpy(float)
    for tgt, m in models.items():
        a = m.predict(X)
        b = np.array([sum(value(t, x) for t in trees[tgt]) for x in X])
        print(tgt, "최대 차이", float(np.abs(a - b).max()))

    years = sorted(int(y) for y in data.valid.dt.year.unique())
    out = {"kind": "ultra-mos", "features": F_OBS, "bases": TARGETS, "trained_years": years,
           "rows": int(len(data)),
           "note": "입력=초단기예보(기온·습도·풍속) + 예보 시차(발표 정시 기준 1~6시간) + 시간·월 + 발표 정시 관측(운영: 만안구 초단기실황). "
                   "정답=안양대 500m 격자. 예측=예보+보정량. 2022~23 학습→2024~25 시험: 종합 74.27점, ±1°C 70.6% (예보 그대로 68.63점, 64.6%)",
           "trees": trees}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print("저장", args.out, years, len(data))
