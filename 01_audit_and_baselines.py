"""
Step 1-2: reproducible audit, whole-day expanding folds, baselines.

Usage:  python 01_audit_and_baselines.py --data <dir with the 4 CSVs> --out results

Rules followed:
  * October 2025 prices are not used to train or select any model here (held out for the final test).
    The descriptive audit() reads the whole labelled file, October included; nothing from it feeds modelling.
  * Every statistic / imputation / outlier rule is learned from the fold's training rows only.
  * Folds split on whole days (month boundaries), so one day never straddles train/validation.
  * Evaluation is on ALL validation rows (no outlier removal on the evaluation side).
  * All targets are compared in dollars after converting back to a price.
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

EQUIP = ["Dry Van", "Flatbed", "Reefer"]
HOLDOUT_START = pd.Timestamp("2025-10-01")


# ------------------------------------------------------------------ load
def find_file(d, stem):
    """Accept both naming styles used for the assessment files, e.g. train_test.csv and train-test.csv."""
    for name in (f"{stem}.csv", f"{stem.replace('_', '-')}.csv"):
        if os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    raise FileNotFoundError(f"{stem}.csv (or hyphenated variant) not found in {d}")


def load(d):
    tr = pd.read_csv(find_file(d, "train_test"), parse_dates=["date"])
    va = pd.read_csv(find_file(d, "validation"), parse_dates=["date"])
    dec = pd.read_csv(find_file(d, "december_chart_inputs"), parse_dates=["date"])
    return tr, va, dec


def haversine(a, b, c, d):
    a, b, c, d = map(np.radians, [a, b, c, d])
    x = np.sin((c - a) / 2) ** 2 + np.cos(a) * np.cos(c) * np.sin((d - b) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(x))


# ------------------------------------------------------------------ audit (descriptive only)
def audit(tr, va, dec):
    a = {}
    a["train_rows"], a["val_rows"], a["dec_rows"] = len(tr), len(va), len(dec)
    a["train_dates"] = [str(tr.date.min().date()), str(tr.date.max().date())]
    a["val_dates"] = [str(va.date.min().date()), str(va.date.max().date())]
    a["val_rows_by_month"] = {str(k): int(v) for k, v in va.groupby(va.date.dt.to_period("M")).size().items()}
    a["train_cities"] = len(set(tr.pickup) | set(tr.delivery))
    a["train_lanes"] = int(tr.groupby(["pickup", "delivery"]).ngroups)
    a["train_dup_rows_excluding_id"] = int(tr.drop(columns="load_id").duplicated().sum())

    # weight
    for n, d in [("train", tr), ("val", va)]:
        a[f"{n}_weight_negative"] = int((d.weight < 0).sum())
        a[f"{n}_weight_nan"] = int(d.weight.isna().sum())
        a[f"{n}_weight_abs_eq_47500"] = int((d.weight.abs() == 47500).sum())
        a[f"{n}_market_index_nan"] = int(d.market_index.isna().sum())
    pos, neg = tr.weight[tr.weight > 0], tr.weight[tr.weight < 0].abs()
    a["weight_pos_quantiles"] = pos.quantile([.01, .5, .99]).round(0).tolist()
    a["weight_neg_abs_quantiles"] = neg.quantile([.01, .5, .99]).round(0).tolist()

    # new cities: rows counted ONCE if either end is new
    new = (set(va.pickup) | set(va.delivery)) - (set(tr.pickup) | set(tr.delivery))
    p, d = va.pickup.isin(new), va.delivery.isin(new)
    a["new_cities"] = sorted(new)
    a["val_rows_new_pickup"], a["val_rows_new_delivery"] = int(p.sum()), int(d.sum())
    a["val_rows_both_ends_new"] = int((p & d).sum())
    a["val_rows_any_new_city"] = int((p | d).sum())  # = pickup + delivery - both
    a["val_share_any_new_city"] = round(float((p | d).mean()), 4)
    lanes = set(zip(tr.pickup, tr.delivery))
    seen = np.array([(x, y) in lanes for x, y in zip(va.pickup, va.delivery)])
    a["val_rows_lane_seen_in_train"] = round(float(seen.mean()), 4)
    a["val_rows_lane_seen_among_no_new_city"] = round(float(seen[~(p | d).values].mean()), 4)

    # coordinate consistency: one coordinate per city? shared coordinates across cities? vs distance?
    long = pd.concat([
        pd.concat([x[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
                   x[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)])
        for x in (tr, va)])
    a["cities_with_more_than_one_coordinate"] = int((long.groupby("city")[["lat", "lon"]].nunique().max(axis=1) > 1).sum())
    shared = long.drop_duplicates().groupby(["lat", "lon"]).city.nunique()
    a["coordinates_shared_by_multiple_cities"] = {f"{k[0]},{k[1]}": int(v) for k, v in shared[shared > 1].items()}
    for n, x in [("train", tr), ("val", va)]:
        r = x.distance / haversine(x.pickup_lat, x.pickup_lon, x.delivery_lat, x.delivery_lon)
        a[f"{n}_distance_over_haversine_q"] = r.quantile([.01, .5, .99]).round(3).tolist()
        a[f"{n}_distance_lt_haversine"] = int((r < 1).sum())
    r = va.distance / haversine(va.pickup_lat, va.pickup_lon, va.delivery_lat, va.delivery_lon)
    a["val_dist_over_hav_median_for_new_city_rows"] = round(float(r[(p | d).values].median()), 3)
    a["val_dist_over_hav_median_for_other_rows"] = round(float(r[~(p | d).values].median()), 3)

    # Dec lane
    one = dec.drop_duplicates(["pickup", "delivery", "distance", "equipment", "weight"])
    a["dec_distinct_specs"] = len(one)
    m = tr[(tr.pickup == one.pickup.iat[0]) & (tr.delivery == one.delivery.iat[0])]
    a["dec_lane_train_loads"] = len(m)
    a["dec_lane_train_distance_range"] = [float(m.distance.min()), float(m.distance.max())] if len(m) else None
    return a


# ------------------------------------------------------------------ folds (whole days)
def make_folds(df, first="2025-05", last="2025-09"):
    day = df.date.dt.normalize()
    for m in pd.period_range(first, last, freq="M"):
        s, e = m.start_time.normalize(), m.end_time.normalize()
        tri, vai = df.index[day < s], df.index[(day >= s) & (day <= e)]
        assert set(day[tri]).isdisjoint(set(day[vai])), "a day straddles train/val"
        assert day[vai].max() < HOLDOUT_START, "October leaked into folds"
        yield str(m), tri, vai


# ------------------------------------------------------------------ features
def weight_fill(train, mode):
    w = train.weight.abs() if mode == "abs" else train.weight.where(train.weight > 0)
    return w.groupby(train.equipment).median()


def build(df, med_w, mode):
    X = pd.DataFrame(index=df.index)
    d = df.distance / 1000
    X["log_d"], X["d"], X["d2"] = np.log(df.distance), d, d ** 2
    for e in EQUIP[1:]:
        X[f"eq_{e}"] = (df.equipment == e).astype(float)
    for k in range(1, 7):
        X[f"dow_{k}"] = (df.date.dt.dayofweek == k).astype(float)
    w = df.weight
    X["w_missing"] = w.isna().astype(float)
    if mode == "abs":
        w = w.abs()
    else:  # negative -> treated as missing, with an explicit "invalid" flag
        X["w_invalid"] = (w < 0).astype(float)
        w = w.where(~(w < 0))
    X["w"] = w.fillna(df.equipment.map(med_w)) / 1e4
    return X


def fit_predict(tr, va, target, mode, trim, alpha=1.0, k=5.0):
    med_w = weight_fill(tr, mode)
    Xtr, Xva = build(tr, med_w, mode), build(va, med_w, mode)
    keep = np.ones(len(tr), bool)
    if trim:  # rule learned from this fold's TRAIN rows only: robust z of log-residual
        sc = StandardScaler().fit(Xtr)
        base = Ridge(alpha=alpha).fit(sc.transform(Xtr), np.log(tr.posted_rate))
        r = np.log(tr.posted_rate) - base.predict(sc.transform(Xtr))
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = (np.abs(r - np.median(r)) <= k * mad).values
    t, Xk = tr[keep], Xtr[keep]
    sc = StandardScaler().fit(Xk)
    A, B = sc.transform(Xk), sc.transform(Xva)
    if target == "raw":
        pred = Ridge(alpha=alpha).fit(A, t.posted_rate).predict(B)
    elif target == "rpm":
        pred = Ridge(alpha=alpha).fit(A, t.posted_rate / t.distance).predict(B) * va.distance.values
    else:  # log / log_nosmear
        m = Ridge(alpha=alpha).fit(A, np.log(t.posted_rate))
        smear = np.mean(np.exp(np.log(t.posted_rate) - m.predict(A))) if target == "log" else 1.0
        pred = np.exp(m.predict(B)) * smear
    return np.clip(pred, 1.0, None), int((~keep).sum())


def simple_baseline(name, tr, va):
    if name == "rpm_sum_global":
        return va.distance * tr.posted_rate.sum() / tr.distance.sum()
    rpm = tr.posted_rate / tr.distance
    if name == "rpm_median_global":
        return va.distance * rpm.median()
    return va.distance * va.equipment.map(rpm.groupby(tr.equipment).median())  # rpm_median_by_equipment


def metrics(y, p):
    e = np.abs(y - p)
    return dict(MAE=e.mean(), RMSE=np.sqrt(np.mean((y - p) ** 2)), MedAE=np.median(e))


# ------------------------------------------------------------------ composition vs time (dev data only)
def month_effects(dev, k=5.0):
    """Month effect on log(rate) controlling for distance, equipment, weight, pickup and delivery city.
    Reference month = September (October is held out)."""
    d = dev.copy()
    w = d.weight.abs()
    w = w.fillna(w.groupby(d.equipment).transform("median"))
    M = pd.get_dummies(d.date.dt.to_period("M").astype(str), prefix="m").drop(columns="m_2025-09")
    X = pd.concat([pd.DataFrame({"c": 1.0, "log_d": np.log(d.distance), "d": d.distance / 1e3,
                                 "d2": (d.distance / 1e3) ** 2, "w": w / 1e4}),
                   pd.get_dummies(d.equipment, prefix="eq", drop_first=True),
                   pd.get_dummies(d.pickup, prefix="pu", drop_first=True),
                   pd.get_dummies(d.delivery, prefix="dl", drop_first=True), M], axis=1).astype(float)
    y = np.log(d.posted_rate).values

    def ols(Xm, ym):
        return np.linalg.lstsq(Xm.values, ym, rcond=None)[0]

    b_all = ols(X, y)
    r = y - X.values @ b_all
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r - np.median(r)) <= k * mad
    b_trim = ols(X[keep], y[keep])
    cols = [c for c in X.columns if c.startswith("m_")]
    idx = [list(X.columns).index(c) for c in cols]
    raw = (d.posted_rate / d.distance).groupby(d.date.dt.to_period("M").astype(str)).median()
    out = pd.DataFrame({"month": [c[2:] for c in cols],
                        "adj_effect_vs_Sep_pct_all_rows": (np.exp(b_all[idx]) - 1) * 100,
                        "adj_effect_vs_Sep_pct_trimmed": (np.exp(b_trim[idx]) - 1) * 100})
    out["raw_median_rpm"] = out.month.map(raw)
    out.loc[len(out)] = ["2025-09", 0.0, 0.0, raw["2025-09"]]
    return out.sort_values("month").round(3)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="."), ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr, va, dec = load(a.data)

    aud = audit(tr, va, dec)
    json.dump(aud, open(f"{a.out}/audit.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(aud, indent=2, ensure_ascii=False))

    dev = tr[tr.date < HOLDOUT_START].reset_index(drop=True)  # Jan-Sep only
    print(f"\nDEV rows (Jan-Sep): {len(dev)} | October rows held out from training/selection: {(tr.date >= HOLDOUT_START).sum()}")

    rows, preds = [], {}
    configs = [("rpm_sum_global", None, None), ("rpm_median_global", None, None), ("rpm_median_by_equipment", None, None)]
    configs += [("ridge", t, (m, tm)) for t in ["raw", "rpm", "log_nosmear", "log"]
                for m in ["abs", "invalid_flag"] for tm in [False, True]]
    for name, t, opt in configs:
        label = name if t is None else f"ridge|target={t}|weight_neg={opt[0]}|train={'trimmed' if opt[1] else 'all'}"
        ys, ps, fold_mae, dropped = [], [], [], []
        for fold, tri, vai in make_folds(dev):
            trf, vaf = dev.loc[tri], dev.loc[vai]
            if t is None:
                p = simple_baseline(name, trf, vaf).values; nd = 0
            else:
                p, nd = fit_predict(trf, vaf, t, "abs" if opt[0] == "abs" else "flag", opt[1])
            ys.append(vaf.posted_rate.values); ps.append(p); dropped.append(nd)
            fold_mae.append(metrics(vaf.posted_rate.values, p)["MAE"])
        m = metrics(np.concatenate(ys), np.concatenate(ps))
        rows.append(dict(model=label, **m, fold_MAE_mean=np.mean(fold_mae), fold_MAE_worst=np.max(fold_mae),
                         mean_rows_dropped_in_training=np.mean(dropped)))
    res = pd.DataFrame(rows).sort_values("MAE").round(2)
    res.to_csv(f"{a.out}/baselines_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print("\n=== Pooled expanding-window results (May-Sep folds, all validation rows, dollars) ===")
    print(res.to_string(index=False))

    me = month_effects(dev)
    me.to_csv(f"{a.out}/month_effects.csv", index=False)
    print("\n=== Composition-adjusted month effects (Jan-Sep only, reference = Sep) ===")
    print(me.to_string(index=False))


if __name__ == "__main__":
    main()
