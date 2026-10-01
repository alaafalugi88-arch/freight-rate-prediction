"""
Step 3: LightGBM / CatBoost experiments on the SAME whole-day expanding folds as 01_audit_and_baselines.py.

  python 02_tree_experiments.py --data <dir> --out results --part main     # objectives + feature sets (LightGBM)
  python 02_tree_experiments.py --data <dir> --out results --part city     # unseen-city holdout test
  python 02_tree_experiments.py --data <dir> --out results --part cat      # CatBoost
  python 02_tree_experiments.py --data <dir> --out results --part horizon  # 2-month-ahead windows + strong linear reference
  python 02_tree_experiments.py --data <dir> --out results --part blend    # reproducible blend + saved predictions + bootstrap CI
  python 02_tree_experiments.py --data <dir> --out results --part round3   # CatBoost w/ time; market_index vs quote_signal (info only)
  python 02_tree_experiments.py --data <dir> --out results --part city2 [--windows 2025-05 ...]  # hidden cities on candidates + size control
  python 02_tree_experiments.py --data <dir> --out results --part city2_summary
  python 02_tree_experiments.py --data <dir> --out results --part roundF     # candidate F vs B+C (final round)
  python 02_tree_experiments.py --data <dir> --out results --part cityF      # hidden cities for F (needs city2 parts)
  python 02_tree_experiments.py --data <dir> --out results --part pathtest   # December-file path check

Notes
  * Dev data = Jan-Sep 2025 only. October prices are not used to train or select models (final test).
  * Hyper-parameters are FIXED and untuned on purpose (tuning comes after feature/target choices).
  * Evaluation is always on all validation rows, in dollars.
  * Feature sets are labelled by whether they exist in the December file:
      base/lane/time  -> derivable from the December file
      coords          -> NOT in the December file; would need a city->coordinate lookup (one coordinate per city in the data)
      market          -> NOT available for December (information only, never deployable)
"""
import argparse, importlib.util, os, time
import numpy as np, pandas as pd, lightgbm as lgb

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_audit_and_baselines.py")
_s = importlib.util.spec_from_file_location("base", _p)
base = importlib.util.module_from_spec(_s); _s.loader.exec_module(base)

EQUIP, T0 = base.EQUIP, pd.Timestamp("2025-01-01")
LGB = dict(n_estimators=700, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, cat_smooth=20, min_data_per_group=50,
           verbose=-1, n_jobs=1, random_state=0)


def stats(tr):
    lanes = (tr.pickup + ">" + tr.delivery).unique()
    return dict(med_w=tr.weight.abs().groupby(tr.equipment).median(),
                cities=sorted(set(tr.pickup) | set(tr.delivery)), lanes=sorted(lanes))


def feats(df, st, fs):
    X = pd.DataFrame(index=df.index)
    X["distance"], X["log_distance"] = df.distance, np.log(df.distance)
    X["w_missing"] = df.weight.isna().astype(float)
    X["weight"] = df.weight.abs().fillna(df.equipment.map(st["med_w"]))
    X["dow"] = df.date.dt.dayofweek
    X["equipment"] = pd.Categorical(df.equipment, categories=EQUIP)
    if "nocity" not in fs:
        X["pickup"] = pd.Categorical(df.pickup, categories=st["cities"])
        X["delivery"] = pd.Categorical(df.delivery, categories=st["cities"])
    if "lane" in fs:
        X["lane"] = pd.Categorical(df.pickup + ">" + df.delivery, categories=st["lanes"])
    if "coords" in fs:
        for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
            X[c] = df[c]
    if "time" in fs:
        X["t"] = (df.date - T0).dt.days
    if "dmi" in fs:  # daily mean market_index, precomputed per batch by add_daily_index()
        X["mi_daily"], X["mi_daily_missing"] = df["mi_daily"], df["mi_daily_missing"]
    if "market" in fs or "mi" in fs:
        X["market_index"] = df.market_index
    if "market" in fs or "qs" in fs:
        X["quote_signal"] = df.quote_signal
    return X


def unk_mask(X, p, seed=0):
    """Randomly blank ONE city end (and the lane) so the model learns a fallback for never-seen cities."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    pick = rng.random(len(X)) < p
    side = rng.random(len(X)) < 0.5
    for col, m in [("pickup", pick & side), ("delivery", pick & ~side)]:
        if col in X:
            X.loc[m, col] = np.nan
    if "lane" in X:
        X.loc[pick, "lane"] = np.nan
    return X


def run_lgb(tr, va, fs, obj, unk=0.0, params=None):
    """Returns {variant_name: predictions in dollars}."""
    st = stats(tr)
    Xtr, Xva = feats(tr, st, fs), feats(va, st, fs)
    if unk > 0:
        Xtr = unk_mask(Xtr, unk)
    cat = [c for c in ["equipment", "pickup", "delivery", "lane"] if c in Xtr]
    P = {**LGB, **(params or {})}
    rate, d = tr.posted_rate.values, tr.distance.values
    if obj.startswith("log"):
        y = np.log(rate)
        kw = {"log_l2": dict(objective="regression"), "log_l1": dict(objective="l1"),
              "log_huber": dict(objective="huber", alpha=0.15)}[obj]
        m = lgb.LGBMRegressor(**P, **kw).fit(Xtr, y, categorical_feature=cat)
        raw = m.predict(Xva)
        if obj == "log_l2":
            smear = np.mean(np.exp(y - m.predict(Xtr)))
            return {"log_l2_nosmear": np.exp(raw), "log_l2_smear": np.exp(raw) * smear}
        return {obj: np.exp(raw)}
    if obj == "raw_l1":
        m = lgb.LGBMRegressor(**P, objective="l1").fit(Xtr, rate, categorical_feature=cat)
        return {obj: m.predict(Xva)}
    if obj == "rpm_l1":
        m = lgb.LGBMRegressor(**P, objective="l1").fit(Xtr, rate / d, categorical_feature=cat)
        return {obj: m.predict(Xva) * va.distance.values}
    raise ValueError(obj)


def run_cat(tr, va, fs, iters=400):
    from catboost import CatBoostRegressor
    st = stats(tr)
    Xtr, Xva = feats(tr, st, fs), feats(va, st, fs)
    cats = [c for c in ["equipment", "pickup", "delivery", "lane"] if c in Xtr]
    for X in (Xtr, Xva):
        for c in cats:
            X[c] = X[c].astype(object).where(X[c].notna(), "UNK").astype(str)
    m = CatBoostRegressor(loss_function="MAE", iterations=iters, learning_rate=0.1, depth=6,
                          thread_count=1, verbose=0, random_seed=0, cat_features=cats)
    m.fit(Xtr, np.log(tr.posted_rate))
    return {"log_mae_catboost": np.exp(m.predict(Xva))}


def summarize(label, ys, ps, fold_mae):
    m = base.metrics(np.concatenate(ys), np.concatenate(ps))
    return dict(model=label, **m, fold_MAE_mean=np.mean(fold_mae), fold_MAE_worst=np.max(fold_mae))


def eval_config(dev, runner, tag):
    ys, ps, fm = {}, {}, {}
    for fold, tri, vai in base.make_folds(dev):
        trf, vaf = dev.loc[tri], dev.loc[vai]
        out = runner(trf, vaf)
        for k, p in out.items():
            p = np.clip(p, 1.0, None)
            ys.setdefault(k, []).append(vaf.posted_rate.values); ps.setdefault(k, []).append(p)
            fm.setdefault(k, []).append(base.metrics(vaf.posted_rate.values, p)["MAE"])
    return [summarize(f"{tag}|{k}", ys[k], ps[k], fm[k]) for k in ys]


def part_main(dev, out):
    rows, t0 = [], time.time()
    for obj in ["log_l2", "log_l1", "log_huber", "raw_l1", "rpm_l1"]:
        rows += eval_config(dev, lambda a, b, o=obj: run_lgb(a, b, frozenset(), o), f"lgbm|feat=base|obj={obj}")
        print(f"[{time.time()-t0:5.0f}s] objective {obj} done", flush=True)
    variants = {"nocity": {"nocity"}, "lane": {"lane"}, "coords": {"coords"}, "time": {"time"},
                "coords+time": {"coords", "time"}, "coords+time+lane": {"coords", "time", "lane"},
                "market(NOT deployable)": {"market"}}
    for name, fs in variants.items():
        rows += eval_config(dev, lambda a, b, f=frozenset(fs): run_lgb(a, b, f, "log_l1"), f"lgbm|feat=base+{name}|obj=log_l1")
        print(f"[{time.time()-t0:5.0f}s] features {name} done", flush=True)
    res = pd.DataFrame(rows).sort_values("MAE").round(2)
    res.to_csv(f"{out}/lgbm_main.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(res.to_string(index=False))


def part_cat(dev, out):
    rows, t0 = [], time.time()
    for name, fs in {"base": set(), "base+lane": {"lane"}}.items():
        rows += eval_config(dev, lambda a, b, f=frozenset(fs): run_cat(a, b, f), f"catboost|feat={name}")
        print(f"[{time.time()-t0:5.0f}s] catboost {name} done", flush=True)
    res = pd.DataFrame(rows).sort_values("MAE").round(2)
    res.to_csv(f"{out}/catboost.csv", index=False)
    print(res.to_string(index=False))


def part_city(dev, out, n_cities=6, n_draws=4):
    """Hide n_cities from training (any end) and score only loads that touch them, vs the same rows when seen."""
    rng = np.random.default_rng(42)
    cities = np.array(sorted(set(dev.pickup) | set(dev.delivery)))
    draws = [set(rng.choice(cities, n_cities, replace=False)) for _ in range(n_draws)]
    folds = [f for f in base.make_folds(dev) if f[0] in ("2025-07", "2025-08", "2025-09")]
    V = {"no city info": (frozenset({"nocity"}), 0.0), "city names only": (frozenset(), 0.0),
         "names+coords": (frozenset({"coords"}), 0.0), "names+coords+unk-mask": (frozenset({"coords"}), 0.1),
         "coords only (no names)": (frozenset({"coords", "nocity"}), 0.0)}
    acc = {v: dict(y=[], unseen=[], seen=[]) for v in V}
    t0 = time.time()
    for fold, tri, vai in folds:
        trf, vaf = dev.loc[tri], dev.loc[vai]
        seen_pred = {v: run_lgb(trf, vaf, fs, "log_l1", unk)["log_l1"] for v, (fs, unk) in V.items()}
        for H in draws:
            mv = (vaf.pickup.isin(H) | vaf.delivery.isin(H)).values
            mt = ~(trf.pickup.isin(H) | trf.delivery.isin(H)).values
            for v, (fs, unk) in V.items():
                p = run_lgb(trf[mt], vaf[mv], fs, "log_l1", unk)["log_l1"]
                acc[v]["y"].append(vaf.posted_rate.values[mv]); acc[v]["unseen"].append(p)
                acc[v]["seen"].append(seen_pred[v][mv])
        print(f"[{time.time()-t0:5.0f}s] fold {fold} done", flush=True)
    rows = []
    for v, a in acc.items():
        y, u, s = map(np.concatenate, (a["y"], a["unseen"], a["seen"]))
        rows.append(dict(variant=v, rows=len(y), MAE_city_seen=np.abs(y - s).mean(),
                         MAE_city_unseen=np.abs(y - u).mean(), MedAE_unseen=np.median(np.abs(y - u))))
    res = pd.DataFrame(rows)
    res["loss_from_unseen"] = res.MAE_city_unseen - res.MAE_city_seen
    res = res.sort_values("MAE_city_unseen").round(2)
    res.to_csv(f"{out}/city_holdout.csv", index=False)
    print(res.to_string(index=False))


def ridge_city(tr, va, time_feat):
    """Strong linear reference: log target, city one-hots (+ optional linear time), smearing."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    med_w = base.weight_fill(tr, "abs")
    A, B = base.build(tr, med_w, "abs"), base.build(va, med_w, "abs")
    cities = sorted(set(tr.pickup) | set(tr.delivery))
    for X, df in ((A, tr), (B, va)):
        oh = {f"pu_{c}": (df.pickup == c).astype(float) for c in cities}
        oh.update({f"dl_{c}": (df.delivery == c).astype(float) for c in cities})
        for k, v in oh.items():
            X[k] = v
        if time_feat:
            X["t"] = (df.date - T0).dt.days / 100.0
    sc = StandardScaler().fit(A)
    y = np.log(tr.posted_rate.values)
    m = Ridge(alpha=1.0).fit(sc.transform(A), y)
    smear = np.mean(np.exp(y - m.predict(sc.transform(A))))
    return {"ridge": np.exp(m.predict(sc.transform(B))) * smear}


def part_horizon(dev, out):
    """2-month-ahead windows (like Nov-Dec after an Oct cut-off). Reports MAE for month 1 and month 2 separately."""
    starts = ["2025-05", "2025-06", "2025-07", "2025-08"]
    V = {"lgbm names": lambda a, b: run_lgb(a, b, frozenset(), "log_l1")["log_l1"],
         "lgbm names+time": lambda a, b: run_lgb(a, b, frozenset({"time"}), "log_l1")["log_l1"],
         "lgbm coords-only": lambda a, b: run_lgb(a, b, frozenset({"coords", "nocity"}), "log_l1")["log_l1"],
         "lgbm coords-only+time": lambda a, b: run_lgb(a, b, frozenset({"coords", "nocity", "time"}), "log_l1")["log_l1"],
         "lgbm coords-only+time (huber)": lambda a, b: run_lgb(a, b, frozenset({"coords", "nocity", "time"}), "log_huber")["log_huber"],
         "lgbm no-city+time": lambda a, b: run_lgb(a, b, frozenset({"nocity", "time"}), "log_l1")["log_l1"],
         "ridge+city": lambda a, b: ridge_city(a, b, False)["ridge"],
         "ridge+city+linear time": lambda a, b: ridge_city(a, b, True)["ridge"]}
    acc = {v: dict(y=[], p=[], h=[]) for v in V}
    day = dev.date.dt.normalize()
    for st in starts:
        s = pd.Period(st, "M")
        tri = dev.index[day < s.start_time]
        vai = dev.index[(day >= s.start_time) & (day <= (s + 1).end_time.normalize())]
        trf, vaf = dev.loc[tri], dev.loc[vai]
        h = (vaf.date.dt.to_period("M") - s).apply(lambda x: x.n).values + 1
        for v, fn in V.items():
            p = np.clip(fn(trf, vaf), 1.0, None)
            acc[v]["y"].append(vaf.posted_rate.values); acc[v]["p"].append(p); acc[v]["h"].append(h)
        print(f"window starting {st} done", flush=True)
    rows = []
    for v, a in acc.items():
        y, p, h = map(np.concatenate, (a["y"], a["p"], a["h"]))
        e = np.abs(y - p)
        rows.append(dict(model=v, MAE_all=e.mean(), MAE_month1=e[h == 1].mean(), MAE_month2=e[h == 2].mean(),
                         MedAE=np.median(e), RMSE=np.sqrt(np.mean((y - p) ** 2))))
    res = pd.DataFrame(rows).sort_values("MAE_all").round(2)
    res.to_csv(f"{out}/horizon2.csv", index=False)
    print(res.to_string(index=False))


BLEND_MODELS = {
    "A_lgbm_names_time": lambda a, b: run_lgb(a, b, frozenset({"time"}), "log_l1")["log_l1"],
    "B_lgbm_coords_time": lambda a, b: run_lgb(a, b, frozenset({"coords", "nocity", "time"}), "log_l1")["log_l1"],
    "C_ridge_city": lambda a, b: ridge_city(a, b, False)["ridge"],
    "D_lgbm_names_notime": lambda a, b: run_lgb(a, b, frozenset(), "log_l1")["log_l1"],
}
# Fixed, untuned equal weights (decided before looking at blend results; not fitted on the windows).
BLENDS = {"A+C": ["A_lgbm_names_time", "C_ridge_city"], "B+C": ["B_lgbm_coords_time", "C_ridge_city"],
          "A+B+C": ["A_lgbm_names_time", "B_lgbm_coords_time", "C_ridge_city"],
          "A+D": ["A_lgbm_names_time", "D_lgbm_names_notime"]}
WINDOW_STARTS = ["2025-05", "2025-06", "2025-07", "2025-08"]


def day_block_bootstrap(err_a, err_b, unit, n_boot=2000, seed=0):
    """CI for MAE(a) - MAE(b), resampling whole (window, day) blocks with replacement."""
    codes, u = pd.factorize(unit)
    sa, sb = np.bincount(codes, err_a), np.bincount(codes, err_b)
    n = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    d = sa[idx].sum(1) / n[idx].sum(1) - sb[idx].sum(1) / n[idx].sum(1)
    return float(err_a.mean() - err_b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def part_blend(dev, out):
    """Reproducible blend evaluation on 2-month-ahead windows; saves every prediction."""
    day = dev.date.dt.normalize()
    frames = []
    for st in WINDOW_STARTS:
        sp = pd.Period(st, "M")
        tri = dev.index[day < sp.start_time]
        vai = dev.index[(day >= sp.start_time) & (day <= (sp + 1).end_time.normalize())]
        trf, vaf = dev.loc[tri], dev.loc[vai]
        f = pd.DataFrame({"load_id": vaf.load_id.values, "window_start": st,
                          "train_end": str((sp.start_time - pd.Timedelta(days=1)).date()),
                          "date": vaf.date.dt.date.astype(str).values,
                          "horizon_month": (vaf.date.dt.to_period("M") - sp).apply(lambda x: x.n).values + 1,
                          "actual": vaf.posted_rate.values})
        for name, fn in BLEND_MODELS.items():
            f[name] = np.clip(fn(trf, vaf), 1.0, None)
        frames.append(f)
        print(f"window {st} done", flush=True)
    P = pd.concat(frames, ignore_index=True)
    for b, parts in BLENDS.items():
        P[f"blend_{b}"] = P[parts].mean(axis=1)
    P.to_csv(f"{out}/blend_window_predictions.csv", index=False)

    cols = list(BLEND_MODELS) + [f"blend_{b}" for b in BLENDS]
    rows = []
    for c in cols:
        e = (P.actual - P[c]).abs()
        rows.append(dict(model=c, MAE_all=e.mean(), MAE_month1=e[P.horizon_month == 1].mean(),
                         MAE_month2=e[P.horizon_month == 2].mean(), MedAE=e.median(),
                         RMSE=np.sqrt(((P.actual - P[c]) ** 2).mean()),
                         worst_window_month_MAE=e.groupby([P.window_start, P.horizon_month]).mean().max()))
    summ = pd.DataFrame(rows).sort_values("MAE_all").round(2)
    summ.to_csv(f"{out}/blend_summary.csv", index=False)

    per = P.assign(**{c: (P.actual - P[c]).abs() for c in cols}).groupby(["window_start", "horizon_month"])[cols].mean()
    per.insert(0, "n_evaluations", P.groupby(["window_start", "horizon_month"]).size())
    per.round(2).to_csv(f"{out}/blend_per_window.csv")

    unit = P.window_start + "|" + P.date
    pairs = [("blend_A+C", "A_lgbm_names_time"), ("blend_A+C", "C_ridge_city"), ("blend_A+C", "blend_B+C")]
    ci = []
    for a_, b_ in pairs:
        d, lo, hi = day_block_bootstrap((P.actual - P[a_]).abs().values, (P.actual - P[b_]).abs().values, unit)
        ci.append(dict(comparison=f"{a_} minus {b_}", MAE_diff=d, ci95_low=lo, ci95_high=hi))
    ci = pd.DataFrame(ci).round(2)
    ci.to_csv(f"{out}/blend_bootstrap.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\nEvaluations: {len(P)} (window x load; a load can appear in two windows) | unique loads: {P.load_id.nunique()}")
    print(summ.to_string(index=False)); print("\nPer window x horizon MAE:\n", per.round(1).to_string())
    print("\nDay-block bootstrap (negative = first is better):\n", ci.to_string(index=False))


# ------------------------------------------------------------------ round 3 (selection rule fixed before running)
def window_split(dev, st):
    day = dev.date.dt.normalize()
    sp = pd.Period(st, "M")
    tri = dev.index[day < sp.start_time]
    vai = dev.index[(day >= sp.start_time) & (day <= (sp + 1).end_time.normalize())]
    trf, vaf = dev.loc[tri], dev.loc[vai]
    h = (vaf.date.dt.to_period("M") - sp).apply(lambda x: x.n).values + 1
    return trf, vaf, h


def summarize_cols(P, cols):
    rows = []
    for c in cols:
        e = (P.actual - P[c]).abs()
        rows.append(dict(model=c, MAE_all=e.mean(), MAE_month1=e[P.horizon_month == 1].mean(),
                         MAE_month2=e[P.horizon_month == 2].mean(), MedAE=e.median(),
                         worst_window_month_MAE=e.groupby([P.window_start, P.horizon_month]).mean().max()))
    return pd.DataFrame(rows).sort_values("MAE_all").round(2)


def part_round3(dev, out):
    """(1) CatBoost with the SAME features as A (names+time); (2) market_index and quote_signal separately.
    market_index / quote_signal are NOT in the December file: information only, not candidates."""
    M = {"A_lgbm_names_time": BLEND_MODELS["A_lgbm_names_time"],
         "C_ridge_city": BLEND_MODELS["C_ridge_city"],
         "E_catboost_names_time": lambda a, b: run_cat(a, b, frozenset({"time"}))["log_mae_catboost"],
         "INFO_lgbm_names_time+market_index": lambda a, b: run_lgb(a, b, frozenset({"time", "mi"}), "log_l1")["log_l1"],
         "INFO_lgbm_names_time+quote_signal": lambda a, b: run_lgb(a, b, frozenset({"time", "qs"}), "log_l1")["log_l1"]}
    frames = []
    for st in WINDOW_STARTS:
        trf, vaf, h = window_split(dev, st)
        f = pd.DataFrame({"load_id": vaf.load_id.values, "window_start": st, "horizon_month": h,
                          "date": vaf.date.dt.date.astype(str).values, "actual": vaf.posted_rate.values})
        for k, fn in M.items():
            f[k] = np.clip(fn(trf, vaf), 1.0, None)
        frames.append(f)
        print(f"window {st} done", flush=True)
    P = pd.concat(frames, ignore_index=True)
    P["blend_A+C"] = P[["A_lgbm_names_time", "C_ridge_city"]].mean(axis=1)
    P["blend_E+C"] = P[["E_catboost_names_time", "C_ridge_city"]].mean(axis=1)
    P.to_csv(f"{out}/round3_window_predictions.csv", index=False)
    cols = [c for c in P.columns if c not in ("load_id", "window_start", "horizon_month", "date", "actual")]
    summ = summarize_cols(P, cols)
    summ.to_csv(f"{out}/round3_summary.csv", index=False)
    per = P.assign(**{c: (P.actual - P[c]).abs() for c in cols}).groupby(["window_start", "horizon_month"])[cols].mean()
    per.round(2).to_csv(f"{out}/round3_per_window.csv")
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False)); print("\nPer window x horizon:\n", per.round(1).to_string())


def part_city2(dev, out, windows, n_cities=6, n_draws=4):
    """Hidden-city test on the CANDIDATES (A, B, C and blends A+C, B+C), 2-month windows.
    Three training conditions, scored on the same evaluation rows (loads touching the hidden cities):
      seen    : full training set
      control : same number of training rows removed at random from loads NOT touching the hidden cities
      unseen  : every training row touching a hidden city removed
    Caveat: control keeps every hidden-city example (raising their share of the remaining training data), so
    control vs unseen compares "removing those cities' data" with "removing other cities' data". It does not
    isolate a pure new-city effect from the change in training composition.
    """
    rng = np.random.default_rng(42)
    cities = np.array(sorted(set(dev.pickup) | set(dev.delivery)))
    draws = [sorted(rng.choice(cities, n_cities, replace=False)) for _ in range(n_draws)]
    models = {k: BLEND_MODELS[k] for k in ["A_lgbm_names_time", "B_lgbm_coords_time", "C_ridge_city"]}
    os.makedirs(f"{out}/city2_parts", exist_ok=True)
    for st in windows:
        trf, vaf, h = window_split(dev, st)
        seen = {k: np.clip(fn(trf, vaf), 1.0, None) for k, fn in models.items()}
        frames = []
        for di, H in enumerate(draws):
            Hs = set(H)
            mv = (vaf.pickup.isin(Hs) | vaf.delivery.isin(Hs)).values
            touch = (trf.pickup.isin(Hs) | trf.delivery.isin(Hs)).values
            drop = np.random.default_rng(1000 + di).choice(np.flatnonzero(~touch), touch.sum(), replace=False)
            keep_ctrl = np.ones(len(trf), bool); keep_ctrl[drop] = False
            v = vaf[mv]
            f = pd.DataFrame({"load_id": v.load_id.values, "window_start": st, "draw": di,
                              "hidden_cities": "|".join(H), "horizon_month": h[mv],
                              "train_rows_removed": int(touch.sum()), "actual": v.posted_rate.values})
            for k, fn in models.items():
                f[f"{k}|seen"] = seen[k][mv]
                f[f"{k}|control"] = np.clip(fn(trf[keep_ctrl], v), 1.0, None)
                f[f"{k}|unseen"] = np.clip(fn(trf[~touch], v), 1.0, None)
            frames.append(f)
        pd.concat(frames, ignore_index=True).to_csv(f"{out}/city2_parts/{st}.csv", index=False)
        print(f"window {st} done", flush=True)


def part_city2_summary(dev, out):
    P = pd.concat([pd.read_csv(f"{out}/city2_parts/{st}.csv") for st in WINDOW_STARTS], ignore_index=True)
    for cond in ["seen", "control", "unseen"]:
        P[f"blend_A+C|{cond}"] = P[[f"A_lgbm_names_time|{cond}", f"C_ridge_city|{cond}"]].mean(axis=1)
        P[f"blend_B+C|{cond}"] = P[[f"B_lgbm_coords_time|{cond}", f"C_ridge_city|{cond}"]].mean(axis=1)
    P.to_csv(f"{out}/city2_predictions.csv", index=False)
    names = ["A_lgbm_names_time", "B_lgbm_coords_time", "C_ridge_city", "blend_A+C", "blend_B+C"]
    rows = []
    for n in names:
        m = {c: (P.actual - P[f"{n}|{c}"]).abs().mean() for c in ["seen", "control", "unseen"]}
        worst = (P.actual - P[f"{n}|unseen"]).abs().groupby([P.window_start, P.draw]).mean().max()
        rows.append(dict(model=n, MAE_seen=m["seen"], MAE_control=m["control"], MAE_unseen=m["unseen"],
                         control_minus_seen=m["control"] - m["seen"], unseen_minus_control=m["unseen"] - m["control"],
                         worst_window_draw_MAE_unseen=worst))
    res = pd.DataFrame(rows).sort_values("MAE_unseen").round(2)
    res.to_csv(f"{out}/city2_summary.csv", index=False)
    print(f"Evaluations: {len(P)} (window x draw x load) | unique loads: {P.load_id.nunique()}")
    print(res.to_string(index=False))


# ------------------------------------------------------------------ candidate F: daily market index (final round)
# Origin: proposed AFTER seeing development results (market_index experiment, round 3). Documented as such.
FS_B = frozenset({"coords", "nocity", "time"})
FS_BMI = frozenset({"coords", "nocity", "time", "dmi"})


def daily_index_table(batch):
    """date -> mean market_index over the rows of THIS batch (features only, never prices)."""
    return batch.groupby(batch.date.dt.normalize()).market_index.mean()


def add_daily_index(frame, daily, fill_value):
    """Attach the batch's daily mean; days without a valid value get the training-derived fill + a flag."""
    v = frame.date.dt.normalize().map(daily)
    out = frame.copy()
    out["mi_daily_missing"] = v.isna().astype(float).values
    out["mi_daily"] = v.fillna(fill_value).values
    return out


def fit_bmi(trf):
    """Fit B + daily index. Training daily means and the fill value come from the training part only."""
    d_tr = daily_index_table(trf)
    fill = float(d_tr.median())
    t = add_daily_index(trf, d_tr, fill)
    st = stats(t)
    X = feats(t, st, FS_BMI)
    m = lgb.LGBMRegressor(**LGB, objective="l1").fit(X, np.log(t.posted_rate.values), categorical_feature=["equipment"])
    return dict(model=m, st=st, fill=fill)


def bmi_features(fit, frame, daily):
    return feats(add_daily_index(frame, daily, fit["fill"]), fit["st"], FS_BMI)


def predict_bmi(fit, frame, daily):
    return np.clip(np.exp(fit["model"].predict(bmi_features(fit, frame, daily))), 1.0, None)


def coord_table(*frames):
    """city -> (lat, lon) from feature columns; fails loudly if a city ever has two coordinates."""
    long = pd.concat([pd.concat([f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
                                 f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)])
                      for f in frames]).drop_duplicates()
    assert not long.city.duplicated().any(), "a city has more than one coordinate"
    return long.set_index("city")


TEMPLATE_COLS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


def rebuild_from_template(tpl, coords):
    """December-file path: only template columns are given; coordinates come from the city lookup."""
    out = tpl[TEMPLATE_COLS].copy()
    for end in ("pickup", "delivery"):
        missing = set(out[end]) - set(coords.index)
        assert not missing, f"no coordinate for {missing}"
        out[f"{end}_lat"] = out[end].map(coords.lat).values
        out[f"{end}_lon"] = out[end].map(coords.lon).values
    return out


def part_roundF(dev, out):
    frames = []
    for st_ in WINDOW_STARTS:
        trf, vaf, h = window_split(dev, st_)
        f = pd.DataFrame({"load_id": vaf.load_id.values, "window_start": st_, "horizon_month": h,
                          "date": vaf.date.dt.date.astype(str).values, "actual": vaf.posted_rate.values})
        f["B_lgbm_coords_time"] = np.clip(BLEND_MODELS["B_lgbm_coords_time"](trf, vaf), 1.0, None)
        f["C_ridge_city"] = np.clip(BLEND_MODELS["C_ridge_city"](trf, vaf), 1.0, None)
        fit = fit_bmi(trf)
        f["Bmi_lgbm_coords_time_dailyindex"] = predict_bmi(fit, vaf, daily_index_table(vaf))
        frames.append(f); print(f"window {st_} done", flush=True)
    P = pd.concat(frames, ignore_index=True)
    P["blend_B+C"] = P[["B_lgbm_coords_time", "C_ridge_city"]].mean(axis=1)
    P["F_blend_Bmi+C"] = P[["Bmi_lgbm_coords_time_dailyindex", "C_ridge_city"]].mean(axis=1)
    P.to_csv(f"{out}/roundF_window_predictions.csv", index=False)
    cols = ["B_lgbm_coords_time", "C_ridge_city", "Bmi_lgbm_coords_time_dailyindex", "blend_B+C", "F_blend_Bmi+C"]
    summ = summarize_cols(P, cols); summ.to_csv(f"{out}/roundF_summary.csv", index=False)
    err = P.assign(**{c: (P.actual - P[c]).abs() for c in cols})
    per = err.groupby(["window_start", "horizon_month"])[cols].mean()
    per["F_minus_BC"] = per["F_blend_Bmi+C"] - per["blend_B+C"]
    per.round(2).to_csv(f"{out}/roundF_per_window.csv")
    win = err.groupby("window_start")[["blend_B+C", "F_blend_Bmi+C"]].mean()
    win["F_minus_BC"] = win["F_blend_Bmi+C"] - win["blend_B+C"]
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False)); print("\nPer window x horizon:\n", per.round(2).to_string())
    print("\nPer window (both months):\n", win.round(2).to_string())


def part_cityF(dev, out):
    """Same hidden-city draws / conditions as city2. B and C predictions are re-used from city2_parts
    (deterministic, same rows); only Bmi is fitted here. Daily means for evaluation come from the FULL
    evaluation batch of the window (as in deployment), training daily means from the rows actually trained on."""
    rng = np.random.default_rng(42)
    cities = np.array(sorted(set(dev.pickup) | set(dev.delivery)))
    draws = [sorted(rng.choice(cities, 6, replace=False)) for _ in range(4)]
    frames = []
    for st_ in WINDOW_STARTS:
        prev = pd.read_csv(f"{out}/city2_parts/{st_}.csv")
        trf, vaf, h = window_split(dev, st_)
        daily_eval = daily_index_table(vaf)
        seen_fit = fit_bmi(trf)
        for di, H in enumerate(draws):
            Hs = set(H)
            mv = (vaf.pickup.isin(Hs) | vaf.delivery.isin(Hs)).values
            touch = (trf.pickup.isin(Hs) | trf.delivery.isin(Hs)).values
            drop = np.random.default_rng(1000 + di).choice(np.flatnonzero(~touch), touch.sum(), replace=False)
            keep_ctrl = np.ones(len(trf), bool); keep_ctrl[drop] = False
            v = vaf[mv]
            f = prev[prev.draw == di].reset_index(drop=True)
            assert (f.load_id.values == v.load_id.values).all() and np.allclose(f.actual.values, v.posted_rate.values)
            f["Bmi|seen"] = predict_bmi(seen_fit, v, daily_eval)
            f["Bmi|control"] = predict_bmi(fit_bmi(trf[keep_ctrl]), v, daily_eval)
            f["Bmi|unseen"] = predict_bmi(fit_bmi(trf[~touch]), v, daily_eval)
            frames.append(f)
        print(f"window {st_} done", flush=True)
    P = pd.concat(frames, ignore_index=True)
    rows = []
    for cond in ["seen", "control", "unseen"]:
        P[f"blend_B+C|{cond}"] = P[[f"B_lgbm_coords_time|{cond}", f"C_ridge_city|{cond}"]].mean(axis=1)
        P[f"F_blend_Bmi+C|{cond}"] = P[[f"Bmi|{cond}", f"C_ridge_city|{cond}"]].mean(axis=1)
    P.to_csv(f"{out}/cityF_predictions.csv", index=False)
    for n in ["blend_B+C", "F_blend_Bmi+C", "B_lgbm_coords_time", "Bmi"]:
        m = {c: (P.actual - P[f"{n}|{c}"]).abs().mean() for c in ["seen", "control", "unseen"]}
        worst = (P.actual - P[f"{n}|unseen"]).abs().groupby([P.window_start, P.draw]).mean().max()
        rows.append(dict(model=n, MAE_seen=m["seen"], MAE_control=m["control"], MAE_unseen=m["unseen"],
                         control_minus_seen=m["control"] - m["seen"], unseen_minus_control=m["unseen"] - m["control"],
                         worst_window_draw_MAE_unseen=worst))
    res = pd.DataFrame(rows).round(2); res.to_csv(f"{out}/cityF_summary.csv", index=False)
    print(f"Evaluations: {len(P)} | unique loads: {P.load_id.nunique()}"); print(res.to_string(index=False))


def part_pathtest(dev, out, data_dir):
    """December-file path check: hide the columns absent from the December template, rebuild them via the
    city->coordinate table and the batch daily index, and require identical features and predictions."""
    _, va_file, dec = base.load(data_dir)
    rows = []
    for st_ in WINDOW_STARTS:
        trf, vaf, _ = window_split(dev, st_)
        fit, daily = fit_bmi(trf), daily_index_table(vaf)
        coords = coord_table(trf, vaf)
        rebuilt = rebuild_from_template(vaf, coords)
        X1, X2 = bmi_features(fit, vaf, daily), bmi_features(fit, rebuilt, daily)
        pd.testing.assert_frame_equal(X1, X2, check_dtype=True)
        p1, p2 = predict_bmi(fit, vaf, daily), predict_bmi(fit, rebuilt, daily)
        rows.append(dict(window=st_, rows=len(vaf), features_identical=True,
                         predictions_identical=bool(np.array_equal(p1, p2)), max_abs_diff=float(np.max(np.abs(p1 - p2)))))
    # the real December template: can it be prepared without gaps? (no model, no prices involved)
    coords_all = coord_table(dev, va_file)
    dec_rebuilt = rebuild_from_template(dec, coords_all)
    dec_daily = dec_rebuilt.date.dt.normalize().map(daily_index_table(va_file))
    res = pd.DataFrame(rows); res.to_csv(f"{out}/pathtest.csv", index=False)
    print(res.to_string(index=False))
    print(f"December template: coordinates found for {dec_rebuilt[['pickup_lat','delivery_lat']].notna().all(axis=1).sum()}/31 rows; "
          f"daily index available from validation features for {dec_daily.notna().sum()}/31 days")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="."); ap.add_argument("--out", default="results")
    ap.add_argument("--part", required=True, choices=["main", "city", "cat", "horizon", "blend",
                                                     "round3", "city2", "city2_summary", "roundF", "cityF", "pathtest"])
    ap.add_argument("--windows", nargs="*", default=WINDOW_STARTS, help="city2 only: which 2-month windows to run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr, _, _ = base.load(a.data)
    dev = tr[tr.date < base.HOLDOUT_START].reset_index(drop=True)
    if a.part == "city2":
        part_city2(dev, a.out, a.windows)
    elif a.part == "pathtest":
        part_pathtest(dev, a.out, a.data)
    else:
        {"main": part_main, "city": part_city, "cat": part_cat, "horizon": part_horizon, "blend": part_blend,
         "round3": part_round3, "city2_summary": part_city2_summary,
         "roundF": part_roundF, "cityF": part_cityF}[a.part](dev, a.out)
