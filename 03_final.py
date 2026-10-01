"""
Final, FROZEN pipeline for model F.

  python 03_final.py --data <dir> --out results --step checks    # implementation checks (no October prices)
  python 03_final.py --data <dir> --out results --step october   # ONE-TIME final test: train Jan-Sep, score October
  python 03_final.py --data <dir> --out .       --step predict   # train on all 48k, write the two output files

Model F (frozen; selected on Jan-Sep development windows BEFORE October was scored):
  F = 0.5 * Bmi + 0.5 * C
  Bmi : LightGBM, objective L1 on log(rate); features = distance, log distance, |weight| (equipment-median fill
        + missing flag), equipment, day of week, pickup/delivery coordinates (no city names), days since 2025-01-01,
        daily mean market_index (+ missing flag). Parameters: LGB dict in 02_tree_experiments.py.
  C   : Ridge (alpha=1) on log(rate) with distance terms, equipment, day of week, weight, pickup/delivery one-hots,
        smearing correction. No time feature.
  Selection rule, weights (50/50) and parameters are frozen. October is reported for F and for the reference B+C;
  the choice does not change with the October result. Genuine bugs, if found, are fixed and documented.

Documented assumptions
  * The daily market_index mean is computed from the FEATURES of the batch being priced (no prices). F therefore
    uses indicators attached to the loads it prices; it is not a forecast from data up to October alone.
  * For the fixed December chart load, the daily mean of the December loads in validation.csv is attached by date.
    This assumes the index is shared by loads on the same day (supported by the data: date explains ~98% of its
    variance in Jan-Sep), not a proven business meaning.
  * Coordinates for the December template come from a city -> coordinate table built from the feature files
    (each city has exactly one coordinate in the data; asserted).
  * The idea of using market_index came from development results (round 3) and is documented as such.
  * The October test measures a ONE-month horizon only; Nov-Dec is up to two months ahead.
"""
import argparse, importlib.util, os, sys
import numpy as np, pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("exp", os.path.join(_here, "02_tree_experiments.py"))
exp = importlib.util.module_from_spec(_s); _s.loader.exec_module(exp)
base = exp.base


# ------------------------------------------------------------------ frozen model F
def fit_F(train):
    return dict(bmi=exp.fit_bmi(train), train=train)


def predict_F(fitted, frame, daily):
    """Returns (F, Bmi, C) predictions in dollars. Identical arithmetic to part_roundF."""
    p_bmi = exp.predict_bmi(fitted["bmi"], frame, daily)
    p_c = np.clip(exp.ridge_city(fitted["train"], frame, False)["ridge"], 1.0, None)
    return (p_bmi + p_c) / 2.0, p_bmi, p_c


def predict_BC(train, frame):
    """Reference B+C (context only)."""
    p_b = np.clip(exp.BLEND_MODELS["B_lgbm_coords_time"](train, frame), 1.0, None)
    p_c = np.clip(exp.BLEND_MODELS["C_ridge_city"](train, frame), 1.0, None)
    return (p_b + p_c) / 2.0


# ------------------------------------------------------------------ checks (no October prices touched)
def step_checks(data, out):
    tr, va, dec = base.load(data)
    dev = tr[tr.date < base.HOLDOUT_START].reset_index(drop=True)
    log = []

    # 1) frozen implementation reproduces the evaluated F predictions
    saved = pd.read_csv(os.path.join(out, "roundF_window_predictions.csv"))
    for st in ["2025-07", "2025-08"]:
        trf, vaf, _ = exp.window_split(dev, st)
        f, _, _ = predict_F(fit_F(trf), vaf, exp.daily_index_table(vaf))
        ref = saved[saved.window_start == st]
        assert (ref.load_id.values == vaf.load_id.values).all()
        d = float(np.max(np.abs(ref["F_blend_Bmi+C"].values - f)))
        assert d < 1e-6, d
        log.append(f"[ok] frozen F reproduces saved development predictions, window {st} (max diff {d:.2e})")

    # 2) December path for the FULL F (Bmi + Ridge): hide non-template columns, rebuild, compare
    for st in exp.WINDOW_STARTS:
        trf, vaf, _ = exp.window_split(dev, st)
        fitted, daily = fit_F(trf), exp.daily_index_table(vaf)
        rebuilt = exp.rebuild_from_template(vaf, exp.coord_table(trf, vaf))
        a, b = predict_F(fitted, vaf, daily), predict_F(fitted, rebuilt, daily)
        for name, x, y in zip(["F", "Bmi", "C"], a, b):
            assert np.array_equal(x, y), (st, name, np.max(np.abs(x - y)))
        log.append(f"[ok] December path: full F, Bmi and C identical to the normal path, window {st} ({len(vaf)} rows)")

    # 3) fill rule for a day without a valid index (synthetic case)
    trf, vaf, _ = exp.window_split(dev, "2025-08")
    tr2, va2 = trf.copy(), vaf.copy()
    d_tr = sorted(tr2.date.dt.normalize().unique())[5]
    d_va = sorted(va2.date.dt.normalize().unique())[3]
    tr2.loc[tr2.date.dt.normalize() == d_tr, "market_index"] = np.nan
    va2.loc[va2.date.dt.normalize() == d_va, "market_index"] = np.nan
    bm = exp.fit_bmi(tr2)
    expected_fill = float(exp.daily_index_table(tr2).median())  # NaN day skipped by median
    assert bm["fill"] == expected_fill
    Xtr = exp.add_daily_index(tr2, exp.daily_index_table(tr2), bm["fill"])
    m_tr = (tr2.date.dt.normalize() == d_tr).values
    assert Xtr.mi_daily_missing.values[m_tr].all() and (Xtr.mi_daily.values[m_tr] == bm["fill"]).all()
    assert not Xtr.mi_daily_missing.values[~m_tr].any()
    daily_va = exp.daily_index_table(va2)
    X = exp.bmi_features(bm, va2, daily_va)
    m_va = (va2.date.dt.normalize() == d_va).values
    assert X.mi_daily_missing.values[m_va].all() and (X.mi_daily.values[m_va] == bm["fill"]).all()
    assert not X.mi_daily_missing.values[~m_va].any()
    # a date entirely absent from the daily table (e.g. a chart date with no batch rows)
    odd = va2.iloc[:3].copy(); odd["date"] = pd.Timestamp("2026-01-15")
    Xo = exp.bmi_features(bm, odd, daily_va)
    assert Xo.mi_daily_missing.eq(1).all() and Xo.mi_daily.eq(bm["fill"]).all()
    p = exp.predict_bmi(bm, va2, daily_va)
    assert np.isfinite(p).all() and (p > 0).all()
    log.append(f"[ok] fill rule: training day {d_tr.date()} and evaluation day {d_va.date()} without index -> "
               f"flag=1 and fill={bm['fill']:.5f} (median of training daily means); other days flag=0; "
               f"absent date handled; predictions finite and positive")

    # 4) real December template can be prepared without gaps
    rebuilt = exp.rebuild_from_template(dec, exp.coord_table(tr, va))
    days = rebuilt.date.dt.normalize().map(exp.daily_index_table(va))
    assert rebuilt[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all()
    assert days.notna().all()
    log.append("[ok] December template: coordinates for 31/31 rows, daily index for 31/31 days from validation features")

    txt = "\n".join(log); print(txt)
    open(os.path.join(out, "final_checks.txt"), "w").write(txt + "\n")


# ------------------------------------------------------------------ one-time October test
def step_october(data, out):
    tr, _, _ = base.load(data)
    train = tr[tr.date < base.HOLDOUT_START].reset_index(drop=True)
    octo = tr[tr.date >= base.HOLDOUT_START].reset_index(drop=True)
    f, p_bmi, p_c = predict_F(fit_F(train), octo, exp.daily_index_table(octo))
    bc = predict_BC(train, octo)
    P = pd.DataFrame({"load_id": octo.load_id, "date": octo.date.dt.date.astype(str), "actual": octo.posted_rate,
                      "F": f, "F_part_Bmi": p_bmi, "F_part_C": p_c, "reference_B+C": bc})
    P.to_csv(os.path.join(out, "october_predictions.csv"), index=False)
    rows = []
    for name in ["F", "reference_B+C"]:
        e = (P.actual - P[name]).abs()
        rows.append(dict(model=name, rows=len(P), MAE=e.mean(), RMSE=np.sqrt(((P.actual - P[name]) ** 2).mean()),
                         MedAE=e.median()))
    res = pd.DataFrame(rows).round(2)
    res.to_csv(os.path.join(out, "october_test.csv"), index=False)
    print("ONE-TIME October test (train Jan-Sep, one-month horizon). Selection is frozen: F.")
    print(res.to_string(index=False))


# ------------------------------------------------------------------ final training and output files
def step_predict(data, out):
    tr, va, _ = base.load(data)
    fitted = fit_F(tr)  # all 48,000 labelled loads (Jan-Oct)
    daily_va = exp.daily_index_table(va)

    # 12,000 validation loads, in the template's order
    tpl = pd.read_csv(base.find_file(data, "validation_predictions_template"))
    f, _, _ = predict_F(fitted, va, daily_va)
    pred = pd.Series(f, index=va.load_id.values)
    sub = pd.DataFrame({"load_id": tpl.load_id, "predicted_rate": pred.reindex(tpl.load_id).values.round(2)})
    assert len(sub) == 12000 and sub.predicted_rate.notna().all() and (sub.predicted_rate > 0).all()
    sub.to_csv(os.path.join(out, "validation_predictions.csv"), index=False)

    # December chart file: keep the original seven columns, order and date strings; fill predicted_rate only
    raw = pd.read_csv(base.find_file(data, "december_chart_inputs"))
    frame = raw.copy(); frame["date"] = pd.to_datetime(frame["date"])
    rebuilt = exp.rebuild_from_template(frame, exp.coord_table(tr, va))
    fd, _, _ = predict_F(fitted, rebuilt, daily_va)
    raw["predicted_rate"] = np.round(fd, 2)
    raw.to_csv(os.path.join(out, "december_chart_inputs.csv"), index=False)
    print(f"Wrote validation_predictions.csv ({len(sub)} rows) and december_chart_inputs.csv ({len(raw)} rows)")
    print(raw[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--step", required=True, choices=["checks", "october", "predict"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"checks": step_checks, "october": step_october, "predict": step_predict}[a.step](a.data, a.out)
