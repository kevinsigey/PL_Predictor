"""Walk-forward test: does a fitted Dixon-Coles rho beat rho=0 for ml_scoreline?

Per fold, rho is fitted ONLY on the fold's training rows: expanding inner
folds (train on seasons < j, predict season j, for each training season j>=1)
give out-of-fold lambdas, and rho maximises their exact-score likelihood.
The evaluation fold never touches the fit. Run: python -m
pl_predictor.evaluate.fitted_rho_walk_forward [out.json]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..models import ml_scoreline
from .walk_forward import prepare_folds

MIN_LAMBDA = ml_scoreline.MIN_LAMBDA
CODE = {"H": 0, "D": 1, "A": 2}


def oof_lambdas(fold: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Out-of-fold (lam_home, lam_away) for the fold's own training rows."""
    tr, X = fold["train_df"], fold["X_train"]
    seasons = sorted(tr["season"].unique())
    lh, la, rows = [], [], []
    for j in range(1, len(seasons)):
        fit = tr["season"].isin(seasons[:j])
        pred = tr["season"] == seasons[j]
        hm, am = ml_scoreline.train_goal_regressors(X[fit], tr.loc[fit, "goals_home"], tr.loc[fit, "goals_away"])
        lh.append(np.maximum(hm.predict(X[pred]), MIN_LAMBDA))
        la.append(np.maximum(am.predict(X[pred]), MIN_LAMBDA))
        rows.append(tr[pred])
    return np.concatenate(lh), np.concatenate(la), pd.concat(rows)


def per_match(grids: list, val: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-match losses for every reported metric (lower is better)."""
    y = val["ftr"].map(CODE).to_numpy()
    gh, ga = val["goals_home"].to_numpy(int), val["goals_away"].to_numpy(int)
    p = np.array([[g.home_win, g.draw, g.away_win] for g in grids])
    ov = np.clip([g.total_goals("over", 2.5) for g in grids], 1e-6, 1 - 1e-6)
    bt = np.clip([g.btts_yes for g in grids], 1e-6, 1 - 1e-6)
    o, b = (gh + ga > 2.5), (gh > 0) & (ga > 0)
    return {
        "rps": np.array([pb.metrics.rps_average(r.reshape(1, -1), int(k)) for r, k in zip(p, y)]),
        "logloss_1x2": -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)),
        "ou25_logloss": -np.log(np.where(o, ov, 1 - ov)),
        "btts_logloss": -np.log(np.where(b, bt, 1 - bt)),
        "exact_logloss": -np.log([max(g.exact_score(h, a), 1e-6) for g, h, a in zip(grids, gh, ga)]),
        "p_draw": p[:, 1],
        "is_draw": (y == 1).astype(float),
    }


def paired_ci(diff: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Mean and percentile 95% CI of per-match (rho0 - fitted) loss; > 0 favours fitted."""
    idx = np.random.default_rng(seed).integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run() -> dict:
    base, fit, rhos = [], [], {}
    for fold in prepare_folds():
        lh, la, tr = oof_lambdas(fold)
        rho = ml_scoreline.fit_rho(lh, la, tr["goals_home"], tr["goals_away"])
        rhos[fold["val_season"]] = rho
        hm, am = ml_scoreline.train_goal_regressors(
            fold["X_train"], fold["train_df"]["goals_home"], fold["train_df"]["goals_away"]
        )
        for rows, r in ((base, 0.0), (fit, rho)):
            rows.append(per_match(ml_scoreline.predict_grids_batch(hm, am, fold["X_val"], rho=r), fold["val_df"]))
        print(fold["val_season"], f"rho={rho:+.4f}", f"n_oof={len(tr)}", flush=True)
    cat = lambda rs, k: np.concatenate([r[k] for r in rs])
    out = {"rho_by_fold": rhos, "n_matches": int(len(cat(base, "rps"))), "metrics": {}}
    for k in ("rps", "logloss_1x2", "ou25_logloss", "btts_logloss", "exact_logloss"):
        b, f = cat(base, k), cat(fit, k)
        d, lo, hi = paired_ci(b - f)
        out["metrics"][k] = {"rho0": float(b.mean()), "fitted": float(f.mean()), "gain": d, "ci95": [lo, hi]}
    out["draw_calibration"] = {
        "actual_draw_rate": float(cat(base, "is_draw").mean()),
        "mean_pred_draw_rho0": float(cat(base, "p_draw").mean()),
        "mean_pred_draw_fitted": float(cat(fit, "p_draw").mean()),
    }
    out["per_fold_rps"] = [
        {"val_season": s, "rho0": float(b["rps"].mean()), "fitted": float(f["rps"].mean())}
        for s, b, f in zip(rhos, base, fit)
    ]
    return out


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(res, indent=2) + "\n")
