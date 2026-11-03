"""Pre-registered test: does a player's history against a specific opponent
predict scoring / goal involvement beyond his overall rate and the opponent's
overall weakness?

Procedure, seasons and thresholds are fixed in
``docs/player_opponent_effect_preregistration.md``. If this file and that
document disagree, the document wins.

Nothing here changes a production model or a serving path. ``run_study``
writes the raw JSON record and nothing else.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import fpl_history
from ..data.team_names import to_canonical
from ..features import player_form
from .player_ratio_search import (
    PRODUCTION_BASE,
    _design,
    _winsorise,
    binary_log_loss,
    expected_calibration_error,
)

# ---- pre-registered constants ----------------------------------------------
ALL_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
HISTORY_ONLY = ["2020-21", "2021-22"]  # no xG; prior history only, never rows
ROW_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]
TEST_FOLDS = ["2023-24", "2024-25", "2025-26"]
M0 = 10.0  # opponent shrinkage, in team-matches
A_PSEUDO_MIN = 900.0  # player-overall shrinkage, in minutes
TIER_CUTS = (0.9, 1.1)
MIN_TIER_MATCHES = 10
KAPPA_GRID = (90.0, 270.0, 540.0, 1080.0, 2160.0, 4320.0)  # minutes
ALPHA_FAMILY = 0.05
K_FAMILY = 4
CI_LEVEL = 1 - ALPHA_FAMILY / K_FAMILY
N_BOOT = 2000
SEED = 20261031
RATIO_EPS = 1e-9

COMPARISONS = {
    # id: (feature family, target)
    "A_scorer": ("A", "scorer"),
    "B_scorer": ("B", "scorer"),
    "A_involvement": ("A", "involvement"),
    "B_involvement": ("B", "involvement"),
}
_TARGET_COL = {"scorer": "scorer_y", "involvement": "inv_y"}
_HIST_NUM = {"scorer": "goals_scored", "involvement": "goal_inv"}


# ---- helpers -----------------------------------------------------------------

def _lagged_by_kickoff(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Sum of value_cols over rows with STRICTLY earlier kickoff, per row.
    Ties at the same kickoff time do not see each other."""
    agg = frame.groupby("kickoff_time")[value_cols].sum().sort_index()
    shifted = agg.cumsum().shift(1).fillna(0.0)
    return shifted.reindex(frame["kickoff_time"]).set_axis(frame.index)


def build_frame(seasons: list[str] = ALL_SEASONS) -> pd.DataFrame:
    """One row per player appearance (minutes > 0), with strictly lagged
    history features. Rows from history-only seasons are kept (they feed the
    lags) and flagged by ``is_row``."""
    raw = fpl_history.load_player_gw_history(seasons=seasons).copy()
    raw["tc"] = raw["team"].map(lambda n: to_canonical(str(n), "fpl"))
    raw = raw.sort_values(["season", "element", "kickoff_time"]).reset_index(drop=True)

    # Fixture table: one row per (season, fixture) with both sides.
    side = raw.drop_duplicates(["season", "fixture", "tc"])
    home = side[side["was_home"].astype(bool)][["season", "fixture", "tc", "team_h_score", "team_a_score", "kickoff_time"]]
    home = home.rename(columns={"tc": "home"})
    away = side[~side["was_home"].astype(bool)][["season", "fixture", "tc"]].rename(columns={"tc": "away"})
    fixtures = home.merge(away, on=["season", "fixture"], how="inner").drop_duplicates(["season", "fixture"])
    fixtures = fixtures.dropna(subset=["team_h_score", "team_a_score"]).reset_index(drop=True)
    fixtures["total_goals"] = fixtures["team_h_score"] + fixtures["team_a_score"]

    # League goals per team-match, strictly earlier kickoffs.
    league = _lagged_by_kickoff(fixtures.assign(n=1.0), ["total_goals", "n"])
    fixtures["L"] = (league["total_goals"] / (2.0 * league["n"])).where(league["n"] > 0)

    # Team-match table with prior (strictly earlier) goals conceded and count.
    tm = pd.concat([
        pd.DataFrame({"season": fixtures["season"], "fixture": fixtures["fixture"], "team": fixtures["home"],
                      "kickoff_time": fixtures["kickoff_time"], "ga": fixtures["team_a_score"], "L": fixtures["L"]}),
        pd.DataFrame({"season": fixtures["season"], "fixture": fixtures["fixture"], "team": fixtures["away"],
                      "kickoff_time": fixtures["kickoff_time"], "ga": fixtures["team_h_score"], "L": fixtures["L"]}),
    ], ignore_index=True).sort_values(["team", "kickoff_time"]).reset_index(drop=True)
    tm["n_prior"] = tm.groupby("team").cumcount().astype(float)
    tm["ga_prior"] = tm.groupby("team")["ga"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    tm["f_opp"] = ((tm["ga_prior"] + M0 * tm["L"]) / (tm["n_prior"] + M0)) / tm["L"]
    tm["tier"] = np.where(tm["n_prior"] < MIN_TIER_MATCHES, np.nan,
                          np.where(tm["f_opp"] < TIER_CUTS[0], 0.0,
                                   np.where(tm["f_opp"] <= TIER_CUTS[1], 1.0, 2.0)))
    tm_key = tm[["season", "fixture", "team", "f_opp", "tier"]]

    # Appearance rows.
    app = raw[raw["minutes"] > 0].copy()
    app = app.merge(fixtures[["season", "fixture", "home", "away", "L"]], on=["season", "fixture"], how="inner")
    is_home = app["was_home"].astype(bool).to_numpy()
    app["team"] = np.where(is_home, app["home"], app["away"])
    app["opp"] = np.where(is_home, app["away"], app["home"])
    app = app.drop(columns=["home", "away"])
    app = app.merge(tm_key.rename(columns={"team": "opp"}), on=["season", "fixture", "opp"], how="left")
    app = app.sort_values(["kickoff_time", "element"]).reset_index(drop=True)
    app["is_row"] = app["season"].isin(ROW_SEASONS)

    app["goal_inv"] = app["goals_scored"].astype(float) + app["assists"].astype(float)
    app["scorer_y"] = (app["goals_scored"] >= 1).astype(float)
    app["inv_y"] = (app["goal_inv"] >= 1).astype(float)
    app["log_f_opp"] = np.log(app["f_opp"])

    # League goals per player-minute, strictly earlier kickoffs (for r0).
    lg = _lagged_by_kickoff(app, ["goals_scored", "minutes"])
    app["r0"] = (lg["goals_scored"] / lg["minutes"]).where(lg["minutes"] > 0)

    # Player overall prior (strictly earlier appearances).
    g = app.groupby("element")
    app["G_prior"] = g["goals_scored"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    app["GI_prior"] = g["goal_inv"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    app["M_prior"] = g["minutes"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    app["pi_player_scorer"] = (app["G_prior"] + A_PSEUDO_MIN * app["r0"]) / (app["M_prior"] + A_PSEUDO_MIN)
    app["pi_player_inv"] = (app["GI_prior"] + A_PSEUDO_MIN * app["r0"]) / (app["M_prior"] + A_PSEUDO_MIN)
    app["pi_pair_scorer"] = app["pi_player_scorer"] * app["f_opp"]
    app["pi_pair_inv"] = app["pi_player_inv"] * app["f_opp"]

    # Player vs the same opponent (A): strictly earlier appearances vs that team.
    gv = app.groupby(["element", "opp"])
    app["GV_goals"] = gv["goals_scored"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    app["GV_inv"] = gv["goal_inv"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    app["MV"] = gv["minutes"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())

    # Player vs opponents of the same tier label (B): tier as of each appearance.
    tiered = app["tier"].notna()
    gt = app[tiered].groupby(["element", "tier"])
    app["GC_goals"] = np.nan
    app["GC_inv"] = np.nan
    app["MC"] = np.nan
    app.loc[tiered, "GC_goals"] = gt["goals_scored"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    app.loc[tiered, "GC_inv"] = gt["goal_inv"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())
    app.loc[tiered, "MC"] = gt["minutes"].transform(lambda s: s.shift(1).fillna(0.0).cumsum())

    # Production features, built by the production builders (same as ratio search).
    starts, start_cols = player_form.build_historical_start_features(app)
    played, _ = player_form.build_historical_player_form(app)
    frame = played.join(starts[start_cols], how="left").copy()
    frame["was_home"] = frame["was_home"].astype(float)
    frame = frame.sort_values(["kickoff_time", "element"]).reset_index(drop=True)
    # Carry the lag and identifier columns through the production join.
    frame = frame.merge(
        app[["element", "kickoff_time", "fixture", "season", "opp", "team", "is_row", "f_opp", "tier",
             "scorer_y", "inv_y", "goal_inv", "goals_scored", "r0", "pi_pair_scorer", "pi_pair_inv",
             "GV_goals", "GV_inv", "MV", "GC_goals", "GC_inv", "MC", "log_f_opp"]],
        on=["element", "kickoff_time", "fixture", "season"], how="left", suffixes=("", "_app"),
    )
    return frame


def candidate_feature(frame: pd.DataFrame, family: str, target: str, kappa: float) -> np.ndarray:
    """z = log(r / pi_pair). Zero when there is no matchup history (r == pi_pair)."""
    pi = frame[f"pi_pair_{'scorer' if target == 'scorer' else 'inv'}"].to_numpy(dtype=float)
    if family == "A":
        g = frame["GV_goals" if target == "scorer" else "GV_inv"].to_numpy(dtype=float)
        m = frame["MV"].to_numpy(dtype=float)
    else:
        g = frame["GC_goals" if target == "scorer" else "GC_inv"].to_numpy(dtype=float)
        m = frame["MC"].to_numpy(dtype=float)
    g = np.nan_to_num(g, nan=0.0)
    m = np.nan_to_num(m, nan=0.0)
    r = (g + kappa * pi) / (m + kappa)
    z = np.log(np.clip(r, RATIO_EPS, None) / np.clip(pi, RATIO_EPS, None))
    z = np.where(np.isfinite(z), z, 0.0)
    if family == "B":  # unknown tier: no cluster feature
        z = np.where(frame["tier"].notna().to_numpy(), z, 0.0)
    return z


def _fit_kappa(frame: pd.DataFrame, family: str, target: str, train_mask: np.ndarray) -> float:
    """Training-rows-only choice of kappa: binary log loss of p = 1 - exp(-90 r)."""
    y = frame.loc[train_mask, _TARGET_COL[target]].to_numpy(dtype=float)
    pi = frame.loc[train_mask, f"pi_pair_{'scorer' if target == 'scorer' else 'inv'}"].to_numpy(dtype=float)
    best, best_loss = None, math.inf
    for kappa in KAPPA_GRID:
        z = candidate_feature(frame.loc[train_mask], family, target, kappa)
        r = pi * np.exp(z)
        p = 1.0 - np.exp(-90.0 * r)
        loss = float(binary_log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)).mean())
        if loss < best_loss:
            best, best_loss = kappa, loss
    return float(best)


def _fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    """Unweighted logistic regression (amendment 1 of the prereg)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    return np.clip(model.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)


def _brier(y, p):
    return (p - y) ** 2


def _fold_predictions(frame: pd.DataFrame, target: str, family: str | None, fold: str) -> dict:
    """Control arm (P + log f_o) and, if family given, the candidate arm
    (control + z). Returns per-row predictions for the test season."""
    train_seasons = [s for s in ROW_SEASONS if s < fold]
    tr = frame["season"].isin(train_seasons).to_numpy() & frame["is_row"].to_numpy()
    te = (frame["season"] == fold).to_numpy()
    y_tr = frame.loc[tr, _TARGET_COL[target]].to_numpy(dtype=float)
    y_te = frame.loc[te, _TARGET_COL[target]].to_numpy(dtype=float)

    base = list(PRODUCTION_BASE)
    X_ctrl = _design(frame, base + ["log_f_opp"])
    X_ctrl_tr, X_ctrl_te = _winsorise(X_ctrl[tr], X_ctrl[te], 0.005, 0.995)
    p_ctrl = _fit_predict(X_ctrl_tr, y_tr, X_ctrl_te)

    X_prod = _design(frame, base)
    X_prod_tr, X_prod_te = _winsorise(X_prod[tr], X_prod[te], 0.005, 0.995)
    p_prod = _fit_predict(X_prod_tr, y_tr, X_prod_te)

    out = {"y": y_te, "p_ctrl": p_ctrl, "p_prod": p_prod, "kappa": None}
    if family is None:
        return out
    kappa = _fit_kappa(frame, family, target, tr)
    z_all = candidate_feature(frame, family, target, kappa)
    frame_z = frame.assign(_z=z_all)
    X_cand = _design(frame_z, base + ["log_f_opp", "_z"])
    X_cand_tr, X_cand_te = _winsorise(X_cand[tr], X_cand[te], 0.005, 0.995)
    out["p_cand"] = _fit_predict(X_cand_tr, y_tr, X_cand_te)
    out["kappa"] = kappa
    out["z_te"] = z_all[te]
    out["n_train"] = int(tr.sum())
    out["n_test"] = int(te.sum())
    return out


def _cluster_bootstrap(d: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """Mean of d, and percentile CI at CI_LEVEL, resampling whole fixtures."""
    codes, inv = np.unique(clusters, return_inverse=True)
    C = len(codes)
    sums = np.bincount(inv, weights=d, minlength=C)
    counts = np.bincount(inv, minlength=C).astype(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, C, size=(N_BOOT, C))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo_q = (1 - CI_LEVEL) / 2
    lo, hi = np.quantile(means, [lo_q, 1 - lo_q])
    return float(d.mean()), float(lo), float(hi)


def evaluate_comparison(frame: pd.DataFrame, name: str) -> dict:
    family, target = COMPARISONS[name]
    folds = {f: _fold_predictions(frame, target, family, f) for f in TEST_FOLDS}
    y = np.concatenate([folds[f]["y"] for f in TEST_FOLDS])
    p_c = np.concatenate([folds[f]["p_ctrl"] for f in TEST_FOLDS])
    p_k = np.concatenate([folds[f]["p_cand"] for f in TEST_FOLDS])
    p_p = np.concatenate([folds[f]["p_prod"] for f in TEST_FOLDS])
    # Row order in predictions matches the test masks, folds in TEST_FOLDS order.
    clusters = np.concatenate([
        np.array([f"{s}|{x}" for s, x in frame.loc[(frame["season"] == f) & frame["is_row"], ["season", "fixture"]].to_numpy()])
        for f in TEST_FOLDS])

    ll_c = binary_log_loss(y, p_c)
    ll_k = binary_log_loss(y, p_k)
    d = ll_k - ll_c
    mean_d, lo, hi = _cluster_bootstrap(d, clusters)
    fold_means = {f: float((binary_log_loss(folds[f]["y"], folds[f]["p_cand"]) -
                            binary_log_loss(folds[f]["y"], folds[f]["p_ctrl"])).mean()) for f in TEST_FOLDS}
    brier_d = float((_brier(y, p_k) - _brier(y, p_c)).mean())
    ece_d = float(expected_calibration_error(y, p_k) - expected_calibration_error(y, p_c))
    prod_d = float((ll_k - binary_log_loss(y, p_p)).mean())
    ctrl_vs_prod = float((ll_c - binary_log_loss(y, p_p)).mean())
    pass_ci = hi < 0
    pass_sign = all(v < 0 for v in fold_means.values())
    pass_others = brier_d <= 0 and ece_d <= 0
    z_te = np.concatenate([folds[f]["z_te"] for f in TEST_FOLDS])
    return {
        "family": family, "target": target,
        "kappa_by_fold": {f: folds[f]["kappa"] for f in TEST_FOLDS},
        "n_test_rows": int(len(y)),
        "n_test_fixtures": int(len(np.unique(clusters))),
        "rows_with_matchup_signal": int((z_te != 0).sum()),
        "mean_logloss_diff_vs_control": mean_d,
        "ci_level": CI_LEVEL,
        "ci_lo": lo, "ci_hi": hi,
        "fold_mean_logloss_diff": fold_means,
        "brier_diff_vs_control": brier_d,
        "ece_diff_vs_control": ece_d,
        "descriptive_logloss_diff_vs_plain_production": prod_d,
        "descriptive_control_logloss_diff_vs_plain_production": ctrl_vs_prod,
        "descriptive_z_std_test": float(np.std(z_te)),
        "descriptive_z_spearman_with_outcome_test": float(pd.Series(z_te).corr(pd.Series(y), method="spearman")),
        "pass_ci_excludes_zero": bool(pass_ci),
        "pass_sign_stable_every_fold": bool(pass_sign),
        "pass_nothing_else_worse": bool(pass_others),
        "PASS": bool(pass_ci and pass_sign and pass_others),
    }


def counts_report(frame: pd.DataFrame) -> dict:
    """Prior-meeting counts for test rows (not used for the pass decision)."""
    te = frame["season"].isin(TEST_FOLDS) & frame["is_row"]
    sub = frame[te]
    # Prior meetings = earlier appearances (any season) vs the same opponent.
    prior_n = frame.groupby(["element", "opp"]).cumcount().to_numpy()
    te_arr = te.to_numpy()
    pairs_in_test = frame.loc[te_arr, ["element", "opp"]].drop_duplicates()
    ge3 = te_arr & (prior_n >= 3)
    return {
        "test_rows": int(te_arr.sum()),
        "test_rows_with_any_prior_meeting": int((te_arr & (prior_n >= 1)).sum()),
        "test_rows_with_ge3_prior_meetings": int(ge3.sum()),
        "distinct_player_opponent_pairs_in_test": int(len(pairs_in_test)),
        "distinct_player_opponent_pairs_with_ge3_prior_meetings": int(
            frame.loc[ge3, ["element", "opp"]].drop_duplicates().shape[0]),
    }


def run_study(output_path: Path) -> dict:
    frame = build_frame()
    results = {name: evaluate_comparison(frame, name) for name in COMPARISONS}
    report = {
        "preregistration": "docs/player_opponent_effect_preregistration.md",
        "seasons": {"history_only": HISTORY_ONLY, "row_seasons": ROW_SEASONS, "test_folds": TEST_FOLDS},
        "constants": {"M0": M0, "A_PSEUDO_MIN": A_PSEUDO_MIN, "TIER_CUTS": list(TIER_CUTS),
                      "MIN_TIER_MATCHES": MIN_TIER_MATCHES, "KAPPA_GRID": list(KAPPA_GRID),
                      "K_FAMILY": K_FAMILY, "CI_LEVEL": CI_LEVEL, "N_BOOT": N_BOOT, "SEED": SEED},
        "counts": counts_report(frame),
        "comparisons": results,
        "any_pass": any(r["PASS"] for r in results.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=float))
    return report
