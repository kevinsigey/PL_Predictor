"""bottomup_vs_dixoncoles.py — research harness (NOT wired into serving).

Pre-registered in docs/bottomup_vs_dixoncoles_preregistration.md. Compares
three arms on the same matches, walk-forward by match date:

  DC  current Dixon-Coles (scoreline.fit_dixon_coles + predict_fixture)
  BU  bottom-up: team lambda = sum_p rate_p * E[min_p]/90, times opponent
      concession and home factors, fed into the same DC scoreline grid
      (penaltyblog create_dixon_coles_grid with the refitted rho)
  BL  probability blend w*BU + (1-w)*DC, w fitted on earlier folds only

Two lineup assumptions, never pooled: (i) actual XI, (ii) predicted from
prior appearances only. Run:  python -m pl_predictor.evaluate.bottomup_vs_dixoncoles
"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import penaltyblog as pb

from ..config import PROJECT_ROOT
from ..data.football_data import fetch_season
from ..data.fpl_history import fetch_season_gw_data
from ..models.scoreline import fit_dixon_coles, predict_fixture

FD_SEASONS = [f"{y}-{y + 1}" for y in range(2014, 2026)]  # football-data, 2014-15..2025-26
FPL_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]  # FPL, 2016-17..2025-26
FIT_SEASON = "2020-2021"  # fit-only fold (blend weights for 2021-22)
TEST_SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
ALL_SEASONS = [FIT_SEASON] + TEST_SEASONS

K_RATE = 5.0  # shrinkage, 90-minute equivalents (pre-registered)
K_OPP = 10.0  # opponent concession shrinkage, matches (pre-registered)
LAST_N = 5  # team fixtures used for predicted lineup (pre-registered)
SUB_MIN_FALLBACK = 20.0
DEFAULT_POS_RATE = 0.1  # goals per 90 if a position has no prior data
MAX_GOALS = 10
N_BOOT = 2000
SEED = 20261031
BLEND_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)
BONFERRONI_FAMILY = 4  # {BU, BL} x {(i), (ii)}
LO_Q = 0.5 * 0.05 / BONFERRONI_FAMILY  # 98.75% two-sided, percentile form
HI_Q = 1.0 - LO_Q

# FPL (vaastav) team names that differ from football-data canonical names.
FPL_TO_FD = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Sheffield Utd": "Sheffield United",
    "Ipswich Town": "Ipswich",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
}
LINEUPS = ("i", "ii")
# POST-REGISTRATION sensitivity arm (see findings "Deviations"): actual XI with
# expected minutes rescaled to 11 x 90 per team. Reported separately, never
# part of the pre-registered ship rule.
SENSITIVITY_LINEUP = "i_norm"
NOMINAL_TEAM_MINUTES = 990.0
ARMS = ("DC", "BU", "BL")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_football_data() -> pd.DataFrame:
    frames = [fetch_season(s) for s in FD_SEASONS]
    fd = pd.concat(frames, ignore_index=True)
    fd["date"] = pd.to_datetime(fd["date"])
    fd["goals_home"] = fd["goals_home"].astype(float)
    fd["goals_away"] = fd["goals_away"].astype(float)
    return fd.sort_values("datetime").reset_index(drop=True)


def load_fpl_rows() -> pd.DataFrame:
    frames = []
    for season in FPL_SEASONS:
        df = fetch_season_gw_data(season)
        if df is None or df.empty:
            continue
        start = int(season[:4])
        # 2016-17..2019-20 carry no `team` column at all (see the pre-registration).
        team = df["team"] if "team" in df else pd.Series(np.nan, index=df.index, dtype=object)
        position = df["position"] if "position" in df else pd.Series("UNK", index=df.index, dtype=object)
        starts = df["starts"] if "starts" in df else pd.Series(0.0, index=df.index)
        out = pd.DataFrame(
            {
                "name": df["name"],
                "position": position,
                "team": team.map(lambda t: FPL_TO_FD.get(t, t) if isinstance(t, str) else np.nan),
                "fk": f"{season}|" + df["fixture"].astype(str),
                "season_fd": f"{start}-{start + 1}",
                "was_home": df["was_home"].astype(bool),
                "kickoff": pd.to_datetime(df["kickoff_time"], utc=True).dt.tz_localize(None),
                "minutes": df["minutes"].fillna(0).astype(float),
                "starts": starts.fillna(0).astype(float),
                "goals": df["goals_scored"].fillna(0).astype(float),
            }
        )
        frames.append(out)
    rows = pd.concat(frames, ignore_index=True)
    rows["position"] = rows["position"].fillna("UNK")
    return rows.dropna(subset=["name", "kickoff"]).sort_values("kickoff").reset_index(drop=True)


def build_fixtures(rows: pd.DataFrame, fd: pd.DataFrame) -> pd.DataFrame:
    """One row per FPL fixture with a team name (2020-21 onward), joined to
    its football-data row on (season, home, away)."""
    teamed = rows.dropna(subset=["team"])
    home = teamed[teamed.was_home].groupby("fk").agg(home=("team", "first"), kickoff=("kickoff", "min"), season_fd=("season_fd", "first"))
    away = teamed[~teamed.was_home].groupby("fk").agg(away=("team", "first"))
    fx = home.join(away, how="inner").reset_index()
    fx = fx.merge(
        fd[["season", "team_home", "team_away", "date", "datetime", "goals_home", "goals_away"]],
        left_on=["season_fd", "home", "away"],
        right_on=["season", "team_home", "team_away"],
        how="left",
    )
    # The football-data `season` column duplicates ours; keep one, named `season`.
    return fx.drop(columns=["season"]).rename(columns={"season_fd": "season"})


# --------------------------------------------------------------------------
# Player and team indexes (strictly-prior lookups)
# --------------------------------------------------------------------------


class PlayerIndex:
    """Cumulative per-player and per-position sums, queried with searchsorted
    so that only rows with kickoff strictly before the query time count."""

    def __init__(self, rows: pd.DataFrame):
        self.player = {}
        for name, g in rows.groupby("name", sort=False):
            g = g.sort_values("kickoff")
            t = g["kickoff"].to_numpy("datetime64[ns]").astype(np.int64)
            m90 = g["minutes"].to_numpy() / 90.0
            sub_mask = (g["starts"].to_numpy() == 0) & (g["minutes"].to_numpy() > 0)
            self.player[name] = {
                "t": t,
                "cumG": np.concatenate([[0.0], np.cumsum(g["goals"].to_numpy())]),
                "cumM": np.concatenate([[0.0], np.cumsum(m90)]),
                "cumSubMin": np.concatenate([[0.0], np.cumsum(np.where(sub_mask, g["minutes"].to_numpy(), 0.0))]),
                "cumSubN": np.concatenate([[0.0], np.cumsum(sub_mask.astype(float))]),
                "pos": g["position"].to_numpy(dtype=object),
            }
        self.position = {}
        for pos, g in rows.groupby("position", sort=False):
            t = g["kickoff"].to_numpy("datetime64[ns]").astype(np.int64)
            self.position[pos] = {
                "t": t,
                "cumG": np.concatenate([[0.0], np.cumsum(g["goals"].to_numpy())]),
                "cumM": np.concatenate([[0.0], np.cumsum(g["minutes"].to_numpy() / 90.0)]),
            }

    def pos_mu(self, pos: str, t: int) -> float:
        info = self.position.get(pos)
        if info is None:
            return DEFAULT_POS_RATE
        i = np.searchsorted(info["t"], t, side="left")
        if info["cumM"][i] <= 0:
            return DEFAULT_POS_RATE
        return info["cumG"][i] / info["cumM"][i]

    def rate(self, name: str, t: int, fallback_pos: str) -> float:
        """Shrunk goals per 90 (pre-registered: K_RATE, position prior)."""
        info = self.player.get(name)
        if info is None:
            G = M = 0.0
            pos = fallback_pos
        else:
            i = np.searchsorted(info["t"], t, side="left")
            G, M = info["cumG"][i], info["cumM"][i]
            pos = info["pos"][i - 1] if i > 0 else fallback_pos
        mu = self.pos_mu(pos, t)
        return (G + K_RATE * mu) / (M + K_RATE)

    def sub_minutes(self, name: str, t: int) -> float:
        info = self.player.get(name)
        if info is None:
            return SUB_MIN_FALLBACK
        i = np.searchsorted(info["t"], t, side="left")
        n = info["cumSubN"][i]
        return info["cumSubMin"][i] / n if n > 0 else SUB_MIN_FALLBACK


def _team_history(rows: pd.DataFrame):
    """Per team: fixture keys sorted by kickoff; per (fixture, team): name ->
    (minutes, starts, position) for that team's players in that fixture."""
    teamed = rows.dropna(subset=["team"])
    hist = defaultdict(list)
    for (team, fk), g in teamed.groupby(["team", "fk"], sort=False):
        hist[team].append((g["kickoff"].min(), fk))
    for team in hist:
        hist[team].sort()
    times = {team: [k for k, _ in v] for team, v in hist.items()}
    fk_rows = {}
    for (fk, team), g in teamed.groupby(["fk", "team"], sort=False):
        fk_rows[(fk, team)] = list(
            zip(g["name"], g["minutes"], g["starts"], g["position"])
        )
    return hist, times, fk_rows


def _prior_fixtures(hist, times, team: str, kickoff: pd.Timestamp) -> list[str]:
    if team not in hist:
        return []
    i = bisect.bisect_left(times[team], kickoff)
    return [fk for _, fk in hist[team][max(0, i - LAST_N) : i]]


def team_contribution(pidx: PlayerIndex, hist, times, fk_rows, team: str, fk: str, kickoff: pd.Timestamp, lineup: str):
    """Sum over players of rate * E[minutes] / 90, or None if undefined.

    Returns (contribution, total expected minutes). The second value is used
    only by the post-registration sensitivity arm to rescale the actual-XI
    lambda to NOMINAL_TEAM_MINUTES."""
    t = np.datetime64(kickoff, "ns").astype(np.int64)
    if lineup == "i":
        players = fk_rows.get((fk, team))
        if not players:
            return None
        total = 0.0
        minutes_sum = 0.0
        for name, minutes, starts, position in players:
            if minutes <= 0:
                continue
            e = 90.0 if starts == 1 else pidx.sub_minutes(name, t)
            total += pidx.rate(name, t, position) * e / 90.0
            minutes_sum += e
        return total, minutes_sum
    prior = _prior_fixtures(hist, times, team, kickoff)
    if not prior:
        return None
    n = len(prior)
    mins: dict[str, float] = defaultdict(float)
    pos: dict[str, str] = {}
    for fk_p in prior:  # chronological, so the latest position wins
        for name, minutes, _starts, position in fk_rows.get((fk_p, team), []):
            mins[name] += minutes
            pos[name] = position
    total = 0.0
    minutes_sum = 0.0
    for name, s in mins.items():
        e = s / n
        if e > 0:
            total += pidx.rate(name, t, pos[name]) * e / 90.0
            minutes_sum += e
    return total, minutes_sum


# --------------------------------------------------------------------------
# Per-date context: Dixon-Coles refit and opponent / home factors
# --------------------------------------------------------------------------


def date_context(fd: pd.DataFrame, date: pd.Timestamp) -> dict:
    prior = fd[fd["date"] < date]
    gh = prior["goals_home"].to_numpy()
    ga = prior["goals_away"].to_numpy()
    n = len(prior)
    mu_ga = (gh.sum() + ga.sum()) / (2.0 * n)
    conc = defaultdict(lambda: [0.0, 0])
    for h, a, g_h, g_a in zip(prior["team_home"], prior["team_away"], gh, ga):
        conc[h][0] += g_a
        conc[h][1] += 1
        conc[a][0] += g_h
        conc[a][1] += 1
    fac = {t: ((s + K_OPP * mu_ga) / (c + K_OPP)) / mu_ga for t, (s, c) in conc.items()}
    return {
        "dc": fit_dixon_coles(prior),
        "fac": fac,
        "h": float(np.sqrt(gh.mean() / ga.mean())),
        "n_prior": n,
    }


def _outcome(gh: float, ga: float) -> int:
    return 0 if gh > ga else (1 if gh == ga else 2)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def rps(p: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Ranked probability score, ordered H, D, A. p: (N,3); out: (N,) in {0,1,2}."""
    o = np.eye(3)[out]
    c = np.cumsum(p, axis=1)[:, :2]
    oc = np.cumsum(o, axis=1)[:, :2]
    return 0.5 * ((c - oc) ** 2).sum(axis=1)


def log_loss_1x2(p: np.ndarray, out: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(out)), out], 1e-15, 1.0))


def brier_ou(p_over: np.ndarray, over: np.ndarray) -> np.ndarray:
    return (p_over - over) ** 2


def paired_bootstrap(diffs: np.ndarray, idx: np.ndarray) -> dict:
    """diffs: arm - DC per match. idx: (B, N) resample indices, shared across comparisons."""
    boots = diffs[idx].mean(axis=1)
    return {
        "mean": float(diffs.mean()),
        "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))),
        "ci_98_75": [float(np.quantile(boots, LO_Q)), float(np.quantile(boots, HI_Q))],
        "ci_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "mde_2_8se": float(2.8 * diffs.std(ddof=1) / np.sqrt(len(diffs))),
    }


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------


def build_predictions(fd: pd.DataFrame, rows: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    pidx = PlayerIndex(rows)
    hist, times, fk_rows = _team_history(rows)
    fx = fx[fx["season"].isin(ALL_SEASONS)].dropna(subset=["date"]).sort_values("kickoff").reset_index(drop=True)
    cache: dict = {}
    recs = []
    for date, grp in fx.groupby("date", sort=True):
        if date not in cache:
            cache[date] = date_context(fd, date)
        ctx = cache[date]
        for _, m in grp.iterrows():
            kick = m["kickoff"]
            rec = {
                "fk": m["fk"], "season": m["season"], "date": str(date.date()),
                "home": m["home"], "away": m["away"],
                "gh": int(m["goals_home"]), "ga": int(m["goals_away"]),
            }
            rec["outcome"] = _outcome(m["goals_home"], m["goals_away"])
            rec["over"] = float(m["goals_home"] + m["goals_away"] > 2.5)
            dc = predict_fixture(ctx["dc"], m["home"], m["away"], max_goals=MAX_GOALS)
            rec["DC"] = [dc["home_win"], dc["draw"], dc["away_win"], dc["over_2_5"]]
            rec["dc_fallback"] = bool(dc["fallback"])
            rho = float(ctx["dc"].params["rho"])
            for lineup in (*LINEUPS, SENSITIVITY_LINEUP):
                base = "i" if lineup == SENSITIVITY_LINEUP else lineup
                resH = team_contribution(pidx, hist, times, fk_rows, m["home"], m["fk"], kick, base)
                resA = team_contribution(pidx, hist, times, fk_rows, m["away"], m["fk"], kick, base)
                cH = cA = None
                if resH is not None and resA is not None:
                    cH, cA = resH[0], resA[0]
                    if lineup == SENSITIVITY_LINEUP:
                        cH, cA = cH * NOMINAL_TEAM_MINUTES / resH[1], cA * NOMINAL_TEAM_MINUTES / resA[1]
                if cH is None or cA is None or cH <= 0 or cA <= 0:
                    rec[f"BU_{lineup}"] = None
                    rec[f"defined_{lineup}"] = False
                    continue
                lam_h = cH * ctx["fac"].get(m["away"], 1.0) * ctx["h"]
                lam_a = cA * ctx["fac"].get(m["home"], 1.0) / ctx["h"]
                g = pb.models.create_dixon_coles_grid(lam_h, lam_a, rho=rho, max_goals=MAX_GOALS)
                rec[f"BU_{lineup}"] = [g.home_win, g.draw, g.away_win, g.total_goals("over", 2.5)]
                rec[f"lam_{lineup}"] = [float(lam_h), float(lam_a)]
                rec[f"defined_{lineup}"] = True
            recs.append(rec)
    return pd.DataFrame(recs)


def lineup_flags(rows: pd.DataFrame, fx: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """Rotation / absence proxies from the ACTUAL record (classification only)."""
    hist, times, fk_rows = _team_history(rows)
    kick_of = dict(zip(fx["fk"], fx["kickoff"]))
    out = []
    for _, r in pred.iterrows():
        kick = kick_of[r["fk"]]
        rot, absn = False, False
        for team in (r["home"], r["away"]):
            now = {n for n, _m, s, _p in fk_rows.get((r["fk"], team), []) if s == 1}
            prior = _prior_fixtures(hist, times, team, kick)
            if prior:
                prev = {n for n, _m, s, _p in fk_rows.get((prior[-1], team), []) if s == 1}
                rot |= len(now - prev) >= 3
            counts: dict[str, int] = defaultdict(int)
            for fk_p in prior:
                for n, _m, s, _p in fk_rows.get((fk_p, team), []):
                    if s == 1:
                        counts[n] += 1
            regulars = {n for n, c in counts.items() if c >= 3}
            absn |= len(regulars - now) >= 2
        out.append(rot or absn)
    pred = pred.copy()
    pred["disrupted"] = out
    return pred


def _blend(pDC: np.ndarray, pBU: np.ndarray, w: float) -> np.ndarray:
    return w * pBU + (1.0 - w) * pDC


def evaluate(pred: pd.DataFrame) -> dict:
    results = {"n_matches_total": int(len(pred))}
    rng = np.random.default_rng(SEED)
    out_all = {}
    weights_used: dict = {}
    for lineup in (*LINEUPS, SENSITIVITY_LINEUP):
        ok = pred[pred[f"defined_{lineup}"]].copy()
        # Blend weight per test season, fitted on earlier folds only.
        blend_pred = np.zeros((len(ok), 4))
        dc_arr = np.array(ok["DC"].tolist(), dtype=float)
        bu_arr = np.array(ok[f"BU_{lineup}"].tolist(), dtype=float)
        out = ok["outcome"].to_numpy()
        over = ok["over"].to_numpy()
        season = ok["season"].to_numpy()
        for s in TEST_SEASONS:
            earlier = np.isin(season, ALL_SEASONS[: ALL_SEASONS.index(s)])
            test = season == s
            if not earlier.any():
                continue
            fit_rps = []
            for w in BLEND_GRID:
                p = _blend(dc_arr[earlier, :3], bu_arr[earlier, :3], w)
                fit_rps.append(rps(p, out[earlier]).mean())
            w_star = float(BLEND_GRID[int(np.argmin(fit_rps))])
            weights_used.setdefault(lineup, {})[s] = w_star
            blend_pred[test, :3] = _blend(dc_arr[test, :3], bu_arr[test, :3], w_star)
            blend_pred[test, 3] = _blend(dc_arr[test, 3], bu_arr[test, 3], w_star)
        blend_pred[season == FIT_SEASON, :] = 0  # fit fold is not evaluated
        test_mask = np.isin(season, TEST_SEASONS)
        T = ok[test_mask].copy()
        idx_t = np.where(test_mask)[0]
        dc = dc_arr[idx_t]
        bu = bu_arr[idx_t]
        bl = blend_pred[idx_t]
        o = out[idx_t]
        ov = over[idx_t]
        metrics = {
            "DC": {"rps": rps(dc[:, :3], o), "ll": log_loss_1x2(dc[:, :3], o), "brier": brier_ou(dc[:, 3], ov)},
            "BU": {"rps": rps(bu[:, :3], o), "ll": log_loss_1x2(bu[:, :3], o), "brier": brier_ou(bu[:, 3], ov)},
            "BL": {"rps": rps(bl[:, :3], o), "ll": log_loss_1x2(bl[:, :3], o), "brier": brier_ou(bl[:, 3], ov)},
        }
        N = len(o)
        idx = rng.integers(0, N, size=(N_BOOT, N))
        comps = {}
        for arm in ("BU", "BL"):
            comps[arm] = {}
            for metric in ("rps", "ll", "brier"):
                d = metrics[arm][metric] - metrics["DC"][metric]
                comps[arm][metric] = paired_bootstrap(d, idx)
        # Exploratory: disrupted vs not, RPS only.
        dis = T["disrupted"].to_numpy()
        sub = {}
        for arm in ("BU", "BL"):
            d = metrics[arm]["rps"] - metrics["DC"]["rps"]
            sub[arm] = {
                "n_disrupted": int(dis.sum()),
                "n_not_disrupted": int((~dis).sum()),
                "mean_diff_disrupted": float(d[dis].mean()) if dis.any() else None,
                "mean_diff_not_disrupted": float(d[~dis].mean()) if (~dis).any() else None,
                "share_of_total_gain_from_disrupted": (
                    float(d[dis].sum() / d.sum()) if d.sum() < 0 else None
                ),
            }
            if dis.any():
                ii = rng.integers(0, dis.sum(), size=(N_BOOT, dis.sum()))
                sub[arm]["ci_95_disrupted"] = list(np.quantile(d[dis][ii].mean(axis=1), [0.025, 0.975]).astype(float))
        per_season = {}
        for s in TEST_SEASONS:
            m_s = T["season"].to_numpy() == s
            per_season[s] = {
                arm: {"rps": float(metrics[arm]["rps"][m_s].mean()), "ll": float(metrics[arm]["ll"][m_s].mean()), "brier": float(metrics[arm]["brier"][m_s].mean())}
                for arm in ARMS
            }
        out_all[lineup] = {
            "n_eval": int(N),
            "means": {arm: {k: float(v.mean()) for k, v in metrics[arm].items()} for arm in ARMS},
            "comparisons_vs_DC": comps,
            "per_season": per_season,
            "subgroups": sub,
            "blend_weights": weights_used.get(lineup, {}),
        }
    results["lineups"] = out_all
    return results


def run(write: bool = True) -> dict:
    fd = load_football_data()
    rows = load_fpl_rows()
    fx = build_fixtures(rows, fd)
    pred = build_predictions(fd, rows, fx)
    pred = lineup_flags(rows, fx, pred)
    coverage = {
        "football_data_matches": int(len(fd)),
        "fpl_fixtures_with_team_names": int(len(fx)),
        "fpl_fixtures_joined_to_football_data": int(fx["date"].notna().sum()),
        "fixtures_in_all_seasons_scope": int(len(pred)),
        "per_season_scope": pred.groupby("season").size().to_dict(),
        "defined_both_lineups": int((pred["defined_i"] & pred["defined_ii"]).sum()),
        "defined_ii": int(pred["defined_ii"].sum()),
        "dc_fallback_matches": int(pred["dc_fallback"].sum()),
        "excluded_undefined_ii": int((~pred["defined_ii"]).sum()),
    }
    # All arms share one match set: the predicted-lineup (ii) defined set.
    pred_common = pred[pred["defined_ii"] & pred["defined_i"] & pred[f"defined_{SENSITIVITY_LINEUP}"]].copy()
    results = evaluate(pred_common)
    # Calibration check: mean team lambda vs mean actual team goals, per lineup.
    results["lambda_calibration"] = {
        L: {
            "mean_lambda_home": float(pred_common[f"lam_{L}"].apply(lambda v: v[0]).mean()),
            "mean_lambda_away": float(pred_common[f"lam_{L}"].apply(lambda v: v[1]).mean()),
        }
        for L in (*LINEUPS, SENSITIVITY_LINEUP)
    }
    results["lambda_calibration"]["actual_mean_goals"] = {
        "home": float(pred_common["gh"].mean()),
        "away": float(pred_common["ga"].mean()),
    }
    results["coverage"] = coverage
    results["pre_registration"] = "docs/bottomup_vs_dixoncoles_preregistration.md"
    results["matches"] = pred_common.drop(columns=[c for c in pred_common.columns if c.startswith("defined_")]).to_dict(orient="records")
    results = json.loads(json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if write:
        out = PROJECT_ROOT / "docs" / "bottomup_vs_dixoncoles_results.json"
        out.write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    res = run()
    print(json.dumps({k: v for k, v in res.items() if k not in ("matches",)}, indent=1)[:4000])
