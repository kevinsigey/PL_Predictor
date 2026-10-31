"""Lag and formula checks for the player-vs-opponent research module.

Synthetic data only: no network, no cache. The production feature builders are
replaced by pass-throughs, because these tests check the history lags, not the
production baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pl_predictor.evaluate import player_opponent_effect as m
from pl_predictor.features import player_form


def _synthetic_history() -> pd.DataFrame:
    teams = {1: "Arsenal", 2: "Chelsea", 3: "Everton", 4: "Fulham"}
    # (fixture, home element, away element, home team, away team, kickoff day, home goals, away goals)
    games = [
        (1, 1, 2, "Arsenal", "Chelsea", 1, 2, 0),
        (2, 3, 4, "Everton", "Fulham", 2, 1, 1),
        (3, 1, 3, "Arsenal", "Everton", 3, 0, 1),
        (4, 2, 4, "Chelsea", "Fulham", 4, 3, 0),
        (5, 1, 2, "Arsenal", "Chelsea", 5, 1, 1),
        (6, 3, 2, "Everton", "Chelsea", 6, 0, 2),
    ]
    rows = []
    for fx, he, ae, ht, at, day, hg, ag in games:
        kick = pd.Timestamp("2022-08-06") + pd.Timedelta(days=day)
        for el, team, was_home, goals in ((he, ht, True, hg), (ae, at, False, ag)):
            rows.append({
                "season": "2022-23", "fixture": fx, "element": el, "name": f"P{el}",
                "position": "MID", "team": team, "was_home": was_home, "kickoff_time": kick,
                "minutes": 90, "goals_scored": goal_for(el, fx, goals), "assists": 0,
                "team_h_score": hg, "team_a_score": ag,
            })
    return pd.DataFrame(rows)


def goal_for(element: int, fixture: int, team_goals: int) -> int:
    return min(team_goals, 1)


def _passthrough(df, *args, **kwargs):
    played = df[df["minutes"] > 0].copy()
    return played, []


def _patch(monkeypatch, raw: pd.DataFrame) -> None:
    monkeypatch.setattr(m.fpl_history, "load_player_gw_history", lambda seasons=None: raw.copy())
    monkeypatch.setattr(player_form, "build_historical_start_features", lambda df, *a, **k: (df, []))
    monkeypatch.setattr(player_form, "build_historical_player_form", lambda df, *a, **k: (df.copy(), []))


HISTORY_COLS = ["f_opp", "pi_pair_scorer", "pi_pair_inv", "GV_goals", "GV_inv", "MV",
                "GC_goals", "GC_inv", "MC", "r0", "tier", "G_prior", "M_prior"]


def test_history_features_ignore_the_current_row(monkeypatch):
    raw = _synthetic_history()
    _patch(monkeypatch, raw)
    base = m.build_frame(seasons=["2022-23"])

    # Change the outcome of one appearance; its own features must not move.
    target_fx, target_el = 5, 1
    bumped = raw.copy()
    mask = (bumped["fixture"] == target_fx) & (bumped["element"] == target_el)
    bumped.loc[mask, "goals_scored"] = 3
    _patch(monkeypatch, bumped)
    changed = m.build_frame(seasons=["2022-23"])

    def row(frame):
        return frame[(frame["fixture"] == target_fx) & (frame["element"] == target_el)].iloc[0]

    for col in HISTORY_COLS:
        a, b = row(base)[col], row(changed)[col]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), col


def test_later_rows_do_see_the_earlier_outcome(monkeypatch):
    raw = _synthetic_history()
    _patch(monkeypatch, raw)
    base = m.build_frame(seasons=["2022-23"])
    # Element 1 scores in fixture 1 (day 1). Zero that out; fixture 3 (day 3) is later.
    bumped = raw.copy()
    bumped.loc[(bumped["fixture"] == 1) & (bumped["element"] == 1), "goals_scored"] = 0
    _patch(monkeypatch, bumped)
    changed = m.build_frame(seasons=["2022-23"])

    def row(frame):
        return frame[(frame["fixture"] == 3) & (frame["element"] == 1)].iloc[0]

    assert row(base)["G_prior"] == pytest.approx(1.0)
    assert row(changed)["G_prior"] == pytest.approx(0.0)


def test_shrunk_rate_formula_and_zero_without_history():
    # One goal in 90 minutes of matchup history, prior 0.01 per minute, kappa 90:
    # r = (1 + 90 * 0.01) / (90 + 90) = 0.0105...; z = log(r / 0.01).
    frame = pd.DataFrame({
        "pi_pair_scorer": [0.01, 0.01], "GV_goals": [1.0, 0.0], "MV": [90.0, 0.0],
        "GV_inv": [1.0, 0.0], "GC_goals": [np.nan, 0.0], "GC_inv": [np.nan, 0.0],
        "MC": [np.nan, 0.0], "tier": [np.nan, 0.0], "pi_pair_inv": [0.01, 0.01],
    })
    z = m.candidate_feature(frame, "A", "scorer", kappa=90.0)
    expected = np.log(((1.0 + 90 * 0.01) / (90 + 90)) / 0.01)
    assert z[0] == pytest.approx(expected)
    assert z[1] == pytest.approx(0.0)  # no matchup history, feature collapses to the prior
    zb = m.candidate_feature(frame, "B", "scorer", kappa=90.0)
    assert zb[0] == pytest.approx(0.0)  # unknown tier: no cluster feature
