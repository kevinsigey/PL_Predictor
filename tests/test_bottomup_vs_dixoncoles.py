"""Synthetic checks for the bottom-up vs Dixon-Coles research harness.

No cache or network: every input is built in-line.
"""

import numpy as np
import pandas as pd
import penaltyblog as pb

from pl_predictor.evaluate import bottomup_vs_dixoncoles as bu


def _rows(entries):
    return pd.DataFrame(
        [
            {
                "name": n,
                "position": pos,
                "kickoff": pd.Timestamp(k),
                "minutes": mins,
                "starts": st,
                "goals": g,
            }
            for n, pos, k, mins, st, g in entries
        ]
    )


def test_player_rate_uses_only_rows_strictly_before_query():
    entries = [
        ("A", "FWD", "2024-08-01 15:00", 90, 1, 1.0),
        ("A", "FWD", "2024-08-08 15:00", 90, 1, 0.0),
    ]
    idx = bu.PlayerIndex(_rows(entries))
    query = pd.Timestamp("2024-08-08 15:00").value
    before = idx.rate("A", query, "FWD")
    # A row at the query kickoff itself must not move the rate for that match.
    idx_future = bu.PlayerIndex(_rows(entries + [("A", "FWD", "2024-08-08 15:00", 90, 1, 5.0)]))
    assert idx_future.rate("A", query, "FWD") == before
    # Prior data: one goal in 1 x 90' -> (1 + K * mu) / (1 + K), mu from pooled FWD rows.
    mu = idx.pos_mu("FWD", query)
    assert np.isclose(before, (1.0 + bu.K_RATE * mu) / (1.0 + bu.K_RATE))


def test_rps_known_values():
    perfect = np.array([[1.0, 0.0, 0.0]])
    assert bu.rps(perfect, np.array([0]))[0] == 0.0
    # Forecast all home, actual away: cumulative diffs are (1, 1), so RPS = 0.5 * 2 = 1.
    assert np.isclose(bu.rps(perfect, np.array([2]))[0], 1.0)


def test_paired_bootstrap_zero_diff_has_zero_interval():
    rng = np.random.default_rng(0)
    diffs = np.zeros(50)
    idx = rng.integers(0, 50, size=(200, 50))
    res = bu.paired_bootstrap(diffs, idx)
    assert res["mean"] == 0.0
    assert res["ci_98_75"] == [0.0, 0.0]


def test_bottomup_grid_is_a_proper_probability_grid():
    g = pb.models.create_dixon_coles_grid(1.4, 1.1, rho=-0.02, max_goals=bu.MAX_GOALS)
    assert np.isclose(g.home_win + g.draw + g.away_win, 1.0)
    assert np.isclose(g.total_goals("over", 2.5) + g.total_goals("under", 2.5), 1.0)
