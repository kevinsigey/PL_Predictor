"""Public /calibration metrics: scored on the counted recorded picks only
(recorded before kickoff), never on after-the-fact rows."""

import sqlite3

import pytest

from pl_predictor.tracking import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "t.db")
    store._connect().close()  # creates the schema
    return tmp_path / "t.db"


def _fixture(db, event, probs, actual_outcome, snapshotted_at, commence_time):
    conn = sqlite3.connect(db)
    for outcome, prob in probs.items():
        conn.execute(
            "INSERT INTO predictions (event_id, team_home, team_away, market, outcome_name, predicted_prob, commence_time, snapshotted_at, "
            "resolved, actual_outcome) VALUES (?,?,?,?,?,?,?,?,1,?)",
            (event, "H", "A", "1x2", outcome, prob, commence_time, snapshotted_at, 1 if outcome == actual_outcome else 0),
        )
    conn.commit()
    conn.close()


def test_metrics_cover_only_picks_recorded_before_kickoff(db):
    # Two counted fixtures: one home win called at 62%, one draw at 40%.
    _fixture(db, "e1", {"home_win": 0.62, "draw": 0.23, "away_win": 0.15}, "home_win",
             "2026-01-01T10:00:00Z", "2026-01-01T15:00:00Z")
    _fixture(db, "e2", {"home_win": 0.35, "draw": 0.40, "away_win": 0.25}, "draw",
             "2026-01-02T10:00:00Z", "2026-01-02T15:00:00Z")
    # One after-the-fact row (snapshotted after kickoff): must not count.
    _fixture(db, "e3", {"home_win": 0.90, "draw": 0.06, "away_win": 0.04}, "home_win",
             "2026-01-03T18:00:00Z", "2026-01-03T15:00:00Z")

    result = store.get_validation_calibration()

    assert result["model"]["n_matches"] == 2
    assert result["naive"]["n_matches"] == 2
    assert 0.0 <= result["model"]["rps"] <= 1.0
    assert result["model"]["brier"] is not None
    assert result["model"]["ignorance"] is not None
    # No bookmaker probability is stored with a recorded pick.
    assert result["bookmaker"] is None


def test_no_counted_picks_reports_zero_matches_not_a_fabricated_score(db):
    result = store.get_validation_calibration()
    assert result["model"]["n_matches"] == 0
    assert result["model"]["rps"] is None
    assert result["bookmaker"] is None
