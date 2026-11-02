"""Bookmaker odds stored with recorded picks, and the two surfaces that
read them: the public calibration's bookmaker series and the track
record's against-the-odds (value-bet ledger) summary.

The honesty rules under test, same as the rest of the tracking store:
odds exist only where the pipeline actually had a price at record time
(historical and backfilled rows stay NULL, never reconstructed), and the
bookmaker series counts only the pre-kickoff population whose rows all
carry a price.
"""

import sqlite3

import pandas as pd
import pytest

from pl_predictor.tracking import store, value_bet_ledger


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    monkeypatch.setattr(store, "TRACKING_DB_PATH", path)
    store._connect().close()  # creates the schema
    return path


def _odds_by_outcome(db, event_id):
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT outcome_name, bookmaker_odds FROM predictions WHERE event_id = ? AND market = '1x2'",
        (event_id,),
    ).fetchall()
    conn.close()
    return dict(rows)


def _value_bet_table(**overrides):
    row = {
        "event_id": "ev1",
        "team_home": "Arsenal",
        "team_away": "Chelsea",
        "commence_time": "2026-11-07T15:00:00Z",
        "home_win_prob": 0.50,
        "draw_prob": 0.27,
        "away_win_prob": 0.23,
        "over_2_5_prob": 0.55,
        "under_2_5_prob": 0.45,
        "btts_yes_prob": 0.52,
        "top_scoreline": "2-1",
        "home_win_price": 2.10,
        "draw_price": 3.40,
        "away_win_price": 3.60,
        "over_2_5_price": 1.95,
        "under_2_5_price": 1.85,
        "btts_yes_price": 1.80,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_record_predictions_persists_the_price_each_outcome_was_published_against(db):
    assert store.record_predictions(_value_bet_table()) == 6

    odds = _odds_by_outcome(db, "ev1")
    assert odds == {"home_win": 2.10, "draw": 3.40, "away_win": 3.60}

    conn = sqlite3.connect(db)
    totals = dict(
        conn.execute(
            "SELECT outcome_name, bookmaker_odds FROM predictions WHERE event_id = 'ev1' AND market = 'totals_2_5'"
        ).fetchall()
    )
    conn.close()
    assert totals == {"over": 1.95, "under": 1.85}


def test_the_first_recorded_price_stands_when_odds_move(db):
    store.record_predictions(_value_bet_table())
    store.record_predictions(_value_bet_table(home_win_price=2.90))

    assert _odds_by_outcome(db, "ev1")["home_win"] == 2.10


def test_backfill_shaped_rows_without_prices_store_null_not_an_invention(db):
    table = _value_bet_table().drop(columns=[c for c in _value_bet_table().columns if c.endswith("_price")])
    store.record_predictions(table, backfilled=True)

    assert set(_odds_by_outcome(db, "ev1").values()) == {None}


def test_nan_and_unusable_prices_store_null(db):
    store.record_predictions(_value_bet_table(draw_price=float("nan"), away_win_price=1.0))

    odds = _odds_by_outcome(db, "ev1")
    assert odds["home_win"] == 2.10
    assert odds["draw"] is None
    assert odds["away_win"] is None


def test_a_pre_odds_schema_database_gains_the_column_and_keeps_its_rows(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL, team_home TEXT NOT NULL, team_away TEXT NOT NULL,
            market TEXT NOT NULL, outcome_name TEXT NOT NULL, predicted_prob REAL NOT NULL,
            commence_time TEXT NOT NULL, snapshotted_at TEXT NOT NULL, model_trained_at TEXT,
            resolved INTEGER NOT NULL DEFAULT 0, actual_outcome INTEGER, resolved_at TEXT,
            UNIQUE(event_id, market, outcome_name)
        )
        """
    )
    conn.execute(
        "INSERT INTO predictions (event_id, team_home, team_away, market, outcome_name, predicted_prob,"
        " commence_time, snapshotted_at) VALUES ('old1', 'H', 'A', '1x2', 'home_win', 0.5,"
        " '2026-08-01T15:00:00', '2026-08-01T10:00:00')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(store, "TRACKING_DB_PATH", path)
    store._connect().close()  # runs the migrations

    conn = sqlite3.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    stored = conn.execute("SELECT bookmaker_odds FROM predictions WHERE event_id = 'old1'").fetchone()[0]
    conn.close()
    assert "bookmaker_odds" in columns
    assert stored is None


def _resolved_fixture(db, event, probs, odds, actual_outcome, snapshotted_at, commence_time):
    conn = sqlite3.connect(db)
    for outcome, prob in probs.items():
        conn.execute(
            "INSERT INTO predictions (event_id, team_home, team_away, market, outcome_name, predicted_prob,"
            " commence_time, snapshotted_at, resolved, actual_outcome, bookmaker_odds)"
            " VALUES (?,?,?,?,?,?,?,?,1,?,?)",
            (
                event, "H", "A", "1x2", outcome, prob, commence_time, snapshotted_at,
                1 if outcome == actual_outcome else 0, odds.get(outcome) if odds else None,
            ),
        )
    conn.commit()
    conn.close()


def test_bookmaker_series_scores_only_counted_fixtures_with_a_full_price_triple(db):
    probs = {"home_win": 0.50, "draw": 0.27, "away_win": 0.23}
    # Counted (recorded before kickoff) with all three prices: in both series.
    _resolved_fixture(db, "full", probs, {"home_win": 2.10, "draw": 3.40, "away_win": 3.60},
                      "home_win", "2026-01-01T10:00:00Z", "2026-01-01T15:00:00Z")
    # Counted but one price missing: model series only.
    _resolved_fixture(db, "partial", probs, {"home_win": 2.10, "away_win": 3.60},
                      "draw", "2026-01-02T10:00:00Z", "2026-01-02T15:00:00Z")
    # Full prices but recorded after kickoff: in neither series.
    _resolved_fixture(db, "late", probs, {"home_win": 2.10, "draw": 3.40, "away_win": 3.60},
                      "home_win", "2026-01-03T18:00:00Z", "2026-01-03T15:00:00Z")

    result = store.get_validation_calibration()

    assert result["model"]["n_matches"] == 2
    assert result["bookmaker"] is not None
    assert result["bookmaker"]["n_matches"] == 1
    assert 0.0 <= result["bookmaker"]["rps"] <= 1.0
    assert result["bookmaker"]["brier"] is not None
    assert result["bookmaker"]["ignorance"] is not None


@pytest.fixture
def ledger_db(tmp_path, monkeypatch):
    path = tmp_path / "ledger.db"
    monkeypatch.setattr(value_bet_ledger, "TRACKING_DB_PATH", path)
    value_bet_ledger._connect().close()  # creates the schema
    return path


def _flag(db, event_id, price, model_prob, commence_time, resolved, won):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO value_bets (event_id, team_home, team_away, market, outcome_name, model_prob,"
        " implied_prob, price, edge, commence_time, snapshotted_at, resolved, won,"
        " actual_goals_home, actual_goals_away, resolved_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            event_id, "H", "A", "1x2", "home_win", model_prob, 0.40, price, 0.10,
            commence_time, commence_time, resolved, won,
            # Reconciled rows always carry the scoreline they were graded on.
            (2 if won else 0) if resolved else None,
            (1 if won else 2) if resolved else None,
            commence_time if resolved else None,
        ),
    )
    conn.commit()
    conn.close()


def test_value_bet_summary_of_an_empty_ledger_is_zeros_and_nones(ledger_db):
    summary = value_bet_ledger.get_value_bet_summary()

    assert summary["n_flagged"] == 0
    assert summary["n_resolved"] == 0
    assert summary["win_rate"] is None
    assert summary["average_odds"] is None
    assert summary["roi_pct"] is None
    assert summary["first_flagged_at"] is None


def test_value_bet_summary_grades_settled_bets_and_leaves_pending_open(ledger_db):
    _flag(ledger_db, "w1", 2.20, 0.50, "2026-10-01T15:00:00", 1, 1)
    _flag(ledger_db, "l1", 2.60, 0.45, "2026-10-02T15:00:00", 1, 0)
    _flag(ledger_db, "p1", 3.10, 0.40, "2026-12-01T15:00:00", 0, None)

    summary = value_bet_ledger.get_value_bet_summary()

    assert summary["n_flagged"] == 3
    assert summary["n_resolved"] == 2
    assert summary["n_pending"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["win_rate"] == 50.0
    assert summary["average_odds"] == pytest.approx(2.40)
    assert summary["first_flagged_at"].startswith("2026-10-01")
    assert summary["last_flagged_at"].startswith("2026-12-01")
    # The staking replay places real (simulated) bets on both settled rows,
    # so a return exists — and says how many bets it covers.
    assert summary["n_staked"] == 2
    assert summary["roi_pct"] is not None
