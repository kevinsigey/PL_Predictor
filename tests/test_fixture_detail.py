import pandas as pd

from pl_predictor.api import routes


def test_live_fpl_fixture_id_resolves_for_detail_and_player_lookup(monkeypatch):
    """A current-gameweek card must use its FPL id after kickoff."""
    live_fixture = pd.DataFrame(
        [{
            "event_id": 16,
            "team_home": "Chelsea",
            "team_away": "Brighton",
            "commence_time": pd.Timestamp("2026-08-30T13:00:00Z"),
            "gameweek": 2,
            "has_odds": False,
        }]
    )
    monkeypatch.setattr(routes, "_value_bet_table", lambda: pd.DataFrame())
    monkeypatch.setattr(routes, "_get_remaining_fixtures_df", lambda: pd.DataFrame())
    monkeypatch.setattr(routes.fixtures_mod, "_fixtures_from_fpl_api", lambda: live_fixture)
    monkeypatch.setattr(routes, "_get_models", lambda: {"scoreline": object()})
    monkeypatch.setattr(routes.scoreline, "predict_fixture", lambda *_args, **_kwargs: {"home_win": 0.5})
    monkeypatch.setattr(routes, "_team_fixture_to_summary", lambda fixture, _prediction: fixture)
    monkeypatch.setattr(
        routes,
        "_build_fixture_detail",
        lambda _summary, home, away, read_only=False: {"home": home, "away": away},
    )

    assert routes.fixture_detail("16") == {"home": "Chelsea", "away": "Brighton"}
    assert routes._resolve_fixture_teams("16") == ("Chelsea", "Brighton")


def test_finished_fixture_in_odds_table_returns_recorded_probs(monkeypatch):
    """Event 105: still inside the odds window, but already played — its
    frozen pre-kickoff pick must win over odds recomputed after the match."""
    recorded = {
        "event_id": "105",
        "team_home": "Coventry",
        "team_away": "Crystal Palace",
        "commence_time": "2026-01-01T20:00:00",  # naive, as tracking_store stores it
        "home_win": 0.1,
        "draw": 0.26,
        "away_win": 0.398,
        "over_2_5": 0.5,
        "under_2_5": 0.5,
        "btts_yes": 0.5,
        "predicted_scoreline": "1-2",
    }
    odds_table = pd.DataFrame([{
        "event_id": 105,
        "team_home": "Coventry",
        "team_away": "Crystal Palace",
        "commence_time": recorded["commence_time"],
    }])
    monkeypatch.setattr(routes, "_value_bet_table", lambda: odds_table)
    monkeypatch.setattr(
        routes.tracking_store,
        "get_fixture_prediction",
        lambda event_id: recorded if event_id == "105" else None,
    )
    # Sentinel for the recomputed path: if this is what comes back, the
    # odds-table branch ran instead of the recorded one.
    monkeypatch.setattr(routes, "_row_to_summary", lambda *_a, **_k: {"source": "odds"})
    monkeypatch.setattr(routes, "_build_fixture_detail", lambda summary, home, away, read_only=False: summary)

    detail = routes.fixture_detail("105")
    assert (detail.home_win.prob, detail.draw.prob, detail.away_win.prob) == (0.1, 0.26, 0.398)

    # An upcoming fixture with a recorded pick must still use the live odds.
    recorded["commence_time"] = "2099-01-01T20:00:00"
    assert routes.fixture_detail("105") == {"source": "odds"}
