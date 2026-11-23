"""The fixture sparkline's history: built from snapshot git history by
scripts/build_pick_history.py, served by /fixtures/{event_id}/pick-history.
These pin the fold (one point per real change, market prices kept when present)
and the endpoint's empty answer, with no git or HTTP involved."""

import importlib.util
from pathlib import Path

from pl_predictor.api import routes

_spec = importlib.util.spec_from_file_location(
    "build_pick_history", Path(__file__).resolve().parents[1] / "scripts" / "build_pick_history.py"
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def _detail(home, draw, away, implied_home=None):
    return {
        "home_win": {"prob": home, "implied": implied_home},
        "draw": {"prob": draw, "implied": None},
        "away_win": {"prob": away, "implied": None},
    }


def test_fold_keeps_one_point_per_change_and_drops_unchanged_snapshots():
    snapshots = [
        ("t1", {"fixture_detail_by_event_id": {"e1": _detail(0.5, 0.3, 0.2)}}),
        ("t2", {"fixture_detail_by_event_id": {"e1": _detail(0.5, 0.3, 0.2)}}),
        ("t3", {"fixture_detail_by_event_id": {"e1": _detail(0.6, 0.25, 0.15, implied_home=0.55)}}),
    ]
    history = builder.build_history(snapshots)
    points = history["e1"]
    assert [p["t"] for p in points] == ["t1", "t3"]
    assert points[1]["p_home"] == 0.6
    assert points[1]["m_home"] == 0.55
    assert points[0]["m_home"] is None


def test_fixture_without_three_way_probabilities_is_skipped():
    snapshots = [("t1", {"fixture_detail_by_event_id": {"e2": {"home_win": None}}})]
    assert builder.build_history(snapshots) == {}


def test_endpoint_answers_empty_when_history_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "PICK_HISTORY_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(routes, "_pick_history_events", None)
    assert routes.fixture_pick_history("nope") == {"event_id": "nope", "points": []}
    monkeypatch.setattr(routes, "_pick_history_events", None)
