"""snapshot-meta: the freshness probe the hub's cache gates on, served from
the public snapshot like every other sport's."""

from pl_predictor.api import routes


def test_snapshot_meta_reports_the_snapshots_generated_at(monkeypatch):
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(
        routes,
        "_public_snapshot",
        lambda: {"generated_at": "2026-11-01T21:20:11+00:00", "model_fingerprint": "abc123"},
    )
    meta = routes.get_snapshot_meta()
    assert meta["generated_at"] == "2026-11-01T21:20:11+00:00"
    assert meta["source"] == "public_snapshot"
    assert meta["model_fingerprint"] == "abc123"


def test_public_calibration_serves_the_baked_snapshot_value(monkeypatch):
    baked = {
        "model": {"rps": 0.21, "brier": 0.62, "ignorance": 2.4, "n_matches": 39},
        "bookmaker": None,
        "naive": {"rps": 0.24, "brier": 0.68, "n_matches": 39},
        "season": "2026-2027",
    }
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: {"calibration": baked})
    assert routes.get_calibration() == baked
