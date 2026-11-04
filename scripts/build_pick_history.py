"""Build data/pick_history.json: each fixture's model probabilities over time.

The refresh workflow commits data/public_snapshot.json on every run, so the
git history of that one file IS the history of the published predictions. This
walks it oldest-first and keeps one point per change in a fixture's three-way
probabilities (and the market's implied ones, where a live price existed).

Run from anywhere inside the repo:

    python3 scripts/build_pick_history.py

Output shape:
    {"built_from": "...", "events": {event_id: [{"t", "p_home", "p_draw",
     "p_away", "m_home", "m_draw", "m_away"}, ...]}}
Points are dropped when the probabilities did not move, so the file only grows
when a prediction actually changed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = "data/public_snapshot.json"
OUT = REPO / "data" / "pick_history.json"
OUTCOMES = (("p_home", "home_win"), ("p_draw", "draw"), ("p_away", "away_win"))


def _probs(detail: dict) -> dict | None:
    point = {}
    for key, outcome in OUTCOMES:
        value = (detail.get(outcome) or {}).get("prob")
        if not isinstance(value, (int, float)):
            return None
        point[key] = round(float(value), 4)
    for key, outcome in OUTCOMES:
        implied = (detail.get(outcome) or {}).get("implied")
        point["m" + key[1:]] = round(float(implied), 4) if isinstance(implied, (int, float)) else None
    return point


def build_history(snapshots: Iterable[tuple[str, dict]]) -> dict[str, list[dict]]:
    """Fold (generated_at, snapshot) pairs, oldest first, into per-event points."""
    events: dict[str, list[dict]] = {}
    for generated_at, snapshot in snapshots:
        for event_id, detail in (snapshot.get("fixture_detail_by_event_id") or {}).items():
            probs = _probs(detail or {})
            if probs is None:
                continue
            points = events.setdefault(event_id, [])
            if points and all(points[-1][k] == probs[k] for k in probs):
                continue
            points.append({"t": generated_at, **probs})
    return {event_id: points for event_id, points in events.items() if points}


def _git_snapshots() -> list[tuple[str, dict]]:
    shas = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--", SNAPSHOT],
        cwd=REPO, check=True, capture_output=True, text=True,
    ).stdout.split()
    out = []
    for sha in shas:
        raw = subprocess.run(["git", "show", f"{sha}:{SNAPSHOT}"], cwd=REPO, capture_output=True).stdout
        try:
            snapshot = json.loads(raw)
        except ValueError:
            continue  # a commit that did not hold a parseable snapshot contributes nothing
        if snapshot.get("fixture_detail_by_event_id"):
            out.append((snapshot.get("generated_at", ""), snapshot))
    return out


def main() -> None:
    snapshots = _git_snapshots()
    events = build_history(snapshots)
    payload = {
        "built_from": f"git history of {SNAPSHOT} ({len(snapshots)} snapshots)",
        "events": events,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    points = sum(len(v) for v in events.values())
    print(f"wrote {OUT.relative_to(REPO)}: {len(events)} fixtures, {points} points")


if __name__ == "__main__":
    main()
