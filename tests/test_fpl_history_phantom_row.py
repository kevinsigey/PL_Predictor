"""FPL pre-creates a history row for the current gameweek's fixture before
it is played (0 minutes, kickoff in the future). Serving must not count it
as a played match; a zero-minute row for a match already kicked off (an
unused substitute) is a real negative example and stays."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from pl_predictor.data.fpl_api import _normalize_history


def _row(round_, minutes, starts, kickoff):
    return {
        "round": round_,
        "minutes": minutes,
        "starts": starts,
        "kickoff_time": kickoff.isoformat().replace("+00:00", "Z"),
    }


def test_the_unplayed_current_gameweek_row_is_dropped():
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        [
            _row(7, 90, 1, now - timedelta(days=14)),
            _row(8, 13, 0, now - timedelta(days=7)),
            _row(9, 0, 0, now + timedelta(days=1)),  # next fixture, not played yet
        ]
    )
    out = _normalize_history(df)
    assert out["GW"].tolist() == [7, 8]
    assert out["minutes"].tolist() == [90, 13]


def test_an_unused_substitute_in_a_played_match_stays():
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        [
            _row(7, 90, 1, now - timedelta(days=14)),
            _row(8, 0, 0, now - timedelta(days=7)),  # played match, never came on
        ]
    )
    out = _normalize_history(df)
    assert out["GW"].tolist() == [7, 8]


def test_rows_without_a_parseable_kickoff_are_kept():
    df = pd.DataFrame([{"round": 7, "minutes": 90, "starts": 1, "kickoff_time": None}])
    out = _normalize_history(df)
    assert out["GW"].tolist() == [7]
