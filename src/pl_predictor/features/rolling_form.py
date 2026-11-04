"""rolling_form.py — team-level rolling form features.

Same idiom as FPL_Optimizer/features.py::build_lag_features
(`shift(1).rolling(w, min_periods=1).mean()` grouped by team, sorted by
date), just at team-match granularity instead of player-gameweek. Each
historical match becomes two team-perspective rows (home + away) so form can
be computed per-team regardless of venue, plus separate home-only/away-only
windows since home/away splits matter a lot for goals and cards markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_WINDOWS = [3, 5, 10]

# Base per-match stats a team-perspective row carries, computed from the raw
# football-data.co.uk columns (hs/as = shots, hst/ast = shots on target,
# hc/ac = corners, hy/ay/hr/ar = yellow/red cards).
BASE_STATS = [
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "fouls_for",
    "fouls_against",
    "cards_for",
    "cards_against",
    "points",
]


def _points(goals_for: pd.Series, goals_against: pd.Series) -> pd.Series:
    return np.select(
        [goals_for > goals_against, goals_for == goals_against],
        [3, 1],
        default=0,
    )


def to_team_perspective(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Melt one row per match into two rows (home perspective, away
    perspective). Columns present in the source but not listed in
    `BASE_STATS` (e.g. odds columns) are dropped — this frame is form-feature
    input only."""
    df = matches_df
    zeros = pd.Series(0, index=df.index)

    home = pd.DataFrame(
        {
            "date": df["date"],
            "season": df["season"],
            "team": df["team_home"],
            "opponent": df["team_away"],
            "was_home": True,
            "goals_for": df["goals_home"],
            "goals_against": df["goals_away"],
            "shots_for": df.get("hs"),
            "shots_against": df.get("as"),
            "shots_on_target_for": df.get("hst"),
            "shots_on_target_against": df.get("ast"),
            "corners_for": df.get("hc"),
            "corners_against": df.get("ac"),
            "fouls_for": df.get("hf"),
            "fouls_against": df.get("af"),
            "cards_for": df.get("hy", zeros).fillna(0) + df.get("hr", zeros).fillna(0),
            "cards_against": df.get("ay", zeros).fillna(0) + df.get("ar", zeros).fillna(0),
        }
    )
    home["points"] = _points(home["goals_for"], home["goals_against"])

    away = pd.DataFrame(
        {
            "date": df["date"],
            "season": df["season"],
            "team": df["team_away"],
            "opponent": df["team_home"],
            "was_home": False,
            "goals_for": df["goals_away"],
            "goals_against": df["goals_home"],
            "shots_for": df.get("as"),
            "shots_against": df.get("hs"),
            "shots_on_target_for": df.get("ast"),
            "shots_on_target_against": df.get("hst"),
            "corners_for": df.get("ac"),
            "corners_against": df.get("hc"),
            "fouls_for": df.get("af"),
            "fouls_against": df.get("hf"),
            "cards_for": df.get("ay", zeros).fillna(0) + df.get("ar", zeros).fillna(0),
            "cards_against": df.get("hy", zeros).fillna(0) + df.get("hr", zeros).fillna(0),
        }
    )
    away["points"] = _points(away["goals_for"], away["goals_against"])

    long_df = pd.concat([home, away], ignore_index=True)
    return long_df.sort_values(["team", "date"]).reset_index(drop=True)


def _add_rolling(long_df: pd.DataFrame, group_cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
    lag_cols = []
    new_cols = {}
    grouped = long_df.groupby(group_cols, sort=False)
    for feat in BASE_STATS:
        if feat not in long_df.columns:
            continue
        for w in LAG_WINDOWS:
            col = f"{prefix}last_{w}_{feat}"
            new_cols[col] = grouped[feat].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
            lag_cols.append(col)
    # Assigning ~40 columns one at a time (long_df[col] = ...) fragments the
    # block manager badly enough that pandas warns on the next read; build
    # them in a dict and concat once instead.
    long_df = pd.concat([long_df, pd.DataFrame(new_cols, index=long_df.index)], axis=1)
    return long_df, lag_cols


def latest_form(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Each team's *current* rolling form (as of right now, i.e. including
    their most recent played match) — used for scoring an upcoming fixture,
    as opposed to `build_rolling_form`'s shift(1) historical features. Same
    column names as `build_rolling_form`'s base feature columns, indexed by
    team."""
    long_df = to_team_perspective(matches_df)
    home_only = long_df[long_df["was_home"]]
    away_only = long_df[~long_df["was_home"]]

    rows = {}
    for team, group in long_df.groupby("team"):
        row = {}
        for feat in BASE_STATS:
            if feat not in group.columns:
                continue
            for w in LAG_WINDOWS:
                row[f"last_{w}_{feat}"] = group[feat].tail(w).mean()
        for team_group, prefix in [(home_only, "home_"), (away_only, "away_")]:
            sub = team_group[team_group["team"] == team]
            for feat in BASE_STATS:
                if feat not in sub.columns:
                    continue
                for w in LAG_WINDOWS:
                    row[f"{prefix}last_{w}_{feat}"] = sub[feat].tail(w).mean()
        rows[team] = row

    return pd.DataFrame.from_dict(rows, orient="index")


def _add_venue_form(long_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Venue-specific form columns `home_last_*` / `away_last_*`.

    A row at venue V uses the team's pre-match rolling form over its V matches
    (shift(1) within venue). The other venue's column carries the rolling form
    as of the team's most recent *completed* match at that venue, forward-filled
    in team/date order. That match is strictly earlier than the row, so the
    row's own result is never used.
    """
    team = long_df["team"]
    is_home = long_df["was_home"].astype(bool)
    grouped = long_df.groupby(["team", "was_home"], sort=False)
    new_cols: dict[str, np.ndarray] = {}
    cols: list[str] = []
    for feat in BASE_STATS:
        if feat not in long_df.columns:
            continue
        for w in LAG_WINDOWS:
            pre = grouped[feat].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
            post = grouped[feat].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            for venue, mask in (("home", is_home), ("away", ~is_home)):
                carried = (
                    pd.Series(np.where(mask, post, np.nan), index=long_df.index)
                    .groupby(team, sort=False)
                    .ffill()
                )
                col = f"{venue}_last_{w}_{feat}"
                new_cols[col] = np.where(mask, pre, carried)
                cols.append(col)
    return pd.concat([long_df, pd.DataFrame(new_cols, index=long_df.index)], axis=1), cols


def build_rolling_form(matches_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns a team-match long frame with rolling-form columns:
    - `last_{3,5,10}_{stat}`: overall form (all matches, home or away)
    - `home_last_{3,5,10}_{stat}`: form from that team's home matches. On home
      rows it is the pre-match value; on away rows it is the form after the
      team's latest completed home match (NaN before their first home match)
    - `away_last_{3,5,10}_{stat}`: same, with the roles of home and away swapped

    Overall columns use shift(1), so a row's features never include that row's
    own match outcome. Venue columns carry only results from strictly earlier
    matches.
    """
    long_df = to_team_perspective(matches_df)

    long_df, overall_cols = _add_rolling(long_df, ["team"], prefix="")
    long_df, venue_cols = _add_venue_form(long_df)
    return long_df, list(overall_cols) + venue_cols


_RESULT_LETTER = {3: "W", 1: "D", 0: "L"}


def recent_form(matches_df: pd.DataFrame, team: str, n: int = 5) -> list[str]:
    """This team's last `n` results as `["W","D","L",...]`, most recent
    first — for a form strip on the fixture card/drawer, not a model
    feature."""
    long_df = to_team_perspective(matches_df)
    team_rows = long_df[long_df["team"] == team].sort_values("date")
    recent = team_rows["points"].tail(n).iloc[::-1]
    return [_RESULT_LETTER[int(p)] for p in recent]
