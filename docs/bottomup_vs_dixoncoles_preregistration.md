# Bottom-up player-based team lambdas vs Dixon-Coles: pre-registration

**Registered:** 2026-10-31, before any code in `src/pl_predictor/evaluate/bottomup_vs_dixoncoles.py` was run against the data.
**Branch:** `pl-bottomup-vs-dixoncoles`. **Raw results (to be written):** `docs/bottomup_vs_dixoncoles_results.json`. **Findings (to be written):** `docs/bottomup_vs_dixoncoles_findings.md`.
**Implementation (research only, not wired into serving):** `src/pl_predictor/evaluate/bottomup_vs_dixoncoles.py`.

## Question

Can a bottom-up team goal expectation, built from the players who are expected to play, beat or improve the current Dixon-Coles match model on 1X2 and over/under 2.5, either alone or blended with it?

## The three arms

1. **DC (baseline).** The repo's current model: `models/scoreline.fit_dixon_coles` (xi = 0.0018) and `scoreline.predict_fixture`, fitted on football-data.co.uk results strictly before the match date.
2. **BU (bottom-up).** Team lambda = sum over expected players of `rate_p * E[minutes_p] / 90`, times an opponent concession factor, times a home/away factor. Home and away lambdas go into `pb.models.create_dixon_coles_grid(lambda_home, lambda_away, rho=rho_DC, max_goals=10)`, where `rho_DC` is the rho of the DC model refitted on the same date. The same scoreline grid machinery therefore produces 1X2, correct score and totals.
3. **BL (blend).** Probability blend `p = w * p_BU + (1 - w) * p_DC`, applied to 1X2 and over/under 2.5 (linear, so equal to a grid mixture for these markets).

## Lineup assumptions (reported separately, never pooled)

- **(i) ACTUAL XI (label: "known about 1h before kickoff").** Players who appear in the actual match record are included. Starters (`starts == 1` in the actual fixture) get E[minutes] = 90. Players who came on from the bench get E[minutes] = the player's prior mean minutes in bench appearances (fallback 20). Players who did not play are excluded. This is an oracle for who starts. The sub-entry minutes are estimated, not realised. The realised minutes are NOT used.
- **(ii) PREDICTED lineup (label: "known days before").** The squad is every player with minutes > 0 in the team's last up to 5 prior fixtures. E[minutes_p] = sum of p's minutes across those fixtures / number of those fixtures. Only information strictly before kickoff is used.

## Player rate (identical in (i) and (ii))

Goals per 90 with shrinkage toward the position mean:

`rate_p = (G_p + K * mu_pos) / (M_p / 90 + K)`, with **K = 5** (90-minute equivalents, fixed, not tuned).

- `G_p`, `M_p` are the player's goals scored and minutes over all prior FPL rows of that name (identity by `name`, since `element` ids are season-specific). Kickoff strictly before the match.
- `mu_pos` is the pooled goals per 90 for the same position label over all prior rows (kickoff strictly before). Position is the player's position label from their most recent prior row, or from the current row if none.
- Own goals are excluded, as in FPL `goals_scored`. Penalties are included, as in the goal totals used by DC.

## Opponent and home factors

- Opponent concession factor: `f_O = ((GA_O + k * mu_GA) / (n_O + k)) / mu_GA`, where `GA_O` and `n_O` are goals conceded and matches for opponent O over football-data matches with date strictly before the match date. `mu_GA` is the league mean goals per team-match over the same window. **k = 10** (fixed).
- Home factor: `h = sqrt(mean home goals / mean away goals)` over the same window. Home lambda is multiplied by `h`, away lambda by `1/h`.
- Team lambda = (sum of player contributions) * f_O * (h or 1/h).

## Data and universe

- Results: football-data.co.uk cache, seasons 2014-15 to 2025-26 (the `YYYY-YYYY.csv` files only; the duplicate `YY-YY.csv` files are ignored).
- Players: FPL merged-GW cache. Usable seasons 2016-17 to 2025-26. The 2026-27 season is in progress and excluded.
- **Team identity exists only from 2020-21 onward.** The FPL cache has no `team` for 2016-17 to 2019-20 (verified: `team` is NaN for every row in those seasons). Rates use name-based history from 2016-17 (no team needed). Squads and team-match history need team names, so **bottom-up predictions exist only for 2020-21 onward.** The 2010-2019 extension is not possible from this cache and is not attempted.
- FPL fixtures are joined to football-data matches on (season, canonical home, canonical away). The FPL fixture home side is the row with `was_home == True`. The unjoined count is reported.
- **Matches with no prior team fixture** (for example 2020-21 gameweek 1, where no in-cache history exists for either side) are excluded from **all arms**, and the count is reported.

## Folds and walk-forward (strict)

- Seasons 2020-21 to 2025-26. **2020-21 is a fit-only fold** (used to fit blend weights for 2021-22). **Test seasons: 2021-22 to 2025-26** (five seasons).
- DC is refitted for each distinct match date, on football-data matches with date strictly before that date. BU rates use FPL rows with kickoff strictly before the match. Opponent and home factors use football-data matches with date strictly before the match date.
- No season-level or within-season retraining asymmetry: DC gets the same date-level refresh the BU arm gets.

## Blend weight

- Grid `w in {0.0, 0.1, ..., 1.0}`, chosen by minimum mean RPS over all out-of-sample rows from **earlier folds only** (2020-21 onward, excluding the current test season). Fitted separately for each lineup assumption. Ties go to the smaller `w`.
- Blend rows for a test season use only predictions made before that season.

## Metrics (per match)

- **Ranked probability score (RPS)** on 1X2, ordered H, D, A: `RPS = 0.5 * sum_{k=1}^{2} (cumP_k - cumO_k)^2`.
- **1X2 log-loss**: `-log p(actual outcome)`, probabilities clipped to [1e-15, 1].
- **Over/under 2.5 Brier**: `(p_over - 1{total goals > 2.5})^2`.

Comparisons are paired differences `arm - DC`. Negative is better for all metrics.

## Inference

- Paired bootstrap over matches, **2000 resamples, seed 20261031**, on the pooled test set (2021-22 to 2025-26, matches where all arms are defined).
- Four comparisons form one family: {BU, BL} x {(i), (ii)}. Bonferroni over that family: each comparison uses the **1 - 0.05/4 = 98.75% percentile interval** (0.625th to 99.375th percentile).
- Standard error and minimum detectable effect (2.8 x SE, roughly 80% power at 5%) are reported for each comparison.

## Ship rule (decided now)

An arm may be called an improvement only if **all** of the following hold for that lineup assumption:

1. The 98.75% bootstrap interval for the **RPS** difference lies entirely below zero.
2. The **1X2 log-loss** and **O/U 2.5 Brier** point differences are both <= 0 (nothing else worse).

If any condition fails, the result is reported as "no demonstrated improvement." An in-sample or single-season gain is not a ship claim.

## Injuries and rotation subgroup (exploratory, not multiplicity-corrected)

The FPL cache does not carry injury or availability flags, so injuries cannot be observed directly. Two proxies are defined from the **actual** match record and used only to classify matches, never as model inputs:

- **Rotation:** for either side, 3 or more actual starters differ from that team's previous fixture starting XI.
- **Absence:** for either side, 2 or more players who started at least 3 of the team's last 5 fixtures did not start this match.

A match is "disrupted" if either proxy holds. Each arm's gain (RPS difference, with its bootstrap interval) is reported for disrupted and non-disrupted matches separately. The report states what fraction of the pooled gain comes from disrupted matches. This is descriptive only.

## Power statement (to be filled in the findings)

The expected effect size is small, and the test set is about 1,900 matches across five seasons. The findings will state the observed SE and MDE and say plainly whether the study can detect effects of the size that would matter.

## Deviations

Any deviation from the above is recorded in the findings file with its reason and a note that it was made after registration.
