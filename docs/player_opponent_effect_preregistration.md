# Player-vs-opponent effect: pre-registration

**Registered:** 2026-10-31, before any code for this question was run against the data.
**Branch:** `pl-player-opponent-effect`. **Raw results:** `docs/player_opponent_effect_results.json`.
**Implementation:** `src/pl_predictor/evaluate/player_opponent_effect.py`.

## Question

Does a player's history against a **specific** opponent predict his anytime-goal
or goal-involvement probability beyond (a) his overall rate and (b) the
opponent's overall weakness? Example: Palmer against Manchester United, beyond
Palmer's season-to-date rate and Manchester United's general goals-conceded rate.

Targets (binary, per appearance with minutes > 0):

- **Scorer:** goals >= 1.
- **Involvement:** goals + assists >= 1.

## Data and folds

- Source: cached FPL player-gameweek history (`fpl_history`), seasons 2020-21 to 2025-26.
  2026-27 is excluded (in progress, and the last full season is the sensible
  cut for this question).
- **xG is absent in 2020-21 and 2021-22** (no `expected_goals` column). Zero-filled
  xG would be a column of fake zeros in the baseline, so those two seasons are
  **history only**: they supply prior matches for the lagged features but are
  never training or test rows.
- Test rows: 2023-24, 2024-25, 2025-26. Each fold trains on every earlier
  training-eligible season from 2022-23. Calibration season = last training season
  (same convention as `player_ratio_search`).
- **Caveat on 2025-26:** that season was read by the run-1 holdout of the
  player ratio search (see `docs/player_ratio_search_preregistration.md`,
  Amendment 1). It is not a clean confirmatory season here. The result must say so.
  A clean confirmation needs a completed season not yet read, for example 2026-27.
- Universe: appearances with `minutes > 0`. Opponent is the other side of the
  fixture, canonicalised with `to_canonical(..., "fpl")`.

## Strict lag

All history features use only matches with **strictly earlier kickoff time**
(ties at the same kickoff are excluded). Global chronological order, not reset
per season. A unit test checks that changing the current row's own goals and
assists does not change any of its own features.

## Features

Let `GC_o`, `n_o` be the opponent's prior goals conceded and prior matches
(team-matches, all seasons in the data, strictly earlier). Let `L` be the league
mean goals per team-match over all strictly earlier fixtures.

1. **Opponent weakness** `f_o = ((GC_o + m0*L)/(n_o + m0)) / L`, with `m0 = 10`.
   Multiplicative; 1 means average.
2. **Player overall rate** `pi_p = (G_p + a*r0)/(M_p + a)`, prior goals `G_p` and
   prior minutes `M_p` over all earlier appearances, `a = 900` minutes, and
   `r0` the league goals per player-minute over earlier appearances.
3. **Matchup prior** `pi_pair = pi_p * f_o`.
4. **Opponent tier** (for candidate B): from `f_o` at the time of each appearance.
   Tier 0 if `f_o < 0.9`, tier 1 if `0.9 <= f_o <= 1.1`, tier 2 if `f_o > 1.1`.
   Unknown if `n_o < 10` (the appearance joins no tier group).
5. **Shrunk matchup rate (candidate A)**
   `r_A = (G_pv + kappa*pi_pair)/(M_pv + kappa)`, with `G_pv`, `M_pv` the player's
   prior goals and minutes against the same opponent (empirical-Bayes shrinkage
   to the prior).
6. **Shrunk cluster rate (candidate B)**
   `r_B = (G_pc + kappa*pi_pair)/(M_pc + kappa)`, with `G_pc`, `M_pc` the player's
   prior goals and minutes against opponents of the same tier label as of that
   appearance. Unknown-tier rows get `z_B = 0`.
7. **Candidate features** `z_A = log(r_A / pi_pair)`, `z_B = log(r_B / pi_pair)`.
   They are 0 when there is no matchup history, so the feature shrinks to the prior.
8. `kappa` is in minutes. Grid `{90, 270, 540, 1080, 2160, 4320}`. Chosen **per
   fold, on training rows only**, as the value minimising binary log loss of
   `p = 1 - exp(-90 * r)` against the target. No test row is read to choose it.
9. Goal involvement versions (candidates A_i, B_i) use goals + assists in place
   of goals, and the involvement target in place of the scorer target.

## Models and arms

- **Production baseline (P):** the production feature set used in the ratio
  search, `models/player_goals.RATE_FEATURES` plus `threat_last10`,
  `creativity_last10`, the lineup and minutes features, and position dummies.
- **Control (C0), the primary comparator:** P plus `log(f_o)`. This is the
  "overall rate plus opponent weakness" bar the question must clear.
- **Candidate arms:** C0 plus one of `z_A`, `z_B`.
- Model: the same balanced logistic regression, Platt-calibrated on the last training
  season (`player_ratio_search._binary_predict`). Every design column, baseline and
  candidate, is winsorised at the training-fold 0.5th and 99.5th percentiles
  identically in each arm.

Four candidate comparisons:

| id | feature | target |
|---|---|---|
| A_s | `z_A` | scorer |
| B_s | `z_B` | scorer |
| A_i | `z_A` (involvement) | involvement |
| B_i | `z_B` (involvement) | involvement |

## Metrics and pass rule

- Primary metric per row: binary log loss. Difference `d = LL(candidate) - LL(C0)`
  (negative means the candidate is better).
- Resampling unit: **fixture (cluster)**. Rows in one match share the opponent and
  team, so row resampling would understate uncertainty. Paired bootstrap over fixtures,
  2000 resamples, seed `20261031`.
- Family of four comparisons, Bonferroni: the interval level is
  `1 - 0.05/4 = 98.75%` (percentile method).
- **Pass** requires all of:
  1. The upper end of the 98.75% CI for the mean `d` is below zero.
  2. Mean `d` is negative in each of the three test folds (sign stability).
  3. Nothing else is worse: mean Brier difference and expected calibration error
     difference (candidate minus C0) are both `<= 0`.
- Each candidate is also reported against plain P (no opponent-weakness term).
  This is descriptive only and is not used for the pass decision.

## Counts and power (reported, not used to pass)

- Number of distinct player-opponent pairs with `>= 3` prior meetings, among test
  rows, and the test rows that have at least one prior meeting.
- Effective sample: test rows with `z != 0` and the number of distinct fixtures they
  span. A fixture-clustered standard error is reported.
- Minimum detectable effect is reported as roughly `2.8 * SE` of the mean `d`.

**Power statement.** A null or inconclusive result does **not** show that opponent
history has no effect. A typical player meets a given club only a few times in
six seasons, so the matchup signal is thin and noisy. The report must state this
and must not present a non-pass as evidence of no effect.

## Decisions fixed in advance

- No change to any serving path, model, or production feature.
- Research module only: `src/pl_predictor/evaluate/player_opponent_effect.py`.
- Seasons 2020-21 and 2021-22 are never used as rows. 2026-27 is excluded.
- The 2025-26 caveat above is reported in the findings, not hidden.
