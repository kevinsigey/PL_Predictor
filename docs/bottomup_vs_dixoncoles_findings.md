# Bottom-up player lambdas vs Dixon-Coles: findings

**Pre-registration:** `docs/bottomup_vs_dixoncoles_preregistration.md` (committed before any run).
**Raw results:** `docs/bottomup_vs_dixoncoles_results.json`. **Harness:** `src/pl_predictor/evaluate/bottomup_vs_dixoncoles.py` (research only, not wired into serving).

## Verdict

**No ship.** The bottom-up arm is significantly worse than Dixon-Coles under both lineup assumptions. The blend is indistinguishable from Dixon-Coles and does not meet the rule (its RPS point difference is positive, and its interval includes zero).

## Scope actually covered

- Test matches: **1,892** Premier League matches, 2021-22 to 2025-26 (five walk-forward seasons). 2020-21 is a fit-only fold for the blend weights.
- Excluded: 20 matches with no prior in-cache team fixture (2020-21 gameweek 1), applied to all arms.
- **Not 2010-2024.** The FPL cache has no team names for 2016-17 to 2019-20 (`team` is absent in those files), so bottom-up squads cannot be built before 2020-21. Football-data results were available from 2014-15, and DC uses all of them.
- The 2026-27 season is in progress and excluded.
- Dixon-Coles is refitted for every distinct match date, on results strictly before that date. The bottom-up arm uses the same date-level refresh.
- Injury data is **not** in the cache. The "injury/rotation" analysis uses two proxies from the actual record (see below).

## Pre-registered results (paired differences vs DC; negative is better)

Bootstrap: 2,000 resamples, seed 20261031. Intervals are 98.75% (Bonferroni over the four comparisons).

| Arm | Lineup | RPS diff [98.75% CI] | 1X2 log-loss diff [CI] | O/U 2.5 Brier diff [CI] |
|---|---|---|---|---|
| BU | (i) actual XI | **+0.0101** [+0.0060, +0.0145] | +0.0294 [+0.0148, +0.0439] | +0.0070 [0.0000, +0.0144] |
| BL | (i) actual XI | 0.0000 (weight 0 every season) | 0.0000 | 0.0000 |
| BU | (ii) predicted | **+0.0055** [+0.0021, +0.0090] | +0.0165 [+0.0033, +0.0288] | +0.0042 [-0.0007, +0.0091] |
| BL | (ii) predicted | +0.0001 [-0.0003, +0.0005] | -0.0008 [-0.0059, +0.0016] | -0.0001 [-0.0006, +0.0004] |

Baseline DC means: RPS 0.2018, 1X2 log-loss 0.9828, O/U 2.5 Brier 0.2440.

Blend weights chosen on earlier folds, (ii): 0.0, 0.1, 0.2, 0.1, 0.0 for 2021-22 to 2025-26. Under (i) the weight was 0 in every season.

**Reading the table:**
- Bottom-up (ii) is worse than DC on every metric. Its RPS and log-loss intervals exclude zero on the worse side.
- The blend gains nothing. Its RPS point estimate is slightly worse, so the "nothing else worse" condition fails, and every interval includes zero.
- Under (i) the bottom-up arm is worse still, which is the opposite of what an oracle lineup should give (see the calibration note below).

## Calibration and signal (diagnostics, not pre-registered)

- Mean team lambda vs actual goals (test matches): predicted lineup (ii) gives home 1.55 / away 1.26, against actual 1.56 / 1.33. Calibration is close.
- **Actual XI (i) overcounts minutes.** Starters are set to 90 minutes and bench entrants are added on top, so each team's expected minutes total about 1,090 against a nominal 990. Its lambdas run about 9% high (home 1.75, away 1.40).
- **Post-registration sensitivity arm `i_norm`** (not registered, excluded from the ship rule): (i) rescaled to 990 expected minutes per team. It still gives RPS +0.0106 [+0.0069, +0.0145] versus DC. The overcount does not explain the gap. The lineup information does not help once the player-goal signal is this weak.
- Signal: the home-minus-away bottom-up lambda correlates 0.40 with the actual goal margin, against 0.44 for Dixon-Coles. The bottom-up model is informative, just less so, and it is not broken.

## Injuries and rotation subgroup (exploratory)

Proxies, defined from the actual match record and used only to classify matches:
- **Rotation:** 3 or more actual starters differ from that team's previous fixture.
- **Absence:** 2 or more regular starters (started 3 of the last 5) did not start.

These proxies are loose. About **70%** of test matches are flagged, which leaves only 571 unflagged matches, so this split has little power.

- Bottom-up (ii) is worse in both groups: about +0.0074 RPS in flagged matches and +0.0013 in unflagged. The gain attributable to disrupted matches is therefore not defined, because there is no gain overall.
- The blend shows no difference in either group (flagged +0.0002, unflagged -0.0002, both intervals include zero).

Whether a real injury feed would change this is untested.

## Power

- The paired RPS standard error for the blend is about 0.0002, so the minimum detectable RPS difference (2.8 SE) is about **0.0006**. That is roughly 0.3% of the baseline RPS, so the study can rule out an improvement of practical size for the blend.
- The bottom-up deficit is large relative to that noise and is detected clearly.
- The study covers five test seasons and does not test the 2010-2019 period.

## Deviations and post-registration additions

1. **`i_norm` sensitivity arm** was added after the (i) calibration issue above was found. It is reported separately, is not part of the ship rule, and does not change any registered number. The registered arms (i), (ii) produced identical numbers before and after this addition.
2. The lambda calibration and signal diagnostics are post-registration and descriptive only.
3. Where the pre-registration said "the same DC scoreline machinery", the implementation uses penaltyblog `create_dixon_coles_grid` with the refitted rho, as written. Bottom-up 1X2 and O/U come from that grid, and the blend is linear on those probabilities.

## Recommendation

Do not wire the bottom-up lambda into serving. A bottom-up model built from player goal rates alone is weaker than Dixon-Coles. If this line is revisited, the next experiment should add team-level defensive information to the player lambdas, or use xG-based player rates (FPL rows carry `expected_goals` in the cache; coverage per season not checked here), and be registered before running.
