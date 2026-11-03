# Player-vs-opponent effect: findings

**Pre-registration:** `docs/player_opponent_effect_preregistration.md` (committed before any analysis code, with one amendment on the model class, also committed before any run).
**Raw results:** `docs/player_opponent_effect_results.json`.
**Code:** `src/pl_predictor/evaluate/player_opponent_effect.py`, tests in `tests/test_player_opponent_effect.py`.

## Answer

No. Across three walk-forward test seasons, a player's history against a specific
opponent (shrunk to his overall rate and the opponent's weakness) did not beat the
control, which is production features plus the opponent-weakness term. No comparison
passes. The null is informative about effects larger than roughly 0.0001 to 0.0002
nats per appearance. It does not rule out smaller effects.

## Results (test rows 2023-24 to 2025-26, n = 34,460 appearances, 1,140 fixtures)

Difference = candidate minus control, log loss per row (negative is better).
Interval = fixture-clustered paired bootstrap, 98.75% (Bonferroni over 4).

| comparison | mean diff | 98.75% CI | Brier diff | ECE diff | sign stable in all folds | PASS |
|---|---|---|---|---|---|---|
| A, scorer (EB matchup) | -0.000013 | [-0.000064, +0.000039] | -0.000003 | -0.00018 | yes | no |
| B, scorer (opponent tier) | +0.000054 | [-0.000129, +0.000225] | -0.000002 | -0.00010 | no | no |
| A, goal involvement | +0.000046 | [-0.000050, +0.000139] | +0.000008 | +0.00027 | no | no |
| B, goal involvement | -0.000038 | [-0.000172, +0.000098] | -0.000008 | -0.00004 | no | no |

Descriptive, not used for the decision:

- **Opponent weakness helps.** Control (production plus `log f_opp`) beats plain production
  by 0.00106 nats per row for the scorer target, and by 0.00196 for involvement.
  This is the signal the matchup terms have to beat, and it is much larger than any matchup effect.
- The matchup feature has almost no rank association with the outcome in test rows:
  Spearman 0.016 (A, scorer), 0.049 (B, scorer), 0.012 (A, involvement), 0.048 (B, involvement).

## Counts and effective sample

- Test appearances: 34,460. With at least one prior meeting against the same opponent: 29,723.
- Test appearances with 3 or more prior meetings: 18,416.
- Distinct player-opponent pairs in test: 14,556. Pairs with 3 or more prior meetings: 8,171.
- Rows with a non-zero matchup feature: about 29,900 (A) and 33,700 (B), spanning 1,140 fixtures.

**The brief expected few pairs with 3 or more meetings. The data say otherwise.** Six
seasons of history give a regular about 3 or more appearances against most clubs, because
each club is met home and away every season. Thin data is therefore not the reason for the
null. The matchup signal itself is weak.

## Why the matchup term is shrunk away

The training criterion chose the largest pre-registered pseudo-minute value
(kappa = 4320) in every fold, for every candidate. The grid was not extended after
seeing this, because that would be a change after the results. In plain terms, on
training seasons the best available weight on a player's history against the same
opponent is close to zero. This is the main reason the candidates collapse to the prior.
It is also a boundary hit, so a wider grid might behave differently. That is a question
for a new, pre-registered test, not a reason to change this one.

## Power

Fixture-clustered standard error of the mean difference is about 0.00002 (A, scorer)
to 0.00008 (B, scorer). The minimum detectable effect is roughly 0.00006 nats per row
for A (scorer), 0.0001 to 0.0002 for the others. An effect near the size of the
opponent-weakness term (0.001) would have been detected. Smaller effects would not.

## Caveats

1. **2025-26 is not a clean season.** The earlier ratio-search run 1 read it (its
   holdout, which was consumed). A clean confirmation needs a completed season not yet
   read, for example 2026-27 once finished.
2. **The model class is a stated comparison device.** Unweighted logistic regression on
   production features, as set in amendment 1 of the pre-registration. It is not
   production's Poisson-from-rates formula.
3. **xG is absent in 2020-21 and 2021-22.** Those seasons were history-only. Training
   starts in 2022-23, so the first test fold has a single training season.
4. **Opponent tiers use fixed cut points** (0.9 and 1.1 of league-relative goals conceded).
   Other cuts, or style-based clusters, were not tested.
5. **Second run, descriptive fields only.** The results were first computed without the
   control-vs-production and z-feature diagnostics. Those were added and the seeded study
   re-run. Every pass decision was identical in both runs.
6. Test rows are resampled by fixture. Player-level correlation across seasons is not
   modelled, so intervals are approximate.

## Reproduce

```
PYTHONPATH=src python -c "from pathlib import Path; from pl_predictor.evaluate import player_opponent_effect as m; m.run_study(Path('docs/player_opponent_effect_results.json'))"
PYTHONPATH=src pytest -q tests/test_player_opponent_effect.py
```
