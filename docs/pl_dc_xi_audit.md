# PL-1 Dixon-Coles audit — xi tuning + crash fix (2026-10-23)

## Claim verified (plan 2026-10-23)
- 8-season multi-season RPS (2017-18..2024-25): mean 0.2083 vs published DC 0.199 → underperforms (~0.009).
- Tick sources: evaluate/walk_forward.py + docs/pl_venue_split_audit.md (from PR #101).

## Crash — root cause
`tune_dc_bp_xi.py` crashes with "negative goal-matrix errors on some folds".
Cause: extreme lambda values in the grid (0.012, 0.02) drive tau/rho to combinations where penaltyblog's score matrix has negative cells. 
Fix (in scoreline.py): `fit_dixon_coles` catches a ValueError whose text contains "negative" or "goal-matrix", restarts the optimiser once from rho=0, and refits. A second failure propagates the ValueError unchanged. No custom RuntimeError is raised. The fitted rho is not clipped afterwards. ponytail: single retry; upgrade path = expose the rho bound to the tune grid.

## Evaluation (requested settings, paired bootstrap B=5000)
Grid: xi in {0.0010, 0.0018, 0.0030}; with/without low-score correction; vs current 0.0018.

### Measured (current baseline)
- 0.0018 (current): RPS 0.2083; CI vs self = [0,0] (baseline)

### Hypotheses (NOT measured; no bootstrap interval has been computed)
- 0.0010: hypothesised RPS near 0.2072, extrapolated from the trend in the quick run. Unverified.
- 0.0030: hypothesised worse than 0.0018 on the same trend. Unverified.
- Low-score correction: not implemented in `quick_tune_xi.py`. The earlier idea of dropping fixtures with <3 total goals selects on the observed outcome, so it is withdrawn. Any correction must act on predicted probabilities and be scored on the same validation rows as the baseline.

### Decision
No shipping decision is made in this document. Any change from xi=0.0018 depends on the pending paired match-level bootstrap (B=5000) against 0.0018. Until that run is recorded here, keep 0.0018 and treat every figure above as a hypothesis.

## CI / commands
`python -m pl_predictor.evaluate.tune_dc_bp_xi` now exits cleanly (no crash); full multi-season walk-forward exit code recorded when run.

## Notes
- Model artifacts in `models/` untouched (research PR; no artifacts changed).
- Retraining of served boosters is SEPARATE (see PR #102 note — 66 venue-split columns now real; boosters must be refit before shipping any new feature impact).
- Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
 🤖 Generated with [Claude Code](https://claude.com/claude-code)
