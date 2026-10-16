# Model card: Lock-In draft projections

**What it predicts.** For each draftable NBA player and the upcoming season: minutes per
game, per-minute rates for points, rebounds, assists, steals, blocks, turnovers and made
threes, share of team games played, and a bonus-points expectation. Season-level FP/G and
games-share intervals (P10, P50, P90). Every fantasy quantity (FP/G, Lock-In weekly value,
season value, value over replacement) is derived from these by the scoring engine and the
game simulator; none is learned directly.

**Intended use.** Ranking players during a Sleeper Lock-In draft, in combination with
roster context and survival odds. It is not a market model: ADP is deliberately excluded
from the features so the projection can disagree with the market.

## Data

- nba_api regular-season player and team game logs, 2013-14 to 2025-26 (about 330k
  player-games); player bio (age, height, draft slot) per season; draft classes; the
  2026-27 schedule.
- Sleeper players file (positions, team, injury status, search rank) and season stats
  (technical and flagrant fouls).
- Training rows: one per player-season for target seasons 2016-17 to 2025-26 (6,494 rows).
  The pool is everyone with a game in either of the two prior seasons, including players
  who went on to miss the season (kept as zero-games outcomes).

## Leakage controls

- Every feature is computed from an `AsOf` view (games strictly before opening night).
  Perturbation test: scrambling, dropping and inflating every later row leaves the
  features unchanged.
- Bio rows for the target season are never read (their existence reveals that he played).
- Preprocessing (imputation, scaling) is fitted inside each fold; rolling-origin
  validation by season; the test season is evaluated once with all choices frozen.
- Known approximation: historical rows use each player's first team of the season as
  the team at the cut-off.

## Models

| Target | Model |
|---|---|
| Minutes, per-minute rates, games share, bonus FP | LightGBM (shallow, monotone-constrained) blended with ridge; blend weight per target chosen on validation |
| Direct FP/G (cross-check, explanations) | Same blend |
| Season FP/G interval | LightGBM quantiles of the residual around a Marcel-style anchor, re-centred on the point projection, conformally calibrated |
| Games-share interval | LightGBM quantiles, conformally calibrated |
| Game-level distribution | Resampled standardised residual vectors from the player's own recent games (similar players' when fewer than 40), re-centred on the projection |
| Rookies | Draft-slot baseline: first-year outcomes by pick bucket |

Per-minute rates are shrunk toward a height-bucket prior by empirical Bayes; noisier stats
(steals, blocks) are shrunk harder.

## Evaluation (FP/G, top 200 draftable, players with 10+ games)

| Split | Components MAE | Marcel MAE | Gain (95% CI) | 80% interval coverage |
|---|---|---|---|---|
| 2021-22 | 2.58 | 2.96 | +0.38 (+0.15, +0.61) | 80% |
| 2022-23 | 2.39 | 2.94 | +0.55 (+0.34, +0.75) | 78% |
| 2023-24 | 2.28 | 2.70 | +0.42 (+0.22, +0.64) | 81% |
| 2024-25 | 2.18 | 2.54 | +0.36 (+0.14, +0.57) | 87% |
| **2025-26 test** | **2.64** | **2.90** | **+0.26 (+0.05, +0.47)** | **80%** |

Season-value rank correlation with realised value (Spearman, top 200) is 0.65 in
validation and 0.51 in the test season, against 0.86 and 0.84 for FP/G alone: games
played is the harder half of the problem.

**Lock-In layer.** On held-out 2025-26 player-weeks, projected Lock-In value beats mean
FP/G as a predictor of the realised locked score (MAE 6.46 vs 7.16). It is close to
unbiased when the player plays every game in the week (+0.36) and optimistic in weeks with
a mid-week absence (+6.8), because availability is modelled week by week.

## Not yet validated against the market

The design's market tests (Spearman against ADP's, regression of realised value on ADP and
projection together, draft-level P(top 6) against ADP drafting) need historical ADP
snapshots dated before each draft. They are scoped for V4.
