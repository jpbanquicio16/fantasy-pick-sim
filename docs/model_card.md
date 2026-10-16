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

## Opponent model (V4)

Each opponent pick is a conditional logit over the available players:
`u = -beta(t) * (market rank among available - 1) + gamma * fills an open starting slot`,
`beta(t) = exp(c0) * t^c1`, mixed with a small share of off-board picks (uniform over the
available players). Fitted by maximum likelihood on real Sleeper drafts (`draftai opponents`;
your own picks excluded) and scored on held-out teams.

- On the one real draft available (LeFantasy 2025, 126 opponent picks) the off-board share is
  about 6.5% and improves held-out log-likelihood per pick from -4.75 to -4.39 (uniform: -6.53;
  constant beta: -4.43).
  These include keeper and IL stashes: that league keeps one player and has five IL slots.
- The need term is not identified by these drafts: with G, F and two UTIL slots almost any
  player fills an open slot until a lineup is full, after which none does, so need never varied
  among the available players at any real pick. The prior (gamma = 0.8) is kept.
- Live drafts and mocks use beta(t) calibrated so simulated drafts reproduce the configured ADP
  spread (sigma = 1.5 + 0.12 ADP; the same spread the survival odds use). The fitted model's
  beta and off-board share were measured against the backtest's proxy market, so they mix
  drafter noise with proxy error and are not carried over.

## Monte Carlo rollouts (V4)

For each of the greedy recommender's top 8 candidates, 600 simulated worlds: every player's
season drawn from the calibrated FP/G and games-share quantiles (absences as one block of
weeks), the rest of the draft simulated with the opponent model and a greedy base policy for
your later picks (highest VOR that fills an open starting slot), and the final roster scored
week by week (best lineup; one waiver claim). Common random numbers across candidates; racing
drops candidates whose paired difference to the leader is clearly negative after 200 worlds.
Confidence is P(A's roster beats B's) over the shared worlds. About 1-2 s per pick on a laptop.

Weekly lineups inside the rollouts use a vectorised greedy fill (most specific open slot first);
on real-board rosters it reaches 99.8% of the exact assignment's value (worst roster seen: 92.6%).

## Draft backtest (V4)

Five seasons (2021-22 to 2025-26), your league's shape (12 teams, 14 rounds, third-round
reversal, one C slot, top 6 make the playoffs). Each season: 96 simulated leagues with your seat
rotated through every slot (the greedy recommender and the rollouts on 24 of them, 120 in all),
opponents drafting around the market with the configured ADP spread, then the real season played
under Lock-In rules. Paired: same leagues, seat, opponent random numbers and schedules.

| Strategy | P(top 6) | P(title) | Realised FP/week |
|---|---|---|---|
| Hindsight oracle (ceiling) | 1.000 | 0.848 | 261.4 |
| **Rollouts (V4)** | **0.931** | **0.315** | **240.1** |
| Greedy value + survival (V3) | 0.926 | 0.258 | 236.2 |
| Greedy VOR (= projected points here) | 0.881 | 0.263 | 234.9 |
| Draft by market ADP | 0.753 | 0.187 | 229.9 |

Paired differences over the 120 shared leagues (95% bootstrap interval):

| Rollouts vs | P(top 6) | FP/week | P(title) |
|---|---|---|---|
| Market ADP | +0.122 (+0.069, +0.180) | +8.4 (+5.9, +11.0) | +0.100 (+0.028, +0.173) |
| Greedy VOR | +0.034 (-0.004, +0.074) | +4.7 (+2.7, +6.6) | +0.044 (-0.020, +0.116) |

**Gate (design section 25): not passed.** The rollouts beat market ADP clearly, and beat greedy
VOR in four of five seasons on P(top 6), but the pooled interval against VOR touches zero, and
2023-24 went the other way (-0.075). On points the rollouts beat greedy VOR in all five
seasons (+0.2 to +11.7 FP/week). A run before a small change to the lineup evaluator passed
narrowly (+0.044, interval +0.005 to +0.084): the result is borderline, and more leagues
(`--expensive-leagues`) would settle it. P(top 6) is close to saturated for every model-based
strategy, which is part of why it separates them poorly.

**Confidence is roughly calibrated.** Re-scoring each rollout decision's two branches with the
real season: decisions shown at 60-70% were right 61% of the time (162 decisions), at 70-80%
75% (20); at 50-60% the recommendation won 46% (1,498): close calls really are coin flips.

**Market tests (against the proxy market).** The projection ranks realised season value better
than the market in every season (Spearman 0.58-0.70 vs 0.51-0.63), and in a regression on both,
its coefficient's interval is above zero every season.

**Your real 2025 LeFantasy draft, replayed.** Keeping every human pick and handing one seat at a
time to the rollouts, the rollout team out-scored the person who held that seat in all 10 seats,
by 21.3 FP/week on average. In your seat (slot 9) it scored 247.4 FP/week against your team's
229.2. Treat this as an upper bound: the league keeps a player and has five IL slots, so humans
spent picks on stashes (Haliburton went 16th) that a one-season replay scores as zeros.

The market in these simulations is a naive proxy (historical ADP does not exist), so gains
against `adp` overstate gains against a real room. Full tables, caveats and per-season results:
`data/artifacts/reports/backtest_report.md` (regenerated by `draftai backtest`, about 15 minutes).

## Limits of the backtest

- **Market.** Historical ADP is not available, so simulated rooms draft around a naive proxy
  (last season's FP/G; rookies from the draft-slot projection; pre-season news half-priced).
  It correlates 0.73 with the real 2025 LeFantasy pick order. Real ADP is sharper, so gains
  against `adp` overstate gains against a real room.
- **Pre-season news** for past seasons comes from a short hand-maintained file
  (`configs/backtest_news.csv`), not from dated Sleeper injury reports.
- **Blend weights and conformal corrections** were chosen on the 2021-22 to 2024-25 validation
  seasons, so those backtest seasons carry a mild hyperparameter look-ahead. 2025-26 is clean.
- **Season play** uses one rule for every team: weekly lineups from in-season expectations,
  per-player lock thresholds (no moving a bench player into a passed slot), waiver claims only
  for empty starting slots, no trades. One realised season per year: decisions within a season
  share their outcome.
- **Replays of real drafts** score one season of a keeper league with IL slots as a redraft,
  which penalises the humans' stash picks.
