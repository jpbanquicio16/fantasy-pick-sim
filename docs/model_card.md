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
  player-games); player bio (age, height, draft slot) per season; draft classes with each
  pick's background (college, or international and other clubs); the 2026-27 schedule.
- Sleeper players file (positions, team, injury status, search rank; birth date, height and
  weight for rookies, who have no NBA bio yet) and season stats (technical and flagrant fouls).
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
  the team at the cut-off (live: Sleeper's current team). Role features use only last
  season's games and that roster; rookies enter at draft-slot minutes estimated from
  earlier draft classes.

## Models

| Target | Model |
|---|---|
| Minutes, per-minute rates, games share, bonus FP | LightGBM (shallow, monotone-constrained) blended with ridge; blend weight per target chosen on validation |
| Team minutes budget | Post-model: each team's expected minutes (rookies included) pulled 25% of the way to the league's minutes per team game, by elasticity m(1 - m/40); strength chosen on validation |
| Direct FP/G (cross-check, explanations) | Same blend |
| Season FP/G interval | LightGBM quantiles of the residual around a Marcel-style anchor, re-centred on the point projection, conformally calibrated |
| Games-share interval | LightGBM quantiles, conformally calibrated |
| Game-level distribution | Resampled standardised residual vectors from the player's own recent games (similar players' when fewer than 40), re-centred on the projection |
| Rookies (V5) | Ridge regressions on draft-day inputs (draft slot, monotone; age; size; college or club; the team's vacated minutes and record), plus the 20 most similar past rookies for the ranges, the game-shape donors and the explanation. The draft-slot baseline (first-year outcomes by pick bucket) is kept as the comparison and for rookies' minutes in the veterans' role features |

**Per-minute rates (V5): hierarchical shrinkage.** Each player's 5/3/2-weighted rates are
shrunk toward a prior whose mean depends on his height bucket, age, experience and draft slot
(a gamma-Poisson model per stat, with the prior's regression and strength fitted by marginal
likelihood on the player-seasons before the cut-off; `features/rates.py`). It replaced the
height-bucket prior after a comparison on the validation seasons (`draftai rates-experiment`,
`rate_prior_report.md`): FP/G MAE 2.331 against 2.337, better in all four seasons but by
0.005, which is noise-sized; the test season went the other way (2.61 against 2.59). As a rate
estimate it can be much better: with the prior's strength put on the scale of the weighted
evidence (the bucket prior counts a minute of last season five times), next season's per-36
error falls 12% for players with few minutes and 5% for second-year players. But that version
made the projections slightly worse (2.340): the downstream models already learn the shrinkage
from minutes, age, experience and draft slot, so a cleaner rate feature adds nothing.

**Team moves and role changes.** Features include a new-team flag, last season's team pace
and net minutes opportunity: minutes that left the player's team since last season, minus
minutes that arrived (trades and signings at last season's minutes, rookies at draft-slot
minutes) and minutes of teammates returning from a long absence (15+ games missed), each
relative to the league's average team that offseason, unweighted (`opp_team_mpg`) and
weighted toward his height bucket (`opp_pos_mpg`: same bucket 1, adjacent 0.5, guard-big 0),
and his elastic share of the latter among teammates at his positions (`opp_own_mpg`), so a
star is credited with little of his team's opportunity. His own arrival or return is excluded. Retirements are not counted as departures (history
cannot tell them from season-long injuries without looking ahead).

The team minutes budget then makes the arithmetic add up. Its strength was chosen on the
four validation seasons (FP/G MAE, top 200): 0 gives 2.338, **0.25 gives 2.331**, 0.5 2.338,
1.0 (a full rescale) 2.389. Proportional rescaling (the design's first proposal) was worse
still in a pre-study, because it cuts stars' minutes for the noise in their bench's
projections; the elasticity puts the change on mid-rotation players.

**Minutes overrides** (`configs/minutes_overrides.csv`, or the app) pin a player's minutes
after preseason news; the change is absorbed in full by his teammates, weighted toward his
Sleeper positions (others 0.3x). They are applied after the model, logged, and never reach
training or evaluation.

## Evaluation (FP/G, top 200 draftable, players with 10+ games)

| Split | Components MAE | Marcel MAE | Gain (95% CI) | 80% interval coverage |
|---|---|---|---|---|
| 2021-22 | 2.53 | 2.96 | +0.43 (+0.20, +0.65) | 80% |
| 2022-23 | 2.35 | 2.94 | +0.58 (+0.39, +0.78) | 77% |
| 2023-24 | 2.26 | 2.70 | +0.44 (+0.23, +0.64) | 83% |
| 2024-25 | 2.19 | 2.54 | +0.36 (+0.15, +0.56) | 88% |
| **2025-26 test** | **2.61** | **2.90** | **+0.29 (+0.08, +0.50)** | **81%** |

The test season has now been looked at five times while features were defined (three times for
the role features, once each for the V5 rate prior and rookie model); every choice was made on
validation, but its numbers are a little optimistic.

Role-change cohorts (players with 10+ games; validation mean, FP/G MAE and bias = projected
minus actual; full tables in `projection_report.md`):

| Cohort | n/season | Marcel MAE | Model MAE | Model bias | 80% coverage |
|---|---|---|---|---|---|
| Everyone | 419 | 3.06 | 2.56 | +0.53 | 80% |
| Changed teams | 116 | 3.41 | 2.82 | +0.81 | 80% |
| Returning from a long absence | 125 | 3.58 | 2.86 | +0.73 | 78% |
| Second season (V5) | 60 | 3.65 | 2.94 | +0.75 | 81% |
| Team filled 15+ min/game at his positions | 90 | 2.99 | 2.41 | +0.40 | 84% |
| Team opened 15+ min/game at his positions | 131 | 3.00 | 2.51 | +0.67 | 80% |
| Minutes fell 5+ (defined by the outcome) | 66 | 4.77 | 4.17 | +4.14 | 65% |

The opportunity signal is real but weak: players whose team opened 10+ minutes a game at
their positions changed their minutes by about 1.3 a game more than those whose team filled
10+ (+0.4 against -0.9). The model beats Marcel in every
cohort; the largest remaining error is the role loss nobody saw coming (last row), which is
what minutes overrides are for.

## Rookies (V5)

Every drafted player of each class, projected as of opening night of his first season, against
the draft-slot baseline, by rolling origin over draft classes (`rookie_report.md`; error in
first-season fantasy points over every pick, zero for those who never played):

| Classes | Rookie model | Draft-slot baseline | Gain (95% CI) |
|---|---|---|---|
| 2021 | 228 | 243 | +15 (-13, +43) |
| 2022 | 213 | 260 | +48 (+14, +83) |
| 2023 | 227 | 237 | +9 (-18, +36) |
| 2024 | 226 | 291 | +66 (+40, +93) |
| **2025 test** | **240** | **274** | **+34 (+1, +71)** |

FP/G error for rookies with 10+ games: 3.06 against 3.47 in validation, 3.07 against 3.23 in
the test class; first-round picks' season error 279 against 323. Both under-project the top
three picks a little (FP/G bias about -1.5 and -1.3); on season points the model is far less
biased there (-35 against -220), because it expects top picks to play far more of the season.
The comparables' plain average is biased low (-3.6 FP/G for top-three picks: their neighbours
include later picks), so comparables give ranges, donors and explanations, not the point.
Intervals are conformalised on the validation classes: 81% (FP/G) and 80% (games) in
validation, 81% and 80% in the test class. Inputs the design lists but the data sources lack:
college per-possession statistics.

Season-value rank correlation with realised value (Spearman, top 200) is 0.65 in
validation and 0.51 in the test season, against 0.87 and 0.84 for FP/G alone: games
played is the harder half of the problem.

**Lock-In layer.** On held-out 2025-26 player-weeks, projected Lock-In value beats mean
FP/G as a predictor of the realised locked score (MAE 6.40 vs 7.14). It is close to
unbiased when the player plays every game in the week (+0.26) and optimistic in weeks with
a mid-week absence (+6.6), because availability is modelled week by week.

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

Rerun with the V5 projections and the learning-to-rank strategies. Five seasons (2021-22 to
2025-26), your league's shape (12 teams, 14 rounds, third-round reversal, one C slot, top 6 make the playoffs). Each
season: 96 simulated leagues with your seat rotated through every slot (the greedy recommender and the
rollouts on 24 of them, 120 in all), opponents drafting around the market with the configured ADP
spread, then the real season played under Lock-In rules. Paired: same leagues, seat, opponent random
numbers and schedules.

| Strategy | P(top 6) | P(title) | Realised FP/week |
|---|---|---|---|
| Hindsight oracle (ceiling) | 1.000 | 0.834 | 261.0 |
| **Rollouts (V4)** | **0.912** | **0.302** | **236.8** |
| Distilled ranker `ltr_sim` (V5) | 0.928 | 0.278 | 238.5 |
| Fine-tuned ranker `ltr` (V5) | 0.891 | 0.287 | 237.1 |
| Greedy value + survival (V3) | 0.802 | 0.182 | 231.8 |
| Greedy VOR (= projected points here) | 0.797 | 0.205 | 231.2 |
| Draft by market ADP | 0.741 | 0.187 | 229.6 |

(The rollouts and the greedy recommender ran in 120 leagues, the others in all 480; the paired
differences below use the leagues each pair shares.)

| Rollouts vs | P(top 6) | FP/week | P(title) |
|---|---|---|---|
| Market ADP | +0.127 (+0.073, +0.184) | +5.2 (+2.7, +7.6) | +0.095 (+0.027, +0.165) |
| Greedy VOR | +0.123 (+0.075, +0.175) | +5.6 (+4.0, +7.3) | +0.089 (+0.021, +0.156) |

**Gate (design section 25): passed.** The rollouts beat market ADP and greedy VOR on P(top 6) in all
five seasons, with pooled intervals above zero, and now on P(title) too. Against the run before V5's
projections (rollouts 238.3, greedy VOR 231.8 FP/week) the differences are within what this backtest
can resolve: five real seasons, and a few early-round busts decide a season. Better projections by
MAE have never shown up as a measurable gain in drafting here.

**Confidence is roughly calibrated.** Re-scoring each rollout decision's two branches with the
real season: decisions shown at 60-70% were right 69% of the time (182 decisions), at 70-80%
69% (16); at 50-60% the recommendation won 46% (1,482): close calls really are coin flips.

**Market tests (against the proxy market).** The projection ranks realised season value better
than the market in every season (Spearman 0.59-0.71 vs 0.51-0.64; 2024-25 a tie), and in a
regression on both, its coefficient's interval is above zero every season.

**Your real 2025 LeFantasy draft, replayed.** Keeping every human pick and handing one seat at a
time to the rollouts, the rollout team out-scored the person who held that seat in all 10 seats,
by 21.8 FP/week on average. In your seat (slot 9) it scored 248.9 FP/week against your team's
227.4. Treat this as an upper bound: the league keeps a player and has five IL slots, so humans
spent picks on stashes (Haliburton went 16th) that a one-season replay scores as zeros.

The market in these simulations is a naive proxy (historical ADP does not exist), so gains
against `adp` overstate gains against a real room. Full tables, caveats and per-season results:
`data/artifacts/reports/backtest_report.md` (regenerated by `draftai backtest`, about 20 minutes).

## Learning-to-rank (V5, design section 17)

The question: can a ranker trained on realised outcomes correct the simulator's systematic
biases? `draftai ltr` (`backtest/ltr.py`, `ltr_report.md`):

- **Queries**: your picks in 24 simulated leagues per backtest season (seat rotated through every
  slot, the backtest's calibrated room, its own random numbers): 1,680 queries, each over the greedy
  recommender's top 24 candidates. Your seat followed the simulator's choice, with a random top-5
  candidate 25% of the time.
- **Labels** for every candidate: the rollout's expected final-roster value (150 worlds, every
  candidate a rollout candidate), and the realised value (take him, let greedy VOR finish your draft
  against the same opponents, score that roster with the real season).
- **Rankers**: LightGBM LambdaRank (NDCG@5, graded labels) on 29 features (Lock-In value, VOR,
  marginal roster value, survival, ADP against the pick, spread, need, round; per week and relative to
  the query's best), trained leaving one season out: `ltr_sim` on simulator labels (300 rounds),
  `ltr` continuing from it on realised labels (100 rounds, the design's plan), `ltr_real_only` on
  realised labels alone.

On the held-out seasons, against realised outcomes:

| Ranking | NDCG@5 | Realised regret (FP/week) | First choice = simulator's |
|---|---|---|---|
| Simulator (rollout over all 24) | **0.615** | **5.44** | 100% |
| `ltr` (distilled, fine-tuned on realised) | 0.598 | 5.54 | 32% |
| `ltr_sim` (distilled) | 0.601 | 5.62 | 43% |
| Market ADP | 0.602 | 5.56 | 18% |
| Greedy VOR | 0.581 | 5.86 | 33% |
| Greedy value + survival (V3) | 0.576 | 5.87 | 35% |
| `ltr_real_only` | 0.562 | 6.10 | 18% |

Paired against the simulator over 1,680 held-out queries: `ltr` regret +0.10 FP/week (95% interval
-0.10 to +0.31), NDCG@5 -0.017 (-0.025 to -0.010); realised labels alone are clearly worse (+0.66,
+0.43 to +0.90). (The intervals resample queries within seasons; queries of a season share one
realisation, so they understate the uncertainty.) **No improvement: the simulator stays.** The labels
say why: within a query the simulator's expected values and the realised ones have a rank correlation
of only 0.07 to 0.23, so a single realised season per candidate is mostly noise, and five seasons
leave little systematic bias to learn. Fine-tuning moved the ranker away from the simulator (first
choices agree 43% before, 32% after) without moving it toward the realised outcomes. The distilled
ranker is a fast surrogate (milliseconds per pick, against 1-2 s for rollouts) that agrees with the
simulator's first choice 43% of the time and loses 0.18 FP/week of realised regret against it.

**In the draft backtest** (the design's judge; each season's rankers never saw it; `ltr_report.md`):

| Paired comparison | Leagues | P(top 6) | FP/week |
|---|---|---|---|
| `ltr_sim` vs rollouts | 120 | +0.011 (-0.025, +0.049) | +1.8 (+0.1, +3.6) |
| `ltr` vs rollouts | 120 | -0.017 (-0.055, +0.018) | 0.0 (-1.8, +1.8) |
| `ltr` vs `ltr_sim` | 480 | -0.037 (-0.056, -0.020) | -1.4 (-2.4, -0.5) |
| `ltr_sim` vs greedy VOR | 480 | +0.131 (+0.108, +0.154) | +7.3 (+6.5, +8.1) |

Fine-tuning on realised outcomes makes the ranker worse (worst in 2021-22: -8.8 FP/week against
`ltr_sim`), confirming the offline result: realised labels do not correct the simulator. The distilled
ranker, by contrast, drafts as well as the rollouts it imitates (better in four of five seasons on
points, by -0.2 to +5.4 FP/week), at milliseconds per pick. A plausible reason it is no worse: its
teacher was a rollout over all 24 candidates, while the backtest's rollouts consider only the greedy
recommender's top 6. With five realised seasons, a small edge either way is not established, and the
rollouts remain the recommendation engine (they also give the calibrated confidence); the ranker is a
candidate for a faster engine or a wider rollout shortlist. In the replay of your real 2025 draft,
`ltr_sim` beat the human in all 10 seats (+16.4 FP/week; rollouts +21.8).

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
