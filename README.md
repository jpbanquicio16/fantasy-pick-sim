# fantasy-pick-sim

A probabilistic draft assistant for **Sleeper Lock-In** fantasy basketball. It projects
every player as a distribution of per-game box scores and a games-played distribution,
turns each into a Lock-In weekly value (the expected score of the game you would
actually lock, an optimal-stopping problem), and recommends picks by marginal roster
value plus the chance each player survives to your next pick.

This repository implements phases **V0–V5** of the technical design: foundations,
projections, distributions and Lock-In, the live draft assistant, simulation (an
opponent model fitted on real Sleeper drafts, Monte Carlo draft rollouts with a calibrated
confidence, and a paired draft backtest on five past seasons plus replays of your real
drafts), and the V5 extensions: a rookie model with comparables, Bayesian (hierarchical)
per-minute rates, a learning-to-rank experiment judged in the backtest, and a FastAPI +
React front end. Offline mocks draft themselves: every other team picks with the opponent
model.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp configs/league.example.yaml configs/league.yaml   # set league_id / username / draft_slot

.venv/bin/draftai ingest      # ~1 min: 13 seasons of nba_api logs + Sleeper players (cached)
.venv/bin/draftai league      # check the settings the engine will use
.venv/bin/draftai train       # ~1 min: rolling-origin evaluation, test season, final models
.venv/bin/draftai project     # ~5 s: the draft board for the season
.venv/bin/draftai app         # live draft assistant (Streamlit), including self-drafting mocks

.venv/bin/draftai opponents   # fit the opponent model on your completed Sleeper drafts
.venv/bin/draftai mock        # terminal mock: the other teams draft themselves
.venv/bin/draftai ltr --no-backtest   # ~35 min once (cached): learning-to-rank queries, rankers
.venv/bin/draftai backtest    # ~20 min: paired draft backtest + replay of your real drafts

make web                      # build the React front end (Node 18+), then:
.venv/bin/draftai api         # FastAPI + React at http://127.0.0.1:8000 (API docs at /docs)
```

Run `nba_api` ingestion from a home connection: stats.nba.com drops cloud and datacenter
IPs. During the season, `draftai ingest --sleeper-only` refreshes eligibility, teams,
injuries and the ID crosswalk (daily is plenty); rerun `draftai project` afterwards.

Other commands:

| Command | What it does |
|---|---|
| `draftai recommend --draft-id ID --username NAME --follow` | Follow a live Sleeper draft in the terminal |
| `draftai recommend --picks "Nikola Jokić" "Victor Wembanyama"` | Offline: recommendation after the given picks |
| `draftai recommend ... --rollout` | Monte Carlo rollouts: 600 simulated drafts and seasons per candidate, with confidence (1-2 s) |
| `draftai mock` | Mock draft in the terminal; the other teams draft themselves. Enter takes the recommendation, a number takes that rank, `auto` lets the engine finish |
| `draftai mock --auto [--rollout] [--seed N]` | The engine drafts your team too; prints every team's projected points, P(playoffs) and P(title) |
| `draftai opponents` | Fit the opponent model on your completed Sleeper drafts; calibrate the live room |
| `draftai backtest [--quick] [--seasons ...]` | The V4 gate: paired draft backtest and replays of your real drafts; writes `backtest_report.md` |
| `draftai adp --username NAME` | Re-estimate ADP from your completed Sleeper (mock) drafts and fit the ADP spread |
| `draftai adp --csv adp.csv` | Use any ADP source (`sleeper_id` or `name`, `adp`, optional `sd`) |
| `draftai lockin-experiment` | The Lock-In headline experiment on the held-out season |
| `draftai rookies` | Refit and evaluate only the rookie model (seconds); writes `rookie_report.md` |
| `draftai rates-experiment` | Compare the per-minute rate priors on the validation seasons; writes `rate_prior_report.md` |
| `draftai ltr [--no-backtest \| --use-backtest-rows]` | The learning-to-rank experiment: collect labelled queries, train leaving one season out, judge offline and in the backtest; writes `ltr_report.md` |
| `draftai api [--port N]` | Serve the API and the React front end (`make web` builds it); `cd web && npm run dev` for front-end development |
| `make test` | Full test suite (offline, ~20 s) |

## Configuration

- `configs/league.yaml`: league id, username or draft slot, playoff weight. Roster
  positions, scoring, team count, rounds, draft type, third-round reversal, traded picks
  and playoff weeks are **read from the Sleeper API** when a league or draft is connected;
  the YAML values are only offline fallbacks.
- `configs/availability_overrides.csv`: manual games-share overrides for news the model
  cannot know (a season-ending injury, a return from one). Copy the `.example.csv`.
  Columns: `sleeper_id`, then `share` (absolute, 0-1) or `multiplier`, and `note`.
- `configs/minutes_overrides.csv`: expected minutes per game for preseason news the model
  cannot know (a starting job announced, a move to the bench). Copy the `.example.csv`.
  Columns: `sleeper_id`, `name` (for readability), `mpg`, `note`. Teammates absorb the change,
  mostly at his positions; the model and its evaluation are untouched. Also editable in the
  app's sidebar, which rebuilds the board (a few seconds). Every application is logged to
  `data/artifacts/board/<season>/minutes_overrides_log.csv`.
- `configs/id_overrides.csv`: manual NBA to Sleeper id matches, if the crosswalk misses one.
- `configs/train.yaml`, `configs/sim.yaml`: seasons, model parameters, simulation size, ADP spread,
  rollout size (`rollout_worlds`, `rollout_shortlist`), waiver cover in simulated seasons.
- `configs/backtest.yaml`: seasons, strategies, simulated leagues per season, the simulated room
  (`calibrated` or `fitted`).
- `configs/backtest_news.csv`: pre-season injury and suspension news for past seasons (season, name,
  share or multiplier, note), standing in for Sleeper's injury status, which has no history.
  Hand-maintained from widely reported news; check and extend it.

## Results (real data, 2013-14 to 2025-26)

**Scoring engine (V0 gate).** 661 real games scored by Sleeper reproduce to the half-point,
including every 2025-26 game of 50+ points. The design's open question is settled: **40+
and 50+ bonuses stack** (+4 for a 50-point game). Sleeper's default score also omits
technical and flagrant fouls; the engine scores them whenever the league does.

**Projections (V1 gate: pass).** FP/G mean absolute error on the top 200 draftable players:

| | Validation 2021-22 to 2024-25 (mean of 4) | Test 2025-26 |
|---|---|---|
| Last season | 2.64 | 2.97 |
| Marcel-style 5/3/2 | 2.78 | 2.90 |
| Ridge baseline | 2.74 | 2.87 |
| **Components (LightGBM + ridge blend, team minutes budget, hierarchical rate prior)** | **2.33** | **2.61** |

The component model beats Marcel in all four validation seasons, each with a paired
bootstrap 95% interval above zero (+0.36 to +0.58 FP/G), and in the test season
(+0.29, interval +0.08 to +0.50). The test season has been looked at five times while features
were defined (every choice was made on validation), so its numbers are a little optimistic.

**Uncertainty.** Conformalised 80% intervals for season FP/G cover 77-88% in validation
and 81% (top 200) in the test season; games-share intervals cover 78%.

**Team moves and reduced roles.** A player's team at the cut-off comes from Sleeper (in
history, his first game of the season). A role change reaches the projection two ways:

- *Net minutes opportunity at his positions* (features): minutes that left his team since
  last season, minus minutes that arrived (trades, signings, rookies at their draft-slot
  minutes) and minutes of teammates returning from long absences; relative to the league's
  average team, weighted toward his position group (guard, wing, big), and his own elastic
  share of it (a star takes little of his team's opportunity, a rotation player much more).
- *A team minutes budget*: each team's expected minutes, rookies included, are pulled toward
  the 241 that exist per team game, mostly from mid-rotation players (36-minute starters
  barely move). Validation chose a strength of 0.25. Pulling all the way to 241, as the design
  proposed, made FP/G worse (2.39 against 2.34): team sums of separately projected players
  are noisy, and their spread is almost uncorrelated with what those players went on to play.

Together they took FP/G MAE from 2.356 to 2.337 in validation and 2.640 to 2.594 in the test
season (before V5's rate prior): small gains. The cohort tables in `projection_report.md` show where it
stays hard: players whose team filled 15+ minutes a game at their positions are still
over-projected by about 0.4 FP/G, and players whose minutes fell by 5+ (defined by the outcome,
so a diagnostic) by 4.1. Minutes overrides (above) are the tool for news the model cannot see.
In simulated seasons (rollouts, mock projections) a player's absence hands his minutes to his
teammates in the weeks he is out, so teammates' seasons are negatively correlated; averages are
unchanged.

**Lock-In (V2 gate: pass).** On 3,870 held-out 2025-26 player-weeks, the projected Lock-In
value predicts the realised locked score better than mean FP/G (MAE 6.40 vs 7.14, gain
+0.75, 95% CI +0.55 to +0.97). In weeks where the player played every game it is close to
unbiased (+0.26), while mean FP/G understates by 3.6. The lockable premium follows the
theory: 0.40, 0.63 and 0.79 SD for 2, 3 and 4 games.

**Draft assistant (V3).** Recommendations take about 0.05 s per pick (budget: 5 s), follow a
live Sleeper draft by polling, and handle snake, linear, third-round reversal and traded picks.
Monte Carlo rollouts (V4) take 1-2 s per pick with 600 simulated drafts and seasons per candidate.

**Draft backtest (V4, rerun with the V5 projections).** Five seasons (2021-22 to 2025-26), your
league's shape (12 teams, 14 rounds, third-round reversal, one C slot, top 6 make the playoffs). Each
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

**V5: rookies.** A separate rookie model (design section 9): ridge regressions on draft slot (later
picks never project more), age, size, college or club background and the minutes his team's veterans
vacated at his positions, plus the 20 most similar past rookies for the ranges, the game shapes and
the explanation ("rookies like him: ..."). Projected as of opening night, every pick of each class,
against the draft-slot baseline (error in first-season points, zero for picks who never played):
better in all four validation classes (by 9 to 66 points per pick; two intervals above zero) and in
the test class (240 against 274; +34, interval +1 to +71). FP/G error for rookies with 10+ games:
3.06 against 3.47 in validation. Intervals cover 81% (FP/G) and 80% (games). Full tables:
`rookie_report.md` (`draftai rookies`, seconds).

**V5: Bayesian per-minute rates.** Rates are shrunk toward a hierarchical gamma-Poisson prior whose
mean depends on size, age, experience and draft slot (fitted by marginal likelihood), instead of the
height bucket's mean. Validation FP/G MAE 2.331 against 2.337: better in all four seasons, so it was
adopted by the rule set beforehand, but by a noise-sized 0.005, and the test season went the other way
(2.61 against 2.59). The surprise is in `rate_prior_report.md`: with the prior's strength on the scale
of the weighted evidence, it is a much better *rate* estimate (next season's per-36 error 12% lower
for players with few minutes, 5% for second-year players) and yet a slightly worse projection
(2.340): the gradient-boosted models already learn the shrinkage from minutes, age and draft slot.

**V5: learning-to-rank.** LambdaRank rankers over the greedy recommender's top 24 candidates,
distilled from the rollouts' labels and then fine-tuned on realised outcomes (1,680 queries in
simulated leagues of five seasons, trained leaving one season out). On held-out seasons the simulator's
own ranking stays best: NDCG@5 against realised outcomes 0.615 against 0.598 for the fine-tuned ranker
(paired difference -0.017, interval -0.025 to -0.010), realised regret 5.44 against 5.54 FP/week.
A single realised season per candidate is mostly noise (its rank correlation with the simulator's
expectation within a pick is 0.07 to 0.23), so there was little systematic bias to learn: as the
design expected, **the realised outcomes do not correct the simulator**. In the paired draft backtest,
the design's judge, fine-tuning on them makes the ranker worse than the distilled one (P(top 6) -0.037,
interval -0.056 to -0.020; -1.4 FP/week, -2.4 to -0.5). The by-product is useful: the distilled
ranker (`ltr_sim`) drafts as well as the rollouts it imitates, at milliseconds instead of 1-2 s per pick
(paired over 120 leagues: P(top 6) +0.011, interval -0.025 to +0.049; +1.8 FP/week, +0.1 to +3.6). It
learned from rollouts over all 24 candidates, while the backtest's rollouts consider the greedy
recommender's top 6, which may be why it is no worse; with one realised season per year, a small edge
either way is not established. Full tables: `ltr_report.md` (`draftai ltr --use-backtest-rows` after
`draftai backtest`).

Full tables: `data/artifacts/reports/projection_report.md` and `lockin_experiment.md`
(regenerated by `draftai train` and `draftai lockin-experiment`). See also
[docs/model_card.md](docs/model_card.md).

## Offline mocks that draft themselves

In the app, open **Offline mock**: with "Other teams draft automatically" on, every other team
picks with the opponent model the moment it is on the clock, so you only make your own picks.
Your league's settings (teams, rounds, third-round reversal, lineup) are used when a league is
configured. Draft the recommendation with one click, pick anyone else, undo your last pick (the
room's picks after it are taken back too), or **Auto-draft the rest** with the greedy engine or,
with the rollout toggle on, the Monte Carlo rollouts. The same room seed replays the same room.
When the draft ends you get every team's projected points per week and simulated P(playoffs)
and P(title); these come from the model that drafted your team, so use them to compare your own
mocks, not as a forecast. `draftai mock` does the same in the terminal.

## The React front end (V5)

`make web` builds it once (Node 18+), then `draftai api` serves it with the API at
http://127.0.0.1:8000 (interactive API docs at `/docs`). It does everything the Streamlit app does,
from the same engine: offline mocks that draft themselves (seeded, with undo and auto-draft), live
Sleeper drafts followed by polling (the server pushes each pick over a WebSocket), and manual entry.
The draft room shows three recommendation cards (with the rollouts' confidence and edge when the
toggle is on, and comparables for rookies), a sortable and searchable board with survival odds to
your next pick, a player panel (projection range, Lock-In profile, minutes and role notes, TreeSHAP
drivers, weekly values, the gap to the top pick), your lineup, a watchlist kept in the browser, the
room's likely next picks, the draft log and the minutes-override editor (saving re-projects the
board); at the end, the league projected. Light and dark themes, and it works on a phone. For
front-end work, `cd web && npm run dev` serves it on :5173 against a running `draftai api`. The
Streamlit app (`draftai app`) remains.

## Known limitations

- **ADP.** Sleeper's API has no ADP. The default is Sleeper's `search_rank` (the order its
  draft room lists players), re-ranked among draftable players. Better: aggregate your own
  mock drafts with `draftai adp --username`, or supply a CSV. Historical ADP snapshots do not
  exist, so the backtest's rooms draft around a naive per-game proxy (see the backtest report).
- **Opponents.** One real draft (LeFantasy 2025) is too little to identify the positional-need
  term or per-manager habits; the live room is calibrated to the configured ADP spread, which
  is itself a default until you fit it from your drafts (`draftai adp --username`).
- **VOR in leagues with UTIL slots** is projected value minus a constant (every player's
  cheapest eligible slot is UTIL), so `points` and `vor` draft identically. Positional scarcity
  reaches decisions through marginal roster value (the recommender and the rollouts).
- **Mid-week injuries.** Weekly value treats availability week by week; weeks in which a
  player is hurt or returns mid-week are where Lock-In value is optimistic (see the
  experiment report). Simulated seasons model absences as one block of weeks.
- **Roster at the cut-off (training only).** Historical rows use each player's first team
  of the season as the cut-off team; live projections use Sleeper's current team.
- **Roles.** Retirements and unsigned free agents are not counted as departures in the
  opportunity features: in historical data they cannot be told apart from season-long injuries
  without looking at the season itself. Live, they leave the team minutes budget instead. Live
  newcomers include undrafted and two-way signings at the second-round baseline (training
  counts the draft class only). Positions for the opportunity weighting come from height, so
  that history and live projections use the same buckets.
- **Rookies.** No college or international statistics: nba_api has none, so the rookie model uses
  draft slot, age, size, college-or-club background and the team's vacated minutes. Age and size
  enter only the models conditional on playing (historically they come from bio rows, which exist only
  for players who reached the NBA). In history a rookie's team is his first team of the season, or his
  draft team if he never played, so draft-night trades are known only for players who played.
- **The API and the React front end** are for one user on your own machine: draft sessions live in
  memory (lost on restart), there is no authentication, and `draftai api` binds to 127.0.0.1. Rollouts
  run in the server's thread pool, but they start when asked rather than ahead of your pick.
- **Not yet modelled:** DNP reasons (injury vs rest vs coach's decision), start rate, moving a
  bench player into a passed slot mid-week, in-season trades, and keeper value. Rollouts run in the
  Streamlit app's main thread (1-2 s) rather than in the background.

## Layout

```text
src/draftai/
  ingestion/   nba_api + Sleeper clients, raw cache, rate limits
  normalise/   typed Parquet tables, NBA <-> Sleeper crosswalk
  validation/  Pandera schemas, cross-table checks
  scoring/     settings from Sleeper's scoring_settings; deterministic engine (half-points)
  features/    as-of contract, player-season features; rates (hierarchical gamma-Poisson prior for
               per-minute rates); roles (team moves, net minutes opportunity)
  models/      baselines, component models, team minutes budget and minutes overrides, quantiles +
               conformal, rookies (ridge + comparables), rate-prior experiment, artifacts, TreeSHAP
  value/       game simulator, Lock-In stopping values, schedule weeks, lineup optimiser, VOR, board;
               worlds (simulated seasons), season (head-to-head schedules and the bracket)
  draft/       pick order, state, ADP, survival, recommender, explanations, league context;
               needs (positional need), opponents (conditional logit), rollout (Monte Carlo), mock,
               ranker (learning-to-rank queries, features, LambdaRank)
  api/         FastAPI app: board, draft sessions, recommendations, WebSocket
  live/        Sleeper draft poller
  backtest/    past seasons frozen at the draft date, real-season play, strategies, paired runner,
               real-draft replay; ltr (the learning-to-rank experiment)
app/           Streamlit app
web/           React + TypeScript front end (Vite), served by `draftai api`
tests/         unit, property (Hypothesis), integration (synthetic league, recorded draft, V4 rollouts,
               mocks and backtest, app mock flow, V5 rookies, rate priors, learning-to-rank, API),
               golden Sleeper games
```

Data under `data/` is git-ignored; the repository ships code and download scripts only.
