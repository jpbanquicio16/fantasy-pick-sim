# fantasy-pick-sim

A probabilistic draft assistant for **Sleeper Lock-In** fantasy basketball. It projects
every player as a distribution of per-game box scores and a games-played distribution,
turns each into a Lock-In weekly value (the expected score of the game you would
actually lock, an optimal-stopping problem), and recommends picks by marginal roster
value plus the chance each player survives to your next pick.

This repository implements phases **V0–V4** of the technical design: foundations,
projections, distributions and Lock-In, the live draft assistant, and simulation (an
opponent model fitted on real Sleeper drafts, Monte Carlo draft rollouts with a calibrated
confidence, and a paired draft backtest on five past seasons plus replays of your real
drafts). Offline mocks draft themselves: every other team picks with the opponent model.

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
.venv/bin/draftai backtest    # ~15 min: paired draft backtest + replay of your real drafts
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
| `make test` | Full test suite (offline, ~20 s) |

## Configuration

- `configs/league.yaml`: league id, username or draft slot, playoff weight. Roster
  positions, scoring, team count, rounds, draft type, third-round reversal, traded picks
  and playoff weeks are **read from the Sleeper API** when a league or draft is connected;
  the YAML values are only offline fallbacks.
- `configs/availability_overrides.csv`: manual games-share overrides for news the model
  cannot know (a season-ending injury, a return from one). Copy the `.example.csv`.
  Columns: `sleeper_id`, then `share` (absolute, 0-1) or `multiplier`, and `note`.
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

| | Validation 2021-22 to 2024-25 (mean of 4) | Test 2025-26 (touched once) |
|---|---|---|
| Last season | 2.64 | 2.97 |
| Marcel-style 5/3/2 | 2.78 | 2.90 |
| Ridge baseline | 2.74 | 2.87 |
| **Components (LightGBM + ridge blend)** | **2.36** | **2.64** |

The component model beats Marcel in all four validation seasons, each with a paired
bootstrap 95% interval above zero (+0.36 to +0.55 FP/G), and in the test season
(+0.26, interval +0.05 to +0.47).

**Uncertainty.** Conformalised 80% intervals for season FP/G cover 78-87% in validation
and 80% (top 200) in the test season; games-share intervals cover 78%.

**Lock-In (V2 gate: pass).** On 3,870 held-out 2025-26 player-weeks, the projected Lock-In
value predicts the realised locked score better than mean FP/G (MAE 6.46 vs 7.16, gain
+0.70, 95% CI +0.46 to +0.92). In weeks where the player played every game it is close to
unbiased (+0.36), while mean FP/G understates by 3.5. The lockable premium follows the
theory: 0.40, 0.63 and 0.79 SD for 2, 3 and 4 games.

**Draft assistant (V3).** Recommendations take about 0.05 s per pick (budget: 5 s), follow a
live Sleeper draft by polling, and handle snake, linear, third-round reversal and traded picks.
Monte Carlo rollouts (V4) take 1-2 s per pick with 600 simulated drafts and seasons per candidate.

**Draft backtest (V4).** Five seasons (2021-22 to 2025-26), your league's shape (12 teams, 14 rounds, third-round
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
- **Not yet modelled:** DNP reasons (injury vs rest vs coach's decision), start rate, moving a
  bench player into a passed slot mid-week, trades, and keeper value. Rollouts run in the app's
  main thread (1-2 s) rather than in the background.

## Layout

```text
src/draftai/
  ingestion/   nba_api + Sleeper clients, raw cache, rate limits
  normalise/   typed Parquet tables, NBA <-> Sleeper crosswalk
  validation/  Pandera schemas, cross-table checks
  scoring/     settings from Sleeper's scoring_settings; deterministic engine (half-points)
  features/    as-of contract, player-season features, empirical-Bayes shrinkage
  models/      baselines, component models, quantiles + conformal, rookies, artifacts, TreeSHAP
  value/       game simulator, Lock-In stopping values, schedule weeks, lineup optimiser, VOR, board;
               worlds (simulated seasons), season (head-to-head schedules and the bracket)
  draft/       pick order, state, ADP, survival, recommender, explanations, league context;
               needs (positional need), opponents (conditional logit), rollout (Monte Carlo), mock
  live/        Sleeper draft poller
  backtest/    past seasons frozen at the draft date, real-season play, strategies, paired runner,
               real-draft replay
app/           Streamlit app
tests/         unit, property (Hypothesis), integration (synthetic league, recorded draft, V4 rollouts,
               mocks and backtest, app mock flow), golden Sleeper games
```

Data under `data/` is git-ignored; the repository ships code and download scripts only.
