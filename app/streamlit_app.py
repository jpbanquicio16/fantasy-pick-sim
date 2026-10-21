"""Live draft assistant (Streamlit).

    draftai app        or        streamlit run app/streamlit_app.py

The page holds no decision logic: it builds a DraftState (from Sleeper polling, manual
input, or an offline mock whose other teams draft themselves), asks the recommender (and,
when switched on, the Monte Carlo rollouts), and displays the result.
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from draftai import pipeline
from draftai.config import Paths, load_league_config, load_sim_config
from draftai.draft.explain import card, minutes_text, role_text
from draftai.draft.league import LeagueContext
from draftai.draft.mock import MockDraft
from draftai.draft.recommender import RosterEvaluator, recommend
from draftai.draft.rollout import RolloutConfig, RolloutEngine
from draftai.draft.survival import survival
from draftai.ingestion.sleeper import SleeperClient
from draftai.live.poller import state_from_picks
from draftai.models.explain import top_drivers

st.set_page_config(page_title="Lock-In Draft Assistant", layout="wide")
SIM = load_sim_config()
ROLLOUT_LOOKAHEAD = 3  # start rollouts when you are this many picks away


@st.cache_resource(show_spinner="Loading the draft board...")
def load_board(season: int):
    board = pipeline.load_board(Paths(), season)
    return board, RosterEvaluator(board)


@st.cache_resource
def client() -> SleeperClient:
    return SleeperClient()


@st.cache_resource(show_spinner="Calibrating the opponent model to this board (once)...")
def opponents(season: int, teams: int, rounds: int, draft_type: str, reversal_round: int, traded: tuple, _ctx):
    board, _ = load_board(season)
    return pipeline.live_opponents(_ctx, board, SIM, Paths(), log=lambda *_: None)


@st.cache_resource(show_spinner="Drawing simulated seasons for the rollouts...")
def rollout_engine(season: int, teams: int, rounds: int, draft_type: str, reversal_round: int, traded: tuple, _ctx):
    board, ev = load_board(season)
    model = opponents(season, teams, rounds, draft_type, reversal_round, traded, _ctx)
    return RolloutEngine(board, model, RolloutConfig.from_sim(SIM), SIM, evaluator=ev)


def ctx_key(ctx: LeagueContext) -> tuple:
    return ctx.season, ctx.teams, ctx.rounds, ctx.draft_type, ctx.reversal_round, tuple(ctx.traded)


ss = st.session_state
ss.setdefault("ctx", None)
ss.setdefault("state", None)
ss.setdefault("mode", None)
ss.setdefault("mock", None)
ss.setdefault("watch", [])
ss.setdefault("last_poll", 0.0)
ss.setdefault("rollout", None)  # (state version, result)
ss.setdefault("projection", None)  # (state version, table)


# --- connect ----------------------------------------------------------------------
def connect_panel() -> None:
    cfg = load_league_config()
    st.title("Lock-In Draft Assistant")
    st.caption("Sleeper Lock-In fantasy basketball. Settings are read from your league; nothing is hard-coded.")
    tab_league, tab_draft, tab_mock = st.tabs(["My Sleeper league", "Draft or mock-draft id", "Offline mock"])
    with tab_league:
        username = st.text_input("Sleeper username", value=cfg.username or "")
        if username:
            user = client().user(username)
            if not user:
                st.error("No Sleeper user with that name.")
            else:
                leagues = client().user_leagues(user["user_id"], cfg.season)
                if not leagues:
                    st.warning(f"No {cfg.season} NBA leagues for {username}.")
                else:
                    names = {f"{lg['name']} ({lg['league_id']})": lg["league_id"] for lg in leagues}
                    choice = st.selectbox("League", list(names))
                    if st.button("Connect to league", type="primary"):
                        ctx = LeagueContext.from_sleeper(cfg, client(), league_id=names[choice], username=username)
                        start(ctx, "live")
    with tab_draft:
        draft_id = st.text_input("Draft id (from the draft room URL)")
        slot = st.number_input("Your slot (if not detected from username)", 1, 32, cfg.draft_slot or 1)
        username2 = st.text_input("Sleeper username (optional, finds your slot)", key="u2")
        if draft_id and st.button("Follow draft", type="primary"):
            cfg.draft_slot = int(slot)
            ctx = LeagueContext.from_sleeper(cfg, client(), draft_id=draft_id, username=username2 or None)
            start(ctx, "live")
    with tab_mock:
        mock_panel(cfg)


def mock_panel(cfg) -> None:
    st.write("Practise against a simulated room. With auto-draft on, every other team picks with the "
             "opponent model (tight to ADP early, looser late), so you only make your own picks.")
    use_league = False
    if cfg.league_id:
        use_league = st.checkbox("Use my Sleeper league's settings (teams, rounds, draft order, lineup)", value=True)
    c1, c2, c3 = st.columns(3)
    if not use_league:
        cfg.teams = c1.number_input("Teams", 4, 32, cfg.teams)
        rr = c3.selectbox("Draft type", ["snake", "snake + third-round reversal", "linear"])
        cfg.draft_type = "linear" if rr == "linear" else "snake"
        cfg.reversal_round = 3 if "reversal" in rr else 0
    slot = c2.number_input("Your slot", 1, 32, cfg.draft_slot or 1)
    a1, a2 = st.columns([2, 1])
    auto = a1.toggle("Other teams draft automatically", value=True)
    seed_text = a2.text_input("Room seed (optional)", help="The same seed replays the same room.")
    if st.button("Start offline mock", type="primary"):
        if use_league:
            ctx = LeagueContext.from_sleeper(cfg, client())
            ctx.my_slot = int(slot)
            ctx.slot_names = {}
        else:
            cfg.draft_slot = int(slot)
            ctx = LeagueContext.offline(cfg)
        seed = int(seed_text) if seed_text.strip().isdigit() else None
        start(ctx, "manual", auto=auto, seed=seed)


def start(ctx: LeagueContext, mode: str, auto: bool = False, seed: int | None = None) -> None:
    ss.ctx, ss.mode = ctx, mode
    ss.rollout = ss.projection = None
    if ctx.scoring.unknown_keys:
        st.warning(f"Unrecognised scoring keys ignored: {ctx.scoring.unknown_keys}")
    ss.mock = None
    if mode == "manual" and auto:
        board, _ = load_board(ctx.season)
        model = opponents(*ctx_key(ctx), _ctx=ctx)
        ss.mock = MockDraft(ctx, board, model, **({"seed": seed} if seed is not None else {}))
        ss.state = ss.mock.state
    else:
        ss.state = ctx.new_state()
    if mode == "live":
        poll(force=True)
    st.rerun()


def poll(force: bool = False) -> bool:
    if ss.mode != "live" or (not force and time.time() - ss.last_poll < SIM.poll_seconds):
        return False
    ss.last_poll = time.time()
    picks = client().draft_picks(ss.ctx.draft_id)
    new = state_from_picks(ss.ctx, picks)
    changed = ss.state is None or new.version != ss.state.version
    ss.state = new
    return changed


# --- main screen ------------------------------------------------------------------
def player_label(row: dict) -> str:
    return f"{row['name']} ({'/'.join(row['positions'])}, {row['team_now']})"


def rollout_for(state, rec, board, ev):
    """Cached per draft state; computed when you are on the clock or nearly so."""
    if ss.rollout is not None and ss.rollout[0] == state.version:
        return ss.rollout[1]
    until = state.picks_until_mine()
    if until is None or until > ROLLOUT_LOOKAHEAD:
        return None
    engine = rollout_engine(*ctx_key(ss.ctx), _ctx=ss.ctx)
    with st.spinner(f"Simulating {engine.cfg.worlds} drafts and seasons per candidate..."):
        res = engine.recommend(state, rec)
    ss.rollout = (state.version, res)
    return res


def mock_controls(state, rec, ro, board, ev, avail_rows) -> None:
    """Your pick in a mock: the recommendation, any player, undo, or auto-draft the rest."""
    mock: MockDraft = ss.mock
    st.subheader("Your pick")
    best = (ro.best.sleeper_id, ro.best.name) if ro is not None and ro.best else (
        (rec.best.sleeper_id, rec.best.name) if rec.best else (None, None))
    c1, c2, c3 = st.columns(3)
    if c1.button(f"Draft {best[1]}" if best[1] else "Draft the recommendation", type="primary",
                 disabled=not (state.my_turn and best[0])):
        mock.pick(best[0], by="you")
        st.rerun()
    if c2.button("Undo my last pick", disabled=not any(m.slot == state.my_slot for m in mock.log)):
        mock.undo_mine()
        ss.rollout = ss.projection = None
        st.rerun()
    use_ro = st.session_state.get("use_rollout", False)
    if c3.button("Auto-draft the rest", help="The engine picks for you" + (" (rollouts)" if use_ro else "") + "."):
        engine = rollout_engine(*ctx_key(ss.ctx), _ctx=ss.ctx) if use_ro else None

        def chooser(st_):
            r = recommend(board, st_, SIM, ev)
            return engine.recommend(st_, r).best.sleeper_id if engine is not None else r.best.sleeper_id

        with st.spinner("Drafting the rest of the mock..."):
            mock.run_to_end(chooser)
        st.rerun()
    choice = st.selectbox("Or draft any available player", [""] + [n for _, n in avail_rows], key="mock_choice")
    if st.button("Draft this player", disabled=not (choice and state.my_turn)):
        mock.pick(next(s for s, n in avail_rows if n == choice), by="you")
        st.rerun()
    if not state.complete and not state.my_turn:
        st.caption("The room is picking...")


def draft_log(limit: int | None = 24) -> None:
    mock: MockDraft = ss.mock
    log = mock.log_frame()
    if log.height == 0:
        return
    st.subheader("Draft log")
    show = log.reverse() if limit is None else log.tail(limit).reverse()
    st.dataframe(show.to_pandas(), hide_index=True, height=320)


def mock_results(board, ev) -> None:
    mock: MockDraft = ss.mock
    st.success("Mock complete.")
    if ss.projection is None or ss.projection[0] != mock.state.version:
        with st.spinner("Simulating the season for every team..."):
            ss.projection = (mock.state.version, mock.projection(ev, n_worlds=300, waiver_slots=SIM.waiver_slots))
    proj = ss.projection[1]
    me = proj.filter(pl.col("team") == "You").row(0, named=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Projected rank", f"{me['proj_rank']} of {proj.height}")
    m2.metric("P(playoffs)", f"{me['p_playoffs']:.0%}")
    m3.metric("P(title)", f"{me['p_title']:.0%}")
    st.caption("Projected by this model against opponents who draft around ADP, so it flatters teams the "
               "engine drafted. Use it to compare your own mocks, not as a forecast.")
    st.dataframe(proj.to_pandas(), hide_index=True)
    names = dict(zip(board.players["sleeper_id"].to_list(), board.players["name"].to_list()))
    st.markdown("**Your roster**: " + ", ".join(names.get(p, p) for p in mock.state.roster()))
    draft_log(limit=None)
    if st.button("New mock with the same settings", type="primary"):
        start(ss.ctx, "manual", auto=True)


OVERRIDES = Paths().configs / "minutes_overrides.csv"


def minutes_overrides_panel(board, ctx: LeagueContext) -> None:
    """Expected minutes for preseason news the model cannot know. Saved to
    configs/minutes_overrides.csv; the board is rebuilt (a few seconds) with the change
    taken from or given to the player's teammates. The model is untouched."""
    p = board.players
    by_name = dict(zip(p["name"].to_list(), p["sleeper_id"].to_list()))
    by_id = {v: k for k, v in by_name.items()}
    with st.expander("Minutes overrides (preseason news)"):
        st.caption("A starting job announced, a move to the bench: enter his expected minutes per game. "
                   "Teammates absorb the change, mostly at his positions.")
        cur = pl.read_csv(OVERRIDES, schema_overrides={"sleeper_id": pl.Utf8, "mpg": pl.Float64, "note": pl.Utf8}) \
            if OVERRIDES.exists() else pl.DataFrame(schema={"sleeper_id": pl.Utf8, "mpg": pl.Float64, "note": pl.Utf8})
        rows = [{"player": by_id.get(sid, sid), "mpg": m, "note": n}
                for sid, m, n in cur.select("sleeper_id", "mpg", "note").iter_rows()]
        edited = st.data_editor(
            rows or [{"player": None, "mpg": None, "note": None}], num_rows="dynamic", hide_index=True, key="mo_edit",
            column_config={
                "player": st.column_config.SelectboxColumn("Player", options=sorted(by_name)),
                "mpg": st.column_config.NumberColumn("Min/game", min_value=0.0, max_value=44.0, step=0.5),
                "note": st.column_config.TextColumn("Note"),
            })
        if st.button("Save and rebuild the board"):
            keep = [r for r in edited if r.get("player") and r.get("mpg") is not None]
            out = pl.DataFrame({"sleeper_id": [by_name.get(r["player"], r["player"]) for r in keep],
                                "name": [r["player"] for r in keep], "mpg": [float(r["mpg"]) for r in keep],
                                "note": [r.get("note") for r in keep]},
                               schema={"sleeper_id": pl.Utf8, "name": pl.Utf8, "mpg": pl.Float64, "note": pl.Utf8})
            out.write_csv(OVERRIDES)
            with st.spinner("Rebuilding the board..."):
                adp = p.select([c for c in ("sleeper_id", "adp", "adp_source", "adp_sd") if c in p.columns])
                pipeline.project(ctx, SIM, Paths(), adp_from=adp, log=lambda *_: None)
            load_board.clear()
            rollout_engine.clear()
            ss.rollout = ss.projection = None
            st.rerun()
        changed = p.filter(pl.col("minutes_override")) if "minutes_override" in p.columns else p.head(0)
        for r in changed.iter_rows(named=True):
            st.write(f"{r['name']}: {minutes_text(r)}")


def main_screen() -> None:
    ctx, state = ss.ctx, ss.state
    try:
        board, ev = load_board(ctx.season)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    p = board.players
    ids = p["sleeper_id"].to_list()
    names = dict(zip(ids, p["name"].to_list()))
    mock: MockDraft | None = ss.mock
    recent = mock.advance() if mock is not None and not state.complete and not state.my_turn else []
    if recent:
        ss.rollout = None

    with st.sidebar:
        st.subheader(ctx.league_name if not mock else f"Mock: {ctx.league_name}")
        st.caption(f"{ctx.teams} teams, {ctx.rounds} rounds, {ctx.draft_type}"
                   + (f" (reversal from round {ctx.reversal_round})" if ctx.reversal_round else "")
                   + f" | your slot {ctx.my_slot}")
        st.caption(f"Bonus stacking 40+/50+: {ctx.scoring.bonus_stacking}")
        if mock is not None:
            st.caption(f"Room seed {mock.seed}. {mock.model.describe()}")
        st.toggle("Monte Carlo rollouts (V4)", key="use_rollout",
                  help=f"Simulate {SIM.rollout_worlds} drafts and seasons per candidate when you are within "
                       f"{ROLLOUT_LOOKAHEAD} picks; adds confidence (P(A's roster beats B's)). About 1-2 s.")
        minutes_overrides_panel(board, ctx)
        if st.button("Disconnect"):
            ss.ctx = ss.state = ss.mode = ss.mock = ss.rollout = ss.projection = None
            st.rerun()
        st.divider()
        st.markdown("**Your picks**: " + ", ".join(str(x) for x in state.my_picks()[:8]) + " ...")
        st.markdown("**Your roster**")
        for pid in state.roster():
            st.write("- " + names.get(pid, pid))

    if state.complete:
        if mock is not None:
            mock_results(board, ev)
        else:
            st.success("Draft complete.")
        return

    until = state.picks_until_mine()
    banner = (f"Pick **{state.current_pick}** of {state.total_picks}. "
              + ("**You are on the clock.**" if state.my_turn else
                 f"Your next pick is {state.my_upcoming()[0]} ({until} picks away)." if until is not None else "No picks left."))
    (st.success if state.my_turn else st.info)(banner)
    if recent:
        st.caption("Since your last pick: " + "; ".join(f"{m.pick}. {m.name}" for m in recent[-12:]))

    rec = recommend(board, state, SIM, ev)
    ro = rollout_for(state, rec, board, ev) if st.session_state.get("use_rollout") else None
    order = [c.row for c in ro.candidates] if ro is not None else [c.row for c in rec.candidates]
    by_row = {c.row: c for c in rec.candidates}
    ro_by_row = {c.row: c for c in ro.candidates} if ro is not None else {}

    # Recommendation cards.
    title = f"Recommendation for pick {rec.pick}" + ("" if rec.pick == state.current_pick else " (planning ahead)")
    if ro is not None:
        title += f" | {ro.worlds} simulated drafts and seasons"
    st.subheader(title)
    cols = st.columns(3)
    for i, (col, row) in enumerate(zip(cols, order[:3])):
        c = by_row[row]
        r = p.row(row, named=True)
        with col.container(border=True):
            st.markdown(f"**{i + 1}. {player_label(r)}**")
            if ro is not None:
                rc = ro_by_row[row]
                gap = ro.per_week(ro.candidates[0].mean - rc.mean)
                if i == 0 and ro.confidence is not None:
                    st.caption(f"top pick: beats #2's roster in {ro.confidence:.0%} of simulations, "
                               f"+{ro.per_week(ro.edge):.1f} FP/week")
                else:
                    st.caption(("dropped early: clearly behind | " if rc.dropped else "")
                               + f"{gap:.1f} FP/week behind #1; best in {rc.p_best:.0%} of simulations")
            else:
                gap = rec.per_week(rec.candidates[0].score - c.score)
                st.caption("top pick" if i == 0 else f"{gap:.1f} FP/week behind #1")
            m1, m2, m3 = st.columns(3)
            m1.metric("FP/G", f"{r['fp_pg']:.1f}", help=f"season P10-P90: {r['fp_p10']:.0f}-{r['fp_p90']:.0f}")
            m2.metric("Lock-In (3 g)", f"{r['lockin_3']:.1f}", f"+{r['lockin_premium_3']:.1f} spread")
            m3.metric("VOR/week", f"{r['vor_per_week']:+.1f}")
            if rec.pick > state.current_pick:
                reach = ro_by_row[row].reach if ro is not None else c.reach
                st.write(f"Still there at your pick {rec.pick}: **{reach:.0%}**")
            if rec.next_pick:
                s_ = ro.survival.get(c.sleeper_id, c.survival) if ro is not None else c.survival
                st.write(f"Survives to pick {rec.next_pick}: **{s_:.0%}**  |  ADP {c.adp:.0f}")
            st.write(f"Games share {r['share_hat']:.0%}" + (f"  |  {r['injury_status']}" if r.get("injury_status") else ""))
            if minutes_text(r):
                st.write("Minutes " + minutes_text(r))
            if role_text(r):
                st.caption("Role: " + role_text(r))
            drv = top_drivers(r, "fp")
            if drv:
                st.caption("Projection drivers (TreeSHAP, FP/G): " + ", ".join(f"{g} {v:+.1f}" for g, v in drv))
    info = card(rec, p, ctx.scoring, rollout=ro)
    if info.get("why"):
        st.markdown("**Why:** " + " ".join(info["why"]))
    if info.get("risk"):
        st.markdown("**Risk:** " + "; ".join(info["risk"]) + ".")
    if info.get("comps"):
        st.caption("Rookies like him (year-one FP/G): " + "; ".join(info["comps"]))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Available players")
        avail = np.array([s not in state.drafted for s in ids])
        nxt = rec.next_pick
        sv = survival(p["adp"].to_numpy(), state.current_pick, nxt, SIM.adp_sigma_a, SIM.adp_sigma_b)
        score = {c.sleeper_id: c for c in rec.candidates}
        table = p.with_columns(
            pl.Series("survival", sv), pl.Series("available", avail),
            pl.Series("score_gap", [rec.per_week(rec.candidates[0].score - score[s].score) if s in score else None for s in ids]),
        ).filter(pl.col("available")).sort("vor", descending=True)
        q = st.text_input("Search", "")
        if q:
            table = table.filter(pl.col("name").str.to_lowercase().str.contains(q.lower(), literal=True))
        show = table.select(
            "name", pl.col("positions").list.join("/").alias("pos"), "team_now", pl.col("adp").round(0),
            pl.col("fp_pg").round(1), pl.col("lockin_3").round(1), pl.col("mpg_hat").round(1).alias("min"),
            pl.col("share_hat").round(2),
            pl.col("vor_per_week").round(1), pl.col("survival").round(2), pl.col("score_gap").round(1), "injury_status",
        ).head(150)
        st.dataframe(show.to_pandas(), hide_index=True, height=420)

        fig = go.Figure(go.Scatter(
            x=table["adp"].head(120).to_list(), y=table["vor_per_week"].head(120).to_list(), mode="markers",
            text=table["name"].head(120).to_list(), hovertemplate="%{text}<br>ADP %{x:.0f}<br>VOR/week %{y:.1f}",
        ))
        fig.add_vline(x=state.current_pick, line_dash="dash")
        if nxt:
            fig.add_vline(x=nxt, line_dash="dot")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title="Value vs market (lines: now, your next pick)",
                          xaxis_title="ADP", yaxis_title="VOR per week")
        st.plotly_chart(fig, width="stretch")
        if mock is not None:
            draft_log()

    with right:
        avail_rows = table.select("sleeper_id", "name").rows()
        if mock is not None:
            mock_controls(state, rec, ro, board, ev, avail_rows)
            st.divider()
        st.subheader("What if")
        pick = st.selectbox("Any available player", [""] + [n for _, n in avail_rows])
        if pick:
            sid = next(s for s, n in avail_rows if n == pick)
            c = score.get(sid)
            if c:
                st.write(f"Score gap to the top pick: **{rec.per_week(rec.candidates[0].score - c.score):.1f} FP/week**; "
                         f"marginal value {rec.per_week(c.delta):.1f} FP/week; survival {c.survival:.0%}.")
            else:
                st.write("Outside the candidate pool: well below the top pick.")
            if st.button("Add to watchlist") and sid not in ss.watch:
                ss.watch.append(sid)
        st.subheader("Watchlist")
        for sid in list(ss.watch):
            if sid in state.drafted:
                st.write(f"~~{names.get(sid, sid)}~~ drafted")
                continue
            i = ids.index(sid)
            s_ = survival(np.array([p["adp"][i]]), state.current_pick, nxt, SIM.adp_sigma_a, SIM.adp_sigma_b)[0]
            st.write(f"{names[sid]}: {s_:.0%} to survive to {nxt}")

        if mock is None:
            st.subheader("Manual pick entry")
            st.caption("For offline mocks, or if polling fails.")
            choice = st.selectbox("Player taken at this pick", [""] + [n for _, n in avail_rows], key="manual")
            c1, c2 = st.columns(2)
            if c1.button("Record pick", disabled=not choice):
                state.add_pick(next(s for s, n in avail_rows if n == choice))
                st.rerun()
            if c2.button("Undo last pick"):
                state.undo()
                st.rerun()


if ss.ctx is None:
    connect_panel()
else:
    if ss.mode == "live":
        @st.fragment(run_every=SIM.poll_seconds)
        def live_poller() -> None:
            if poll():
                st.rerun(scope="app")
            st.caption(f"Following draft {ss.ctx.draft_id}; last update {time.strftime('%H:%M:%S')}")

        live_poller()
    main_screen()
