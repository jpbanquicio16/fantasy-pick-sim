"""Live draft assistant (Streamlit).

    draftai app        or        streamlit run app/streamlit_app.py

The page holds no decision logic: it builds a DraftState (from Sleeper polling or manual
input), asks the recommender, and displays the result.
"""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from draftai import pipeline
from draftai.config import Paths, load_league_config, load_sim_config
from draftai.draft.explain import card
from draftai.draft.league import LeagueContext
from draftai.draft.recommender import RosterEvaluator, recommend
from draftai.draft.survival import survival
from draftai.ingestion.sleeper import SleeperClient
from draftai.live.poller import state_from_picks
from draftai.models.explain import top_drivers

st.set_page_config(page_title="Lock-In Draft Assistant", layout="wide")
SIM = load_sim_config()


@st.cache_resource(show_spinner="Loading the draft board...")
def load_board(season: int):
    board = pipeline.load_board(Paths(), season)
    return board, RosterEvaluator(board)


@st.cache_resource
def client() -> SleeperClient:
    return SleeperClient()


ss = st.session_state
ss.setdefault("ctx", None)
ss.setdefault("state", None)
ss.setdefault("mode", None)
ss.setdefault("watch", [])
ss.setdefault("last_poll", 0.0)


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
        c1, c2, c3 = st.columns(3)
        cfg.teams = c1.number_input("Teams", 4, 32, cfg.teams)
        cfg.draft_slot = c2.number_input("Your slot", 1, 32, cfg.draft_slot or 1)
        rr = c3.selectbox("Draft type", ["snake", "snake + third-round reversal", "linear"])
        cfg.draft_type = "linear" if rr == "linear" else "snake"
        cfg.reversal_round = 3 if "reversal" in rr else 0
        if st.button("Start offline mock", type="primary"):
            start(LeagueContext.offline(cfg), "manual")


def start(ctx: LeagueContext, mode: str) -> None:
    ss.ctx, ss.mode = ctx, mode
    if ctx.scoring.unknown_keys:
        st.warning(f"Unrecognised scoring keys ignored: {ctx.scoring.unknown_keys}")
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

    with st.sidebar:
        st.subheader(ctx.league_name)
        st.caption(f"{ctx.teams} teams, {ctx.rounds} rounds, {ctx.draft_type}"
                   + (f" (reversal from round {ctx.reversal_round})" if ctx.reversal_round else "")
                   + f" | your slot {ctx.my_slot}")
        st.caption(f"Bonus stacking 40+/50+: {ctx.scoring.bonus_stacking}")
        if st.button("Disconnect"):
            ss.ctx = ss.state = ss.mode = None
            st.rerun()
        st.divider()
        st.markdown("**Your picks**: " + ", ".join(str(x) for x in state.my_picks()[:8]) + " ...")
        st.markdown("**Your roster**")
        for pid in state.roster():
            st.write("- " + names.get(pid, pid))

    if state.complete:
        st.success("Draft complete.")
        return

    until = state.picks_until_mine()
    banner = (f"Pick **{state.current_pick}** of {state.total_picks}. "
              + ("**You are on the clock.**" if state.my_turn else
                 f"Your next pick is {state.my_upcoming()[0]} ({until} picks away)." if until is not None else "No picks left."))
    (st.success if state.my_turn else st.info)(banner)

    rec = recommend(board, state, SIM, ev)

    # Recommendation cards.
    st.subheader(f"Recommendation for pick {rec.pick}" + ("" if rec.pick == state.current_pick else " (planning ahead)"))
    cols = st.columns(3)
    for i, (col, c) in enumerate(zip(cols, rec.candidates[:3])):
        r = p.row(c.row, named=True)
        with col.container(border=True):
            st.markdown(f"**{i + 1}. {player_label(r)}**")
            gap = rec.per_week(rec.candidates[0].score - c.score)
            st.caption("top pick" if i == 0 else f"{gap:.1f} FP/week behind #1")
            m1, m2, m3 = st.columns(3)
            m1.metric("FP/G", f"{r['fp_pg']:.1f}", help=f"season P10-P90: {r['fp_p10']:.0f}-{r['fp_p90']:.0f}")
            m2.metric("Lock-In (3 g)", f"{r['lockin_3']:.1f}", f"+{r['lockin_premium_3']:.1f} spread")
            m3.metric("VOR/week", f"{r['vor_per_week']:+.1f}")
            if rec.pick > state.current_pick:
                st.write(f"Still there at your pick {rec.pick}: **{c.reach:.0%}**")
            if rec.next_pick:
                st.write(f"Survives to pick {rec.next_pick}: **{c.survival:.0%}**  |  ADP {c.adp:.0f}")
            st.write(f"Games share {r['share_hat']:.0%}" + (f"  |  {r['injury_status']}" if r.get("injury_status") else ""))
            drv = top_drivers(r, "fp")
            if drv:
                st.caption("Projection drivers (TreeSHAP, FP/G): " + ", ".join(f"{g} {v:+.1f}" for g, v in drv))
    info = card(rec, p, ctx.scoring)
    if info.get("why"):
        st.markdown("**Why:** " + " ".join(info["why"]))
    if info.get("risk"):
        st.markdown("**Risk:** " + "; ".join(info["risk"]) + ".")

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
            pl.col("fp_pg").round(1), pl.col("lockin_3").round(1), pl.col("share_hat").round(2),
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

    with right:
        st.subheader("What if")
        avail_rows = table.select("sleeper_id", "name").rows()
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
