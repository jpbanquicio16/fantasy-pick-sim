"""FastAPI app: the draft assistant as an API, plus the React front end (``web/``).

    draftai api                     http://127.0.0.1:8000 (the built front end) and /docs
    cd web && npm run dev           front-end development server, proxied to the API

Routes (all JSON, under /api):

    GET  /league                             league settings, slots, board size
    GET  /board                              every board player's projection
    GET  /players/{sleeper_id}               one player: ranges, minutes, role, drivers, weekly values
    POST /drafts                             start a mock (other teams draft themselves), manual or live draft
    GET  /drafts/{id}                        state: picks, rosters, your next pick, survival odds
    POST /drafts/{id}/picks                  make a pick (mocks: yours; manual: whoever is on the clock)
    POST /drafts/{id}/undo | advance | autodraft | sync
    GET  /drafts/{id}/recommendation?rollout=true   greedy (and Monte Carlo) recommendation with its card
    GET  /drafts/{id}/what-if/{sleeper_id}   any available player against the top pick
    GET  /drafts/{id}/odds                   what the room will take next (mocks)
    GET  /drafts/{id}/projection             every team's projected points and playoff odds
    WS   /drafts/{id}/ws                     state pushed on every change; live drafts are polled
    GET/PUT /overrides/minutes               minutes overrides; saving rebuilds the board

Sync handlers run in FastAPI's thread pool, so a 1-2 s rollout never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from draftai.api.schemas import AutoDraft, MinutesOverride, NewDraft, Pick
from draftai.api.service import DraftService, Engine, Session, minutes_overrides

WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
log = logging.getLogger(__name__)


def create_app(service: DraftService, rebuild=None, overrides_path: Path | None = None,
               poll_seconds: float = 3.0, web_dist: Path | None = WEB_DIST) -> FastAPI:
    """``rebuild(overrides_path) -> Board`` re-projects after a minutes-override change."""
    app = FastAPI(title="Lock-In Draft Assistant", version="0.5.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.service = service

    def session(draft_id: str) -> Session:
        try:
            return service.get(draft_id)
        except KeyError:
            raise HTTPException(404, f"no draft {draft_id}") from None

    def attempt(fn, *args):
        try:
            return fn(*args)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "players": service.engine.board.players.height}

    @app.get("/api/league")
    def league() -> dict:
        return service.engine.league()

    @app.get("/api/board")
    def board() -> list[dict]:
        return service.engine.board_table()

    @app.get("/api/players/{sleeper_id}")
    def player(sleeper_id: str) -> dict:
        p = service.engine.player(sleeper_id)
        if p is None:
            raise HTTPException(404, f"no player {sleeper_id}")
        return p

    @app.post("/api/drafts")
    def new_draft(body: NewDraft) -> dict:
        if body.mode == "mock":
            s = service.create_mock(body.slot, body.seed, body.auto)
        elif body.mode == "manual":
            s = service.create_manual(body.slot)
        else:
            if not body.draft_id:
                raise HTTPException(422, "a live draft needs draft_id")
            try:
                s = service.create_live(body.draft_id)
            except Exception as e:  # noqa: BLE001 - network, unknown draft or slot: all a 502 to the client
                raise HTTPException(502, f"could not follow draft {body.draft_id}: {e}") from None
        return service.state_view(s)

    @app.get("/api/drafts/{draft_id}")
    def get_draft(draft_id: str) -> dict:
        return service.state_view(session(draft_id))

    @app.post("/api/drafts/{draft_id}/picks")
    def make_pick(draft_id: str, body: Pick) -> dict:
        s = session(draft_id)
        attempt(service.pick, s, body.sleeper_id)
        return service.state_view(s)

    @app.post("/api/drafts/{draft_id}/undo")
    def undo(draft_id: str) -> dict:
        s = session(draft_id)
        service.undo(s)
        return service.state_view(s)

    @app.post("/api/drafts/{draft_id}/advance")
    def advance(draft_id: str) -> dict:
        s = session(draft_id)
        service.advance(s)
        return service.state_view(s)

    @app.post("/api/drafts/{draft_id}/autodraft")
    def autodraft(draft_id: str, body: AutoDraft) -> dict:
        s = session(draft_id)
        attempt(service.autodraft, s, body.rollout)
        return service.state_view(s)

    @app.post("/api/drafts/{draft_id}/sync")
    def sync(draft_id: str) -> dict:
        s = session(draft_id)
        service.sync(s)
        return service.state_view(s)

    @app.get("/api/drafts/{draft_id}/recommendation")
    def recommendation(draft_id: str, rollout: bool = False, top: int = 12) -> dict:
        return service.recommendation(session(draft_id), rollout, top)

    @app.get("/api/drafts/{draft_id}/what-if/{sleeper_id}")
    def what_if(draft_id: str, sleeper_id: str) -> dict:
        return attempt(service.what_if, session(draft_id), sleeper_id)

    @app.get("/api/drafts/{draft_id}/odds")
    def odds(draft_id: str) -> list[dict]:
        return service.odds(session(draft_id))

    @app.get("/api/drafts/{draft_id}/projection")
    def projection(draft_id: str) -> list[dict]:
        return service.projection(session(draft_id))

    @app.get("/api/overrides/minutes")
    def get_overrides() -> list[dict]:
        return minutes_overrides(overrides_path) if overrides_path else []

    @app.put("/api/overrides/minutes")
    def put_overrides(rows: list[MinutesOverride]) -> dict:
        if rebuild is None or overrides_path is None:
            raise HTTPException(501, "this server cannot rebuild the board")
        import polars as pl

        pl.DataFrame([r.model_dump() for r in rows], schema={"sleeper_id": pl.Utf8, "name": pl.Utf8,
                                                              "mpg": pl.Float64, "note": pl.Utf8}).write_csv(overrides_path)
        service.engine.set_board(rebuild(overrides_path))
        for s in service.sessions.values():
            s._rec = s._ro = None
            s.notify()
        return {"saved": len(rows), "players": service.engine.board.players.height}

    @app.websocket("/api/drafts/{draft_id}/ws")
    async def stream(ws: WebSocket, draft_id: str) -> None:
        try:
            s = service.get(draft_id)
        except KeyError:
            await ws.close(code=4404)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()
        listener = lambda: loop.call_soon_threadsafe(changed.set)
        s.listeners.append(listener)

        async def poll() -> None:
            while s.kind == "live" and not s.state.complete:
                await asyncio.sleep(poll_seconds)
                try:
                    await run_in_threadpool(service.sync, s)
                except Exception as e:  # noqa: BLE001 - a failed poll is retried at the next interval
                    log.warning("polling draft %s failed: %s", draft_id, e)

        poller = asyncio.create_task(poll()) if s.kind == "live" else None
        try:
            await ws.send_json({"type": "state", "state": await run_in_threadpool(service.state_view, s)})
            while True:
                await changed.wait()
                changed.clear()
                await ws.send_json({"type": "state", "state": await run_in_threadpool(service.state_view, s)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            s.listeners.remove(listener)
            if poller is not None:
                poller.cancel()

    if web_dist is not None and (web_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    else:
        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return ("<h1>Lock-In Draft Assistant API</h1><p>The front end is not built: run "
                    "<code>make web</code> (or <code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>). "
                    "The API is live: see <a href='/docs'>/docs</a>.</p>")
    return app


def app_from_config() -> FastAPI:
    """The app for ``draftai api``: this season's board, your league, the calibrated room."""
    from draftai import pipeline
    from draftai.config import Paths, load_league_config, load_sim_config
    from draftai.draft.league import LeagueContext
    from draftai.ingestion.sleeper import SleeperClient
    from draftai.live.poller import LiveDraft

    paths, sim, cfg = Paths(), load_sim_config(), load_league_config()
    try:
        ctx = LeagueContext.from_sleeper(cfg) if (cfg.league_id or cfg.draft_id) else LeagueContext.offline(cfg)
    except Exception:  # noqa: BLE001 - offline: fall back to the YAML settings
        ctx = LeagueContext.offline(cfg)
    if ctx.my_slot is None:
        ctx.my_slot = cfg.draft_slot or 1
    board = pipeline.load_board(paths, ctx.season)
    if board.teams != ctx.teams:
        board = board.with_teams(ctx.teams)
    opponents = pipeline.live_opponents(ctx, board, sim, paths, log=lambda *_: None)

    def live_factory(draft_id: str):
        live_ctx = LeagueContext.from_sleeper(cfg, draft_id=draft_id)
        if live_ctx.my_slot is None:
            live_ctx.my_slot = ctx.my_slot
        return live_ctx, LiveDraft(live_ctx, SleeperClient())

    def rebuild(path: Path):
        b = pipeline.project(ctx, sim, paths, adp_from=service.engine.board.players, log=lambda *_: None)
        return b.with_teams(ctx.teams) if b.teams != ctx.teams else b

    service = DraftService(Engine(ctx, board, sim, opponents), live_factory)
    return create_app(service, rebuild=rebuild, overrides_path=paths.configs / "minutes_overrides.csv",
                      poll_seconds=sim.poll_seconds)
