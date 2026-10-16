"""Follow a live Sleeper draft: poll picks and convert them to a ``DraftState``.

Contains no decision logic. Polling every few seconds is far below Sleeper's limit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from draftai.draft.league import LeagueContext
from draftai.draft.state import DraftState
from draftai.ingestion.sleeper import SleeperClient, SleeperPick


def state_from_picks(ctx: LeagueContext, picks: list[SleeperPick]) -> DraftState:
    """Rebuild the state from the full pick list.

    A pick's ``draft_slot`` is its board column (the original owner), so a traded pick
    is attributed by ``roster_id`` instead; without a roster mapping, the pick order
    (which already applies traded picks) decides.
    """
    state = ctx.new_state()
    roster_to_slot = {r: s for s, r in ctx.slot_to_roster.items()}
    for p in sorted(picks, key=lambda x: x.pick_no):
        if p.pick_no != state.current_pick:
            raise ValueError(f"pick {p.pick_no} arrived out of order (expected {state.current_pick})")
        slot = roster_to_slot.get(p.roster_id) if p.roster_id is not None else None
        state.add_pick(str(p.player_id), slot=slot)
    return state


@dataclass
class LiveDraft:
    ctx: LeagueContext
    client: SleeperClient
    state: DraftState | None = None
    status: str | None = None

    def poll(self) -> bool:
        """Refresh from the API; True if a new pick arrived."""
        draft = self.client.draft(self.ctx.draft_id)
        self.status = draft.status
        picks = self.client.draft_picks(self.ctx.draft_id)
        old = self.state.version if self.state else -1
        self.state = state_from_picks(self.ctx, picks)
        return self.state.version != old

    def follow(self, on_change: Callable[[DraftState], None], interval: float = 3.0, max_polls: int | None = None) -> None:
        n = 0
        while max_polls is None or n < max_polls:
            if self.poll():
                on_change(self.state)
            if self.status == "complete" or (self.state and self.state.complete):
                return
            n += 1
            time.sleep(interval)
