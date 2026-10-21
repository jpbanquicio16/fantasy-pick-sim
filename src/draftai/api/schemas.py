"""Request bodies (Pydantic): the API fails loudly on malformed input."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NewDraft(BaseModel):
    mode: str = Field("mock", pattern="^(mock|manual|live)$")
    slot: int | None = Field(None, ge=1, le=32)
    seed: int | None = None
    auto: bool = True  # mocks: the other teams draft themselves
    draft_id: str | None = None  # live: the Sleeper draft (or mock-draft) id


class Pick(BaseModel):
    sleeper_id: str


class AutoDraft(BaseModel):
    rollout: bool = False


class MinutesOverride(BaseModel):
    sleeper_id: str
    name: str | None = None
    mpg: float = Field(..., ge=0, le=48)
    note: str | None = None
