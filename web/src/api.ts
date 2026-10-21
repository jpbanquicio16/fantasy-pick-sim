// Typed client for the draftai API (src/draftai/api/app.py). Every number is per week
// unless a field says otherwise (fp_pg: fantasy points per game).

export interface League {
  season: number;
  name: string;
  teams: number;
  rounds: number;
  draft_type: string;
  reversal_round: number;
  my_slot: number | null;
  roster_positions: string[];
  slots: string[];
  playoff_teams: number;
  bonus_stacking: boolean;
  draft_id: string | null;
  players: number;
  weeks_weight: number;
}

export interface BoardPlayer {
  row: number;
  sleeper_id: string;
  name: string;
  team_now: string | null;
  positions: string[];
  adp: number | null;
  fp_pg: number | null;
  fp_p10: number | null;
  fp_p90: number | null;
  lockin_3: number | null;
  lockin_premium_3: number | null;
  mpg_hat: number | null;
  share_hat: number | null;
  games_proj: number | null;
  vor_per_week: number | null;
  season_value: number | null;
  season_p10: number | null;
  season_p90: number | null;
  rookie: boolean | null;
  injury_status: string | null;
  age: number | null;
}

export interface PlayerDetail extends BoardPlayer {
  lockin_2?: number | null;
  lockin_4?: number | null;
  p_game_40?: number | null;
  p_game_50?: number | null;
  rookie_source?: string | null;
  comps?: string[] | null;
  draft_pick?: number | null;
  injury_body_part?: string | null;
  minutes: string | null;
  role: string | null;
  drivers: { group: string; value: number }[];
  weekly: { weeks: number[]; value: number[] };
}

export interface PickRow {
  pick: number;
  round: number;
  slot: number;
  team: string;
  sleeper_id: string;
  name: string;
  positions: string[];
  adp: number | null;
  by: string;
}

export interface DraftState {
  id: string;
  kind: "mock" | "manual" | "live";
  version: number;
  teams: number;
  rounds: number;
  my_slot: number;
  current_pick: number;
  total_picks: number;
  complete: boolean;
  on_the_clock: number | null;
  on_the_clock_team: string | null;
  my_turn: boolean;
  my_next_pick: number | null;
  picks_until_mine: number | null;
  survival_to: number | null;
  my_picks: number[];
  picks: PickRow[];
  drafted: string[];
  survival: Record<string, number>;
  roster: { slot: string; sleeper_id: string | null; name: string | null }[];
  seed: number | null;
  auto: boolean;
  draft_status: string | null;
}

export interface Candidate {
  rank: number;
  sleeper_id: string;
  name: string;
  positions: string[];
  team: string | null;
  adp: number;
  delta_pw: number;
  e_next_pw: number;
  score_gap_pw: number;
  survival: number;
  reach: number;
  fp_pg: number | null;
  lockin_3: number | null;
  vor_per_week: number | null;
  rookie: boolean | null;
}

export interface RolloutCandidate {
  rank: number;
  sleeper_id: string;
  name: string;
  positions: string[];
  value_pw: number;
  gap_pw: number;
  se_pw: number;
  p_best: number;
  reach: number;
  dropped: boolean;
  survival_sim: number | null;
  greedy_rank: number;
}

export interface Card {
  pick: number;
  player: string;
  runner_up: string | null;
  edge_per_week: number | null;
  confidence: number | null;
  edge_se: number | null;
  survival: number;
  runner_up_survival: number | null;
  next_pick: number | null;
  why: string[];
  risk: string[];
  comps: string[];
  minutes: string | null;
  role: string | null;
  drivers: { group: string; value: number }[];
  rollout_worlds: number | null;
}

export interface Recommendation {
  version: number;
  pick: number | null;
  next_pick: number | null;
  planning_ahead: boolean;
  candidates: Candidate[];
  card: Card | null;
  rollout: null | {
    worlds: number;
    elapsed: number;
    confidence: number | null;
    edge_pw: number | null;
    edge_se_pw: number | null;
    candidates: RolloutCandidate[];
  };
}

export interface TeamProjection {
  proj_rank: number;
  slot: number;
  team: string;
  proj_per_week: number;
  p_playoffs: number;
  p_title: number;
  mean_seed: number;
}

export interface MinutesOverride {
  sleeper_id: string;
  name: string | null;
  mpg: number;
  note: string | null;
}

export interface WhatIf {
  sleeper_id: string;
  name: string;
  delta_pw?: number;
  score_gap_pw?: number;
  survival?: number;
  in_pool?: boolean;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return (await res.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  call<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  league: () => call<League>("/league"),
  board: () => call<BoardPlayer[]>("/board"),
  player: (id: string) => call<PlayerDetail>(`/players/${encodeURIComponent(id)}`),
  newDraft: (body: { mode: string; slot?: number; seed?: number | null; auto?: boolean; draft_id?: string }) =>
    post<DraftState>("/drafts", body),
  draft: (id: string) => call<DraftState>(`/drafts/${id}`),
  pick: (id: string, sleeper_id: string) => post<DraftState>(`/drafts/${id}/picks`, { sleeper_id }),
  undo: (id: string) => post<DraftState>(`/drafts/${id}/undo`),
  advance: (id: string) => post<DraftState>(`/drafts/${id}/advance`),
  autodraft: (id: string, rollout: boolean) => post<DraftState>(`/drafts/${id}/autodraft`, { rollout }),
  sync: (id: string) => post<DraftState>(`/drafts/${id}/sync`),
  recommendation: (id: string, rollout: boolean) =>
    call<Recommendation>(`/drafts/${id}/recommendation?rollout=${rollout}`),
  whatIf: (id: string, sleeper_id: string) => call<WhatIf>(`/drafts/${id}/what-if/${encodeURIComponent(sleeper_id)}`),
  odds: (id: string) => call<{ player: string; prob: number }[]>(`/drafts/${id}/odds`),
  projection: (id: string) => call<TeamProjection[]>(`/drafts/${id}/projection`),
  overrides: () => call<MinutesOverride[]>("/overrides/minutes"),
  saveOverrides: (rows: MinutesOverride[]) =>
    call<{ saved: number; players: number }>("/overrides/minutes", { method: "PUT", body: JSON.stringify(rows) }),
};

/** State pushed by the server on every change (and polled from Sleeper for live drafts). */
export function subscribe(id: string, onState: (s: DraftState) => void): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws: WebSocket | null = null;
  let closed = false;
  let retry: number | undefined;
  const open = () => {
    ws = new WebSocket(`${proto}://${location.host}/api/drafts/${id}/ws`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === "state") onState(msg.state as DraftState);
    };
    ws.onclose = () => {
      if (!closed) retry = window.setTimeout(open, 2000);
    };
  };
  open();
  return () => {
    closed = true;
    window.clearTimeout(retry);
    ws?.close();
  };
}

export const fmt = {
  n: (v: number | null | undefined, d = 1) => (v === null || v === undefined ? "–" : v.toFixed(d)),
  signed: (v: number | null | undefined, d = 1) =>
    v === null || v === undefined ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`,
  pct: (v: number | null | undefined) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`),
  pos: (p: string[] | null | undefined) => (p && p.length ? p.join("/") : "–"),
};
