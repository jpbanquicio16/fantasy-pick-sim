import type { DraftState } from "../api";

interface Props {
  state: DraftState;
  busy: boolean;
  useRollout: boolean;
  onRollout: (v: boolean) => void;
  onUndo: () => void;
  onAdvance: () => void;
  onAutodraft: (rollout: boolean) => void;
  onSync: () => void;
}

export default function Banner({ state, busy, useRollout, onRollout, onUndo, onAdvance, onAutodraft, onSync }: Props) {
  const s = state;
  const status = s.complete
    ? "Draft complete"
    : s.my_turn
      ? "You are on the clock"
      : s.picks_until_mine !== null
        ? `Your next pick is ${s.my_next_pick} (${s.picks_until_mine} away)`
        : "No picks left for you";
  return (
    <div className={`banner ${s.my_turn && !s.complete ? "mine" : ""}`}>
      <div>
        <div className="big">
          {s.complete ? "Done" : `Pick ${s.current_pick}`}
          <span className="muted small num"> of {s.total_picks}</span>
        </div>
        <div className="small muted">
          {s.kind === "live" ? (
            <span className="badge live">live{s.draft_status ? `: ${s.draft_status}` : ""}</span>
          ) : (
            <span className="badge">{s.kind === "mock" ? `mock${s.seed !== null ? ` · seed ${s.seed}` : ""}` : "manual"}</span>
          )}{" "}
          slot {s.my_slot}
          {!s.complete && s.on_the_clock_team && !s.my_turn ? ` · on the clock: ${s.on_the_clock_team}` : ""}
        </div>
      </div>
      <div className="big" style={{ fontSize: 15 }}>
        {status}
      </div>
      <span className="spacer" />
      {!s.complete && (
        <label className="toggle" title="Simulate drafts and seasons for each shortlisted player: adds a confidence (P(A's roster beats B's)). One to two seconds.">
          <input type="checkbox" checked={useRollout} onChange={(e) => onRollout(e.target.checked)} />
          Monte Carlo rollouts
        </label>
      )}
      {s.kind === "live" && (
        <button className="btn" onClick={onSync} disabled={busy}>
          Refresh
        </button>
      )}
      {s.kind !== "live" && (
        <button className="btn" onClick={onUndo} disabled={busy || s.picks.length === 0}>
          Undo{s.kind === "mock" ? " my pick" : ""}
        </button>
      )}
      {s.kind === "mock" && !s.auto && !s.my_turn && !s.complete && (
        <button className="btn" onClick={onAdvance} disabled={busy}>
          Let the room pick
        </button>
      )}
      {s.kind === "mock" && !s.complete && (
        <button className="btn" onClick={() => onAutodraft(useRollout)} disabled={busy}>
          {busy ? <span className="spinner" /> : null} Auto-draft the rest
        </button>
      )}
    </div>
  );
}
