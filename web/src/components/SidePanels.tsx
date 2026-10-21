import { useEffect, useState } from "react";
import { api, fmt, type BoardPlayer, type DraftState, type League } from "../api";

export function Roster({ state, league }: { state: DraftState; league: League }) {
  const filled = state.roster.filter((r) => r.sleeper_id).length;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Your roster</h2>
        <span className="small muted num">
          {filled}/{league.rounds}
        </span>
      </div>
      <div className="panel-body lineup">
        {state.roster.map((r, i) => (
          <div className="slot" key={`${r.slot}-${i}`}>
            <span className="pos">{r.slot}</span>
            {r.name ? <span>{r.name}</span> : <span className="empty">open</span>}
          </div>
        ))}
        <div className="small faint" style={{ marginTop: 4 }}>
          Your picks: {state.my_picks.slice(0, 8).join(", ")}
          {state.my_picks.length > 8 ? "…" : ""}
        </div>
      </div>
    </section>
  );
}

interface WatchProps {
  watch: string[];
  state: DraftState;
  byId: Map<string, BoardPlayer>;
  drafted: Set<string>;
  onSelect: (id: string) => void;
  onWatch: (id: string) => void;
}

export function Watchlist({ watch, state, byId, drafted, onSelect, onWatch }: WatchProps) {
  if (watch.length === 0) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Watchlist</h2>
        <span className="spacer" />
        {state.survival_to && <span className="small muted">to pick {state.survival_to}</span>}
      </div>
      <div className="panel-body lineup">
        {watch.map((id) => {
          const p = byId.get(id);
          if (!p) return null;
          return (
            <div key={id} style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ flex: 1, cursor: "pointer", textDecoration: drafted.has(id) ? "line-through" : "none" }} onClick={() => onSelect(id)}>
                {p.name}
              </span>
              <span className="num small">{drafted.has(id) ? "drafted" : fmt.pct(state.survival[id])}</span>
              <button className="btn ghost small" onClick={() => onWatch(id)} aria-label="Remove">
                ✕
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function RoomOdds({ state }: { state: DraftState }) {
  const [odds, setOdds] = useState<{ player: string; prob: number }[]>([]);
  useEffect(() => {
    api.odds(state.id).then(setOdds).catch(() => setOdds([]));
  }, [state.id, state.version]);
  if (odds.length === 0) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>{state.on_the_clock_team} is likely to take</h2>
      </div>
      <div className="panel-body lineup">
        {odds.map((o) => (
          <div key={o.player} style={{ display: "grid", gridTemplateColumns: "1fr 80px 40px", gap: 8, alignItems: "center" }}>
            <span>{o.player}</span>
            <span className="meter">
              <span style={{ width: `${Math.round(o.prob * 100)}%` }} />
            </span>
            <span className="num small" style={{ textAlign: "right" }}>
              {fmt.pct(o.prob)}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function DraftLog({ state }: { state: DraftState }) {
  const picks = [...state.picks].reverse().slice(0, 60);
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Draft log</h2>
        <span className="small muted num">{state.picks.length} picks</span>
      </div>
      <div className="panel-body log">
        {picks.length === 0 && <span className="muted">No picks yet.</span>}
        {picks.map((p) => (
          <div key={p.pick} className={`row ${p.slot === state.my_slot ? "mine" : ""}`}>
            <span className="num faint">{p.pick}</span>
            <span title={p.team}>{p.team}</span>
            <span title={`${p.name} (${fmt.pos(p.positions)})`}>
              {p.name} <span className="faint">{fmt.pos(p.positions)}</span>
            </span>
            <span className="num faint" title="Pick minus ADP: positive means he fell">
              {p.adp !== null ? fmt.signed(p.pick - p.adp, 0) : ""}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
