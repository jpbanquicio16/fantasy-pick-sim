import { useEffect, useState } from "react";
import { api, fmt, type DraftState, type PlayerDetail, type WhatIf } from "../api";

interface Props {
  id: string;
  state: DraftState;
  drafted: boolean;
  watched: boolean;
  canPick: boolean;
  onPick: (id: string) => void;
  onWatch: (id: string) => void;
  onClose: () => void;
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 320;
  const h = 64;
  const max = Math.max(...values, 1e-9);
  const pts = values.map((v, i) => [(i / (values.length - 1)) * w, h - 4 - (v / max) * (h - 10)]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  const area = `${line}L${w},${h}L0,${h}Z`;
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label="Expected locked score by week">
      <path className="area" d={area} />
      <path className="line" d={line} />
    </svg>
  );
}

function Range({ lo, mid, hi, max }: { lo: number | null; mid: number | null; hi: number | null; max: number }) {
  if (lo === null || hi === null || mid === null) return null;
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`;
  return (
    <div className="range" title={`P10 ${lo.toFixed(1)} · projection ${mid.toFixed(1)} · P90 ${hi.toFixed(1)}`}>
      <div className="band" style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }} />
      <div className="dot" style={{ left: pct(mid) }} />
    </div>
  );
}

export default function PlayerPanel({ id, state, drafted, watched, canPick, onPick, onWatch, onClose }: Props) {
  const [p, setP] = useState<PlayerDetail | null>(null);
  const [wi, setWi] = useState<WhatIf | null>(null);

  useEffect(() => {
    setP(null);
    api.player(id).then(setP).catch(() => setP(null));
  }, [id]);

  useEffect(() => {
    setWi(null);
    if (drafted || state.complete) return;
    api.whatIf(state.id, id).then(setWi).catch(() => setWi(null));
  }, [id, state.id, state.version, drafted, state.complete]);

  if (!p) {
    return (
      <section className="panel">
        <div className="panel-body muted">
          <span className="spinner" /> Loading…
        </div>
      </section>
    );
  }
  const maxDriver = Math.max(1, ...p.drivers.map((d) => Math.abs(d.value)));
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>
          {p.name} {p.rookie ? <span className="badge rookie">rookie</span> : null}
        </h2>
        <span className="spacer" />
        <button className="btn ghost small" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </div>
      <div className="panel-body col" style={{ gap: 12 }}>
        <div className="small muted">
          {fmt.pos(p.positions)} · {p.team_now ?? "FA"}
          {p.age ? ` · age ${p.age.toFixed(0)}` : ""}
          {p.draft_pick ? ` · pick ${p.draft_pick.toFixed(0)}` : ""}
          {p.injury_status ? ` · ${p.injury_status}${p.injury_body_part ? ` (${p.injury_body_part})` : ""}` : ""}
        </div>
        <div>
          <div className="small muted">FP/G with its season P10–P90</div>
          <div className="num" style={{ fontSize: 20, fontWeight: 700 }}>
            {fmt.n(p.fp_pg)} <span className="small muted">({fmt.n(p.fp_p10)} to {fmt.n(p.fp_p90)})</span>
          </div>
          <Range lo={p.fp_p10} mid={p.fp_pg} hi={p.fp_p90} max={Math.max(60, (p.fp_p90 ?? 0) * 1.1)} />
        </div>
        <dl className="kv">
          <dt>Lock-In 2/3/4 games</dt>
          <dd>
            {fmt.n(p.lockin_2)} / {fmt.n(p.lockin_3)} / {fmt.n(p.lockin_4)}
          </dd>
          <dt>40+ / 50+ game</dt>
          <dd>
            {fmt.pct(p.p_game_40)} / {fmt.pct(p.p_game_50)}
          </dd>
          <dt>Games, minutes</dt>
          <dd>
            {fmt.pct(p.share_hat)} of games{p.minutes ? ` · ${p.minutes}` : ""}
          </dd>
          <dt>VOR</dt>
          <dd>{fmt.signed(p.vor_per_week)} per week</dd>
          <dt>ADP</dt>
          <dd>
            {fmt.n(p.adp, 0)}
            {state.survival[id] !== undefined && state.survival_to ? ` · ${fmt.pct(state.survival[id])} to survive to ${state.survival_to}` : ""}
          </dd>
          {wi && wi.score_gap_pw !== undefined && (
            <>
              <dt>Against the top pick</dt>
              <dd>
                {fmt.n(wi.score_gap_pw)} FP/week behind; adds {fmt.n(wi.delta_pw)} to your lineup
              </dd>
            </>
          )}
          {wi && wi.in_pool === false && (
            <>
              <dt>Against the top pick</dt>
              <dd className="muted">outside the candidate pool: well below the top pick</dd>
            </>
          )}
        </dl>
        {p.role && <div className="small">Role: {p.role}</div>}
        {p.drivers.length > 0 && (
          <div>
            <div className="small muted" style={{ marginBottom: 4 }}>
              Projection drivers (TreeSHAP, FP/G against the average player)
            </div>
            <div className="drivers">
              {p.drivers.map((d) => (
                <div className="bar" key={d.group}>
                  <span>{d.group}</span>
                  <span className="track">
                    <span
                      className={`fill ${d.value >= 0 ? "pos" : "neg"}`}
                      style={{ width: `${(Math.abs(d.value) / maxDriver) * 50}%` }}
                    />
                  </span>
                  <span className="num" style={{ textAlign: "right" }}>
                    {fmt.signed(d.value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        {p.comps && p.comps.length > 0 && (
          <div className="small">
            <span className="muted">Comparable rookies (year-one FP/G):</span> {p.comps.join("; ")}
          </div>
        )}
        <div>
          <div className="small muted">Expected locked score by week (schedule and rest)</div>
          <Sparkline values={p.weekly.value} />
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn primary small" disabled={!canPick || drafted} onClick={() => onPick(id)}>
            {drafted ? "Drafted" : "Draft"}
          </button>
          <button className="btn small" onClick={() => onWatch(id)}>
            {watched ? "Unwatch" : "Watch"}
          </button>
        </div>
      </div>
    </section>
  );
}
