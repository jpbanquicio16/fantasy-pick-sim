import { fmt, type BoardPlayer, type Recommendation } from "../api";

interface Props {
  rec: Recommendation | null;
  thinking: boolean;
  useRollout: boolean;
  byId: Map<string, BoardPlayer>;
  canPick: boolean;
  onPick: (id: string) => void;
  onSelect: (id: string) => void;
}

export default function Recommendations({ rec, thinking, useRollout, byId, canPick, onPick, onSelect }: Props) {
  const ro = rec?.rollout ?? null;
  const order = ro ? ro.candidates.map((c) => c.sleeper_id) : (rec?.candidates ?? []).map((c) => c.sleeper_id);
  const greedy = new Map((rec?.candidates ?? []).map((c) => [c.sleeper_id, c]));
  const rollout = new Map((ro?.candidates ?? []).map((c) => [c.sleeper_id, c]));
  const card = rec?.card ?? null;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>
          {rec?.pick ? `Recommendation for pick ${rec.pick}` : "Recommendation"}
          {rec?.planning_ahead ? <span className="muted"> · planning ahead</span> : null}
        </h2>
        <span className="spacer" />
        {thinking ? (
          <span className="small muted">
            <span className="spinner" /> {useRollout ? "simulating drafts and seasons…" : "thinking…"}
          </span>
        ) : ro ? (
          <span className="small muted num">
            {ro.worlds} simulated drafts and seasons · {ro.elapsed.toFixed(1)} s
          </span>
        ) : null}
      </div>
      <div className="panel-body">
        {!rec || order.length === 0 ? (
          <p className="muted">{thinking ? "…" : "No picks left for you."}</p>
        ) : (
          <>
            <div className="cards">
              {order.slice(0, 3).map((id, i) => {
                const c = greedy.get(id);
                const r = rollout.get(id);
                const p = byId.get(id);
                if (!c || !p) return null;
                const surv = r?.survival_sim ?? c.survival;
                return (
                  <article key={id} className={`card ${i === 0 ? "top" : ""}`}>
                    <div className="rank">#{i + 1}</div>
                    <div>
                      <span className="name" onClick={() => onSelect(id)}>
                        {p.name}
                      </span>{" "}
                      {p.rookie ? <span className="badge rookie">rookie</span> : null}{" "}
                      {p.injury_status ? <span className="badge injury">{p.injury_status}</span> : null}
                      <div className="sub">
                        {fmt.pos(p.positions)} · {p.team_now ?? "FA"} · ADP {fmt.n(p.adp, 0)}
                      </div>
                    </div>
                    <div className="headline">
                      {i === 0 ? (
                        ro && ro.confidence !== null ? (
                          <div className="confidence" title="Share of simulated drafts and seasons in which taking him finished with a better roster than taking #2">
                            <strong className="num">{fmt.pct(ro.confidence)}</strong>
                            <span className="meter">
                              <span style={{ width: `${Math.round((ro.confidence ?? 0) * 100)}%` }} />
                            </span>
                            <span className="num">{fmt.signed(ro.edge_pw)} FP/wk</span>
                          </div>
                        ) : (
                          <strong>Top pick</strong>
                        )
                      ) : r ? (
                        <>
                          <strong className="num">{fmt.n(r.gap_pw)}</strong> FP/week behind #1 · best in{" "}
                          {fmt.pct(r.p_best)} of simulations{r.dropped ? " · dropped early" : ""}
                        </>
                      ) : (
                        <>
                          <strong className="num">{fmt.n(c.score_gap_pw)}</strong> FP/week behind #1
                        </>
                      )}
                    </div>
                    <div className="metrics">
                      <div className="metric" title="Projected fantasy points per game (season P10-P90 below)">
                        <div className="label">FP/G</div>
                        <div className="value">{fmt.n(p.fp_pg)}</div>
                        <div className="note num">
                          {fmt.n(p.fp_p10, 0)}–{fmt.n(p.fp_p90, 0)}
                        </div>
                      </div>
                      <div className="metric" title="Expected locked score per slot-week with 3 games (optimal stopping)">
                        <div className="label">Lock-In 3g</div>
                        <div className="value">{fmt.n(p.lockin_3)}</div>
                        <div className="note num">{fmt.signed(p.lockin_premium_3)} spread</div>
                      </div>
                      <div className="metric" title="Marginal value to your roster, per week">
                        <div className="label">Adds/wk</div>
                        <div className="value">{fmt.n(c.delta_pw)}</div>
                        <div className="note num">VOR {fmt.signed(p.vor_per_week)}</div>
                      </div>
                    </div>
                    <div className="lines">
                      {rec.planning_ahead && <span>Still there at pick {rec.pick}: {fmt.pct(r?.reach ?? c.reach)}</span>}
                      {rec.next_pick && (
                        <span>
                          Survives to pick {rec.next_pick}: <b className="num">{fmt.pct(surv)}</b>
                        </span>
                      )}
                      <span>
                        Games {fmt.pct(p.share_hat)} · {fmt.n(p.mpg_hat)} min
                      </span>
                    </div>
                    <div className="actions">
                      <button className="btn primary small" disabled={!canPick} onClick={() => onPick(id)}>
                        Draft
                      </button>
                      <button className="btn small" onClick={() => onSelect(id)}>
                        Details
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
            {card && (card.why.length > 0 || card.risk.length > 0 || card.comps.length > 0) && (
              <div className="explain">
                {card.why.length > 0 && (
                  <div>
                    <b>Why:</b> {card.why.join(" ")}
                  </div>
                )}
                {card.risk.length > 0 && (
                  <div>
                    <b>Risk:</b> {card.risk.join("; ")}.
                  </div>
                )}
                {card.comps.length > 0 && (
                  <div>
                    <b>Rookies like him</b> (year-one FP/G): {card.comps.join("; ")}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
