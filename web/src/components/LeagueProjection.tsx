import { useEffect, useState } from "react";
import { api, fmt, type DraftState, type TeamProjection } from "../api";

export default function LeagueProjection({ state }: { state: DraftState }) {
  const [rows, setRows] = useState<TeamProjection[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.projection(state.id).then(setRows).catch((e) => setError(String(e.message)));
  }, [state.id, state.version]);

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>The league, projected</h2>
        <span className="spacer" />
        <span className="small muted">simulated seasons; your own model drafted your team, so compare mocks, not forecasts</span>
      </div>
      {error && <div className="panel-body error">{error}</div>}
      {!rows && !error && (
        <div className="panel-body muted">
          <span className="spinner" /> Simulating seasons…
        </div>
      )}
      {rows && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="left">#</th>
                <th className="left">Team</th>
                <th>Projected FP/week</th>
                <th>P(playoffs)</th>
                <th>P(title)</th>
                <th>Mean seed</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.slot} className={r.slot === state.my_slot ? "selected" : ""}>
                  <td className="left num">{r.proj_rank}</td>
                  <td className="left player">{r.team}</td>
                  <td className="num">{fmt.n(r.proj_per_week)}</td>
                  <td className="num">{fmt.pct(r.p_playoffs)}</td>
                  <td className="num">{fmt.pct(r.p_title)}</td>
                  <td className="num">{fmt.n(r.mean_seed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
