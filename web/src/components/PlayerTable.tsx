import { useMemo, useState } from "react";
import { fmt, type BoardPlayer, type DraftState } from "../api";

type Key = "vor_per_week" | "adp" | "fp_pg" | "lockin_3" | "mpg_hat" | "share_hat" | "survival" | "name";
const FILTERS = ["All", "PG", "SG", "SF", "PF", "C", "Rookies", "Watchlist"] as const;

interface Props {
  board: BoardPlayer[];
  state: DraftState;
  drafted: Set<string>;
  watch: string[];
  selected: string | null;
  canPick: boolean;
  onSelect: (id: string) => void;
  onPick: (id: string) => void;
  onWatch: (id: string) => void;
}

export default function PlayerTable({ board, state, drafted, watch, selected, canPick, onSelect, onPick, onWatch }: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("All");
  const [sort, setSort] = useState<{ key: Key; desc: boolean }>({ key: "vor_per_week", desc: true });
  const [limit, setLimit] = useState(150);

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const value = (p: BoardPlayer): number | string => {
      if (sort.key === "survival") return state.survival[p.sleeper_id] ?? -1;
      if (sort.key === "name") return p.name;
      return (p[sort.key] as number | null) ?? (sort.desc ? -Infinity : Infinity);
    };
    return board
      .filter((p) => !drafted.has(p.sleeper_id))
      .filter((p) => !q || p.name.toLowerCase().includes(q) || (p.team_now ?? "").toLowerCase() === q)
      .filter((p) =>
        filter === "All"
          ? true
          : filter === "Rookies"
            ? !!p.rookie
            : filter === "Watchlist"
              ? watch.includes(p.sleeper_id)
              : p.positions.includes(filter),
      )
      .sort((a, b) => {
        const x = value(a);
        const y = value(b);
        const c = typeof x === "string" ? x.localeCompare(y as string) : (x as number) - (y as number);
        return sort.desc ? -c : c;
      });
  }, [board, drafted, query, filter, sort, state.survival, watch]);

  const head = (key: Key, label: string, title?: string, left = false) => (
    <th
      className={`${sort.key === key ? "sorted" : ""} ${left ? "left" : ""}`}
      title={title}
      onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "adp" && key !== "name" }))}
    >
      {label}
      {sort.key === key ? (sort.desc ? " ▾" : " ▴") : ""}
    </th>
  );

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Available players</h2>
        <span className="small muted num">{rows.length}</span>
        <span className="spacer" />
        {state.survival_to && <span className="small muted">Survival: to pick {state.survival_to}</span>}
      </div>
      <div className="table-tools">
        <input
          className="input"
          placeholder="Search player or team"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: "1 1 180px" }}
        />
        <div className="chips">
          {FILTERS.map((f) => (
            <button key={f} className={`chip ${filter === f ? "on" : ""}`} onClick={() => setFilter(f)}>
              {f}
            </button>
          ))}
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {head("name", "Player", undefined, true)}
              <th className="left hide-sm">Pos</th>
              {head("adp", "ADP", "Average draft position (market)")}
              {head("fp_pg", "FP/G", "Projected fantasy points per game")}
              {head("lockin_3", "Lock-In", "Expected locked score per slot-week with 3 games")}
              {head("mpg_hat", "Min", "Projected minutes per game")}
              {head("share_hat", "Games", "Projected share of games played")}
              {head("vor_per_week", "VOR/wk", "Value over replacement per week")}
              {head("survival", "Surv", "Chance he is still there at your next pick")}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, limit).map((p) => {
              const s = state.survival[p.sleeper_id];
              const starred = watch.includes(p.sleeper_id);
              return (
                <tr key={p.sleeper_id} className={selected === p.sleeper_id ? "selected" : ""} onClick={() => onSelect(p.sleeper_id)}>
                  <td className="player">
                    <span
                      className={`star ${starred ? "on" : ""}`}
                      title={starred ? "Remove from watchlist" : "Add to watchlist"}
                      onClick={(e) => {
                        e.stopPropagation();
                        onWatch(p.sleeper_id);
                      }}
                    >
                      ★
                    </span>
                    {p.name} <span className="faint small">{p.team_now ?? ""}</span>{" "}
                    {p.rookie ? <span className="badge rookie">R</span> : null}{" "}
                    {p.injury_status ? <span className="badge injury">{p.injury_status}</span> : null}
                  </td>
                  <td className="left hide-sm muted">{fmt.pos(p.positions)}</td>
                  <td className="num">{fmt.n(p.adp, 0)}</td>
                  <td className="num">{fmt.n(p.fp_pg)}</td>
                  <td className="num">{fmt.n(p.lockin_3)}</td>
                  <td className="num">{fmt.n(p.mpg_hat)}</td>
                  <td className="num">{fmt.pct(p.share_hat)}</td>
                  <td className="num">{fmt.signed(p.vor_per_week)}</td>
                  <td className={`num ${s === undefined ? "" : s >= 0.7 ? "surv-hi" : s <= 0.3 ? "surv-lo" : ""}`}>{fmt.pct(s)}</td>
                  <td>
                    <button
                      className="btn small"
                      disabled={!canPick}
                      onClick={(e) => {
                        e.stopPropagation();
                        onPick(p.sleeper_id);
                      }}
                    >
                      Draft
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows.length > limit && (
          <div className="panel-body">
            <button className="btn small" onClick={() => setLimit((l) => l + 150)}>
              Show more
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
