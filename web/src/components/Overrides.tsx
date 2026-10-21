import { useEffect, useMemo, useState } from "react";
import { api, type BoardPlayer, type MinutesOverride } from "../api";

/** Minutes per game for preseason news the model cannot know. Saving re-projects the board
 * (a few seconds): teammates absorb the change, mostly at his positions. */
export default function Overrides({ board, onSaved }: { board: BoardPlayer[]; onSaved: () => void }) {
  const [rows, setRows] = useState<MinutesOverride[] | null>(null);
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const byName = useMemo(() => new Map(board.map((p) => [p.name.toLowerCase(), p])), [board]);
  const byId = useMemo(() => new Map(board.map((p) => [p.sleeper_id, p])), [board]);

  useEffect(() => {
    if (open && rows === null) api.overrides().then(setRows).catch(() => setRows([]));
  }, [open, rows]);

  const add = () => {
    const p = byName.get(name.trim().toLowerCase());
    if (!p || !rows || rows.some((r) => r.sleeper_id === p.sleeper_id)) return;
    setRows([...rows, { sleeper_id: p.sleeper_id, name: p.name, mpg: Math.round((p.mpg_hat ?? 20) * 10) / 10, note: "" }]);
    setName("");
  };

  const save = async () => {
    if (!rows) return;
    setBusy(true);
    setNote(null);
    try {
      const r = await api.saveOverrides(rows);
      setNote(`Saved ${r.saved}; board rebuilt (${r.players} players).`);
      onSaved();
    } catch (e) {
      setNote(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-head" style={{ cursor: "pointer" }} onClick={() => setOpen((o) => !o)}>
        <h2>Minutes overrides</h2>
        <span className="spacer" />
        <span className="small muted">{open ? "▾" : "▸"}</span>
      </div>
      {open && (
        <div className="panel-body col" style={{ gap: 10 }}>
          <span className="small muted">
            Expected minutes for news the model cannot see (a starting job, a move to the bench). Teammates absorb the
            change; the model and its evaluation are untouched.
          </span>
          {rows === null ? (
            <span className="muted">
              <span className="spinner" /> Loading…
            </span>
          ) : (
            rows.map((r, i) => (
              <div key={r.sleeper_id} style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 64px minmax(0,1fr) auto", gap: 6, alignItems: "center" }}>
                <span title={`model: ${byId.get(r.sleeper_id)?.mpg_hat?.toFixed(1) ?? "?"} min`}>{r.name ?? byId.get(r.sleeper_id)?.name ?? r.sleeper_id}</span>
                <input
                  className="input num"
                  type="number"
                  min={0}
                  max={48}
                  step={0.5}
                  value={r.mpg}
                  onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, mpg: Number(e.target.value) } : x)))}
                />
                <input
                  className="input"
                  placeholder="note"
                  value={r.note ?? ""}
                  onChange={(e) => setRows(rows.map((x, j) => (j === i ? { ...x, note: e.target.value } : x)))}
                />
                <button className="btn ghost small" aria-label="Remove" onClick={() => setRows(rows.filter((_, j) => j !== i))}>
                  ✕
                </button>
              </div>
            ))
          )}
          <div style={{ display: "flex", gap: 6 }}>
            <input
              className="input"
              list="override-players"
              placeholder="Add a player"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && add()}
              style={{ flex: 1 }}
            />
            <datalist id="override-players">
              {board.slice(0, 400).map((p) => (
                <option key={p.sleeper_id} value={p.name} />
              ))}
            </datalist>
            <button className="btn small" onClick={add}>
              Add
            </button>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button className="btn primary small" onClick={save} disabled={busy || rows === null}>
              {busy ? <span className="spinner" /> : null} Save and rebuild the board
            </button>
            {note && <span className="small muted">{note}</span>}
          </div>
        </div>
      )}
    </section>
  );
}
