import { useState } from "react";
import { api, type DraftState, type League } from "../api";

type Mode = "mock" | "live" | "manual";

export default function Setup({ league, onStart }: { league: League; onStart: (s: DraftState) => void }) {
  const [mode, setMode] = useState<Mode>("mock");
  const [slot, setSlot] = useState(league.my_slot ?? 1);
  const [seed, setSeed] = useState("");
  const [auto, setAuto] = useState(true);
  const [draftId, setDraftId] = useState(league.draft_id ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await api.newDraft(
        mode === "live"
          ? { mode, draft_id: draftId.trim() }
          : { mode, slot, seed: seed.trim() === "" ? null : Number(seed), auto },
      );
      onStart(s);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="setup">
      <h1>Draft with the engine</h1>
      <p className="lead">
        Projections, Lock-In values and survival odds for {league.players} players. Recommendations weigh what a
        player adds to your lineup against the chance he is still there at your next pick; Monte Carlo rollouts add
        a calibrated confidence.
      </p>
      <section className="panel">
        <div className="tabs" role="tablist">
          {(
            [
              ["mock", "Offline mock"],
              ["live", "Live Sleeper draft"],
              ["manual", "Manual entry"],
            ] as [Mode, string][]
          ).map(([m, label]) => (
            <button key={m} role="tab" className={`tab ${mode === m ? "on" : ""}`} onClick={() => setMode(m)}>
              {label}
            </button>
          ))}
        </div>
        <div className="form">
          {mode === "live" ? (
            <div className="field">
              <label htmlFor="draft-id">Sleeper draft or mock-draft id</label>
              <input
                id="draft-id"
                className="input"
                value={draftId}
                onChange={(e) => setDraftId(e.target.value)}
                placeholder="e.g. 1274194656334204928"
              />
              <span className="small muted">
                Picks are followed by polling Sleeper every few seconds; your slot comes from your username.
              </span>
            </div>
          ) : (
            <div className="row-inline">
              <div className="field">
                <label htmlFor="slot">Your draft slot</label>
                <select id="slot" className="input" value={slot} onChange={(e) => setSlot(Number(e.target.value))}>
                  {Array.from({ length: league.teams }, (_, i) => i + 1).map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>
              {mode === "mock" && (
                <div className="field">
                  <label htmlFor="seed">Room seed (optional)</label>
                  <input
                    id="seed"
                    className="input"
                    inputMode="numeric"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
                    placeholder="random"
                    style={{ width: 120 }}
                  />
                </div>
              )}
            </div>
          )}
          {mode === "mock" && (
            <label className="toggle">
              <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
              Other teams draft automatically (the opponent model)
            </label>
          )}
          {mode === "manual" && (
            <span className="small muted">Enter every pick yourself, for any team: for a draft run elsewhere.</span>
          )}
          {error && <div className="error">{error}</div>}
          <div>
            <button className="btn primary" onClick={start} disabled={busy || (mode === "live" && !draftId.trim())}>
              {busy ? <span className="spinner" /> : null} Start
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
