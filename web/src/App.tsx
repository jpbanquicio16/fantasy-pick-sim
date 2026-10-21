import { useEffect, useState } from "react";
import { api, type BoardPlayer, type DraftState, type League } from "./api";
import DraftRoom from "./components/DraftRoom";
import Setup from "./components/Setup";

export default function App() {
  const [league, setLeague] = useState<League | null>(null);
  const [board, setBoard] = useState<BoardPlayer[] | null>(null);
  const [draft, setDraft] = useState<DraftState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.league(), api.board()])
      .then(([l, b]) => {
        setLeague(l);
        setBoard(b);
      })
      .catch((e) => setError(String(e.message ?? e)));
    // Resume the draft in the URL (#draft=<id>) after a reload.
    const id = new URLSearchParams(location.hash.slice(1)).get("draft");
    if (id) api.draft(id).then(setDraft).catch(() => (location.hash = ""));
  }, []);

  // After a minutes-override change the server re-projects; fetch the new board.
  const reloadBoard = () => api.board().then(setBoard).catch((e) => setError(String(e.message ?? e)));

  const open = (s: DraftState | null) => {
    setDraft(s);
    location.hash = s ? `draft=${s.id}` : "";
  };

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">L</span>
          Lock-In Draft Assistant
        </div>
        {league && (
          <span className="meta">
            {league.name !== "offline" ? `${league.name} · ` : ""}
            {league.season}-{String((league.season + 1) % 100).padStart(2, "0")} · {league.teams} teams ·{" "}
            {league.rounds} rounds · {league.draft_type}
            {league.reversal_round ? ` (reversal from round ${league.reversal_round})` : ""}
          </span>
        )}
        <span className="spacer" />
        {draft && (
          <button className="btn" onClick={() => open(null)}>
            New draft
          </button>
        )}
      </header>
      <main className="page">
        {error && <div className="error">Could not reach the API: {error}. Is `draftai api` running?</div>}
        {!error && (!league || !board) && (
          <p className="muted">
            <span className="spinner" /> Loading the board…
          </p>
        )}
        {league && board && !draft && <Setup league={league} onStart={open} />}
        {league && board && draft && <DraftRoom league={league} board={board} initial={draft} onBoardChanged={reloadBoard} />}
      </main>
    </>
  );
}
