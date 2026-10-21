import { useCallback, useEffect, useMemo, useState } from "react";
import { api, subscribe, type BoardPlayer, type DraftState, type League, type Recommendation } from "../api";
import Banner from "./Banner";
import LeagueProjection from "./LeagueProjection";
import Overrides from "./Overrides";
import PlayerPanel from "./PlayerPanel";
import PlayerTable from "./PlayerTable";
import Recommendations from "./Recommendations";
import { DraftLog, Roster, RoomOdds, Watchlist } from "./SidePanels";

const WATCH_KEY = "lockin.watchlist";

function loadWatch(): string[] {
  try {
    return JSON.parse(localStorage.getItem(WATCH_KEY) ?? "[]");
  } catch {
    return [];
  }
}

interface Props {
  league: League;
  board: BoardPlayer[];
  initial: DraftState;
  onBoardChanged: () => void;
}

export default function DraftRoom({ league, board, initial, onBoardChanged }: Props) {
  const [state, setState] = useState<DraftState>(initial);
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [useRollout, setUseRollout] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [watch, setWatch] = useState<string[]>(loadWatch);

  const byId = useMemo(() => new Map(board.map((p) => [p.sleeper_id, p])), [board]);
  const drafted = useMemo(() => new Set(state.drafted), [state.drafted]);

  // The server pushes the state on every change (in order; an undo lowers the version), and
  // polls Sleeper for live drafts.
  useEffect(() => subscribe(state.id, (s) => setState((old) => (s.id === old.id ? s : old))), [state.id]);

  useEffect(() => {
    try {
      localStorage.setItem(WATCH_KEY, JSON.stringify(watch));
    } catch {
      /* private window */
    }
  }, [watch]);

  // Recommendation for this version of the state (rollouts take a second or two).
  useEffect(() => {
    let cancelled = false;
    if (state.complete) {
      setRec(null);
      return;
    }
    setThinking(true);
    api
      .recommendation(state.id, useRollout)
      .then((r) => !cancelled && setRec(r))
      .catch((e) => !cancelled && setError(String(e.message)))
      .finally(() => !cancelled && setThinking(false));
    return () => {
      cancelled = true;
    };
  }, [state.id, state.version, state.complete, useRollout]);

  const act = useCallback(async (fn: () => Promise<DraftState>) => {
    setBusy(true);
    setError(null);
    try {
      setState(await fn());
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }, []);

  const canPick = !state.complete && state.kind !== "live" && (state.kind === "manual" || state.my_turn);
  const pick = (id: string) => act(() => api.pick(state.id, id));
  const toggleWatch = (id: string) => setWatch((w) => (w.includes(id) ? w.filter((x) => x !== id) : [...w, id]));

  return (
    <div className="col">
      <Banner
        state={state}
        busy={busy}
        useRollout={useRollout}
        onRollout={setUseRollout}
        onUndo={() => act(() => api.undo(state.id))}
        onAdvance={() => act(() => api.advance(state.id))}
        onAutodraft={(rollout) => act(() => api.autodraft(state.id, rollout))}
        onSync={() => act(() => api.sync(state.id))}
      />
      {error && <div className="error">{error}</div>}
      {state.complete ? (
        <LeagueProjection state={state} />
      ) : (
        <div className="grid">
          <div className="col">
            <Recommendations
              rec={rec}
              thinking={thinking}
              useRollout={useRollout}
              byId={byId}
              canPick={canPick}
              onPick={pick}
              onSelect={setSelected}
            />
            <PlayerTable
              board={board}
              state={state}
              drafted={drafted}
              watch={watch}
              selected={selected}
              canPick={canPick}
              onSelect={setSelected}
              onPick={pick}
              onWatch={toggleWatch}
            />
          </div>
          <div className="col">
            {selected && byId.has(selected) && (
              <PlayerPanel
                id={selected}
                state={state}
                drafted={drafted.has(selected)}
                watched={watch.includes(selected)}
                canPick={canPick}
                onPick={pick}
                onWatch={toggleWatch}
                onClose={() => setSelected(null)}
              />
            )}
            <Roster state={state} league={league} />
            <Watchlist watch={watch} state={state} byId={byId} drafted={drafted} onSelect={setSelected} onWatch={toggleWatch} />
            {state.kind === "mock" && !state.my_turn && <RoomOdds state={state} />}
            <DraftLog state={state} />
            <Overrides board={board} onSaved={onBoardChanged} />
          </div>
        </div>
      )}
    </div>
  );
}
