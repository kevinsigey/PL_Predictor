import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { CurrentGameweekResponse } from "../types";
import { FixtureModal } from "../components/FixtureModal";
import { CurrentGameweekSection } from "../components/CurrentGameweekSection";
import { PUBLIC_MODE } from "../lib/publicMode";
import { fixtureHash, fixtureIdFromHash } from "../lib/fixtureLink";
import { ErrorState, Skeleton } from "../predictor-ui";

export function FixturesPage() {
  const [gameweek, setGameweek] = useState<CurrentGameweekResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // An admin refresh failing is not the gameweek failing to load: keep the
  // two apart so each says what went wrong and retries the right thing.
  const [actionError, setActionError] = useState<string | null>(null);
  // The open fixture is the URL's `#fixture/<id>`, so a link opens it and Back
  // and Forward walk through it. The hash is the one source of truth.
  const [selected, setSelected] = useState<string | null>(() => fixtureIdFromHash(window.location.hash));
  useEffect(() => {
    const onHash = () => setSelected(fixtureIdFromHash(window.location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const openFixture = (eventId: string) => {
    window.location.hash = fixtureHash(eventId);
  };
  // Closing replaces the fixture's history entry instead of adding one, so Back
  // does not reopen a fixture the reader just closed.
  const closeFixture = () => {
    setSelected(null);
    window.history.replaceState(null, "", "#fixtures");
  };
  const [busy, setBusy] = useState<string | null>(null);
  // undefined = "current gameweek" (server decides); once the user
  // navigates, this pins to whichever gameweek they're browsing.
  const [viewGameweek, setViewGameweek] = useState<number | undefined>(undefined);

  const load = (gw?: number) => {
    setLoading(true);
    setError(null);
    api
      .currentGameweek(gw)
      .then(setGameweek)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => load(viewGameweek), [viewGameweek]);

  useEffect(() => {
    if (!gameweek?.fixtures.some((fixture) => fixture.finished && fixture.player_events_pending)) return;
    const timer = window.setTimeout(() => load(viewGameweek), 5000);
    return () => window.clearTimeout(timer);
  }, [gameweek, viewGameweek]);

  const navigate = (gw: number | undefined) => setViewGameweek(gw);

  const runAction = async (key: string, fn: () => Promise<unknown>) => {
    setBusy(key);
    setActionError(null);
    try {
      await fn();
      load(viewGameweek);
    } catch (e) {
      setActionError(`That refresh failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div>
      {/* Admin-only: public visitors never trigger backend work; the public
          snapshot refreshes odds and fixtures on a schedule. */}
      {!PUBLIC_MODE && (
        <div className="mb-6 flex flex-wrap items-center gap-3">
          <div className="ml-auto flex gap-2">
            <button
              disabled={busy !== null}
              onClick={() => runAction("fixtures", api.refreshFixtures)}
              className="rounded-lg border border-pl-border bg-pl-850/70 px-3 py-2 text-sm text-pl-text-dim transition hover:text-pl-text disabled:opacity-50"
            >
              {busy === "fixtures" ? "Refreshing…" : "Refresh fixtures"}
            </button>
            <button
              disabled={busy !== null}
              onClick={() => runAction("odds", api.refreshOddsPublic)}
              className="rounded-lg border border-pl-border bg-pl-850/70 px-3 py-2 text-sm text-pl-text-dim transition hover:text-pl-text disabled:opacity-50"
            >
              {busy === "odds" ? "Refreshing…" : "Refresh odds"}
            </button>
          </div>
        </div>
      )}

      {actionError && (
        <p role="alert" className="mb-4 rounded-pr border border-pr-loss/50 bg-pr-panel px-4 py-3 text-sm text-pr-text">
          {actionError}
        </p>
      )}

      {error && (
        <div className="mb-4">
          <ErrorState
            message={gameweek ? "We couldn't load that gameweek, so the one below is the last that loaded. Check your connection and try again." : "We couldn't load this gameweek. Check your connection and try again."}
            onRetry={() => load(viewGameweek)}
          />
        </div>
      )}

      {loading && !gameweek ? (
        <Skeleton label="Loading fixtures…" />
      ) : (
        gameweek && <CurrentGameweekSection data={gameweek} onSelect={openFixture} onNavigate={navigate} />
      )}

      {selected && <FixtureModal eventId={selected} onClose={closeFixture} explain={api.explain} loadContext={api.loadContext} />}
    </div>
  );
}
