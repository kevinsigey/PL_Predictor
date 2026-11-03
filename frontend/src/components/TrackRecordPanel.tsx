// The PL track record, rebuilt on the shared Track Record kit.
//
// PL's counting rule already matches PRODUCT.md: the headline is every counted
// pick, and `pre_kickoff` is the subset made before kickoff with its own n.
// This page now expresses that through the shared kit (`RecordHeadline`,
// `WeekStrip`) instead of home-made StatCards and a recharts trend chart.
// PL's own blocks (current gameweek card, biggest-upsets table, per-gameweek
// grids, fixture modal) are plugged in as they were.
//
// The adapter lives in `lib/recordModel.ts`; the backend contract is unchanged.

import { useEffect, useState } from "react";
import type { TrackRecordByMarket, TrackRecordResponse, ValueBetSummary } from "../types";
import { InfoTooltip } from "./InfoTooltip";
import { MissesTable } from "./MissesTable";
import { GameweekResultsGrid } from "./GameweekResultsGrid";
import { FixtureModal } from "./FixtureModal";
import { api } from "../api/client";
import { GLOSSARY } from "../lib/glossary";
import {
  CalibrationBlock,
  HowCounted,
  RecordBlock,
  RecordHeadline,
  StatTile,
  WeekStrip,
  momentFor,
} from "../predictor-ui";
import { toRecordModel } from "../lib/recordModel";

const n0 = (x: number) => x.toLocaleString("en-US");

function shortDate(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d.toLocaleDateString(undefined, { dateStyle: "medium" });
}

/**
 * "Against the odds": the value-bet ledger's own record — every bet the app
 * flagged as value, recorded at the bookmaker price shown when it was
 * flagged and graded when the match finished. Strike rate and average odds
 * cover settled bets only; the return is the ledger's staking replay
 * (tenth-Kelly, the offline backtest's rule) and is labelled a simulation,
 * because that is what it is. An empty ledger says so rather than vanishing.
 */
function AgainstTheOdds({ bets }: { bets: ValueBetSummary }) {
  const since = shortDate(bets.first_flagged_at);
  return (
    <RecordBlock
      id="tr-odds"
      title="Against the odds"
      caption="Every bet the app flagged as value against the bookmaker's price, recorded at that price and graded when the match finished."
    >
      {bets.n_flagged === 0 ? (
        <p className="text-sm text-pl-text-dim" data-testid="odds-empty">
          No value bets have been flagged yet. When the model's price beats the bookmaker's by enough to flag, the
          bet is recorded here at the price shown and graded when the match finishes.
        </p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatTile
              label="Value bets recorded"
              value={n0(bets.n_flagged)}
              sub={bets.n_pending > 0 ? `${n0(bets.n_pending)} still to settle` : "All settled"}
            />
            <StatTile
              label="Strike rate"
              value={bets.win_rate != null ? `${bets.win_rate.toFixed(0)}%` : "—"}
              sub={
                bets.n_resolved > 0
                  ? `${n0(bets.wins)} won · ${n0(bets.losses)} lost of ${n0(bets.n_resolved)} settled`
                  : "Nothing settled yet"
              }
            />
            <StatTile
              label="Average odds"
              value={bets.average_odds != null ? bets.average_odds.toFixed(2) : "—"}
              sub="Settled bets, at the flagged price"
            />
            <StatTile
              label="Simulated return"
              value={bets.roi_pct != null ? `${bets.roi_pct >= 0 ? "+" : ""}${bets.roi_pct.toFixed(1)}%` : "—"}
              sub={
                bets.roi_pct != null
                  ? `Tenth-Kelly staking on ${n0(bets.n_staked)} settled bets`
                  : "No settled bet has been staked yet"
              }
            />
          </div>
          <p className="mt-3 max-w-prose text-xs text-pl-text-faint">
            {since ? `Flagged since ${since}. ` : ""}The return replays the settled bets at tenth-Kelly stakes on a
            100-unit bankroll — the same staking rule as the offline backtest. It is a simulation of the flags, not a
            betting record.
          </p>
        </>
      )}
    </RecordBlock>
  );
}

function StatCard({ label, value, info }: { label: string; value: string; info?: string }) {
  return (
    <div className="clip-corner rounded-xl border border-pl-border bg-pl-850/70 p-4">
      <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-pl-text-faint">
        {label}
        {info && <InfoTooltip text={info} align="left" />}
      </div>
      <div className="mt-1 font-display text-3xl font-semibold tracking-wide text-pl-text">{value}</div>
    </div>
  );
}

export function TrackRecordPanel({ data }: { data: TrackRecordResponse }) {
  const { summary, biggest_upsets, gameweeks, value_bets } = data;
  const model = toRecordModel(data);
  const moment = momentFor("pl");

  const [selected, setSelected] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  // The record's freshness stamp, from the snapshot's own metadata. It is a
  // courtesy line in HowCounted: if the read fails, the block simply goes
  // unstamped rather than failing with it.
  const [asOf, setAsOf] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    api
      .snapshotMeta()
      .then((meta) => {
        if (live) setAsOf(meta.generated_at);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  const load = async (force = false) => {
    setRefreshing(true);
    try {
      await api.trackRecord(force);
      // DataHubPage owns error/retry for this panel; the refresh button is a
      // reader affordance to re-fetch from the page header.
    } finally {
      setRefreshing(false);
    }
  };

  // Defensive: an older public_snapshot.json built before a field existed can
  // still be live for a few minutes after a deploy (the frontend ships
  // instantly; the data it reads is regenerated on its own schedule).
  const byMarket: TrackRecordByMarket = summary.by_market ?? ({} as TrackRecordByMarket);
  const nRebuilt = summary.n_rebuilt_fixtures ?? 0;
  const preKickoff = summary.pre_kickoff ?? null;

  // Only a record with NOTHING at all falls back to the short message. A
  // record whose every pick was made after kickoff still has graded picks and
  // a list worth showing, so it renders in full — the headline is the record,
  // and the pre-kickoff figure beside it is where the absence is read.
  if (summary.n_resolved_fixtures === 0 && nRebuilt === 0) {
    return (
      <section aria-labelledby="tr-empty-h">
        <h3 id="tr-empty-h" className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
          Live track record
          <InfoTooltip text={GLOSSARY.trackRecordScore} align="left" />
        </h3>
        <p className="text-xs text-pl-text-faint">
          No scored predictions yet — this builds up automatically as fixtures are predicted and results come in.
        </p>
      </section>
    );
  }

  return (
    <section aria-labelledby="tr-h" className="flex flex-col gap-4">
      <h2 id="tr-h" className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-pl-text-dim">
        Track Record
        <button
          type="button"
          onClick={() => { void load(true); }}
          disabled={refreshing}
          className="ml-auto rounded-lg border border-pl-border bg-pl-850/50 px-3 py-2 text-xs font-semibold text-pl-text-dim transition hover:text-pl-text disabled:opacity-50"
          aria-label={refreshing ? "Refreshing track record…" : "Refresh track record"}
        >
          {refreshing ? "Refreshing…" : "Refresh live data"}
        </button>
      </h2>

      {/* Calibration chart as hero: full width under the header, above the record. */}
      <CalibrationBlock series={model.calibration ?? []} markets={model.markets} />

      <RecordBlock id="tr-record" title="Record" caption="Every counted pick, whenever it was made; the figure beside each market is the pre-kickoff subset.">
        {model.markets.length > 0 ? (
          <>
            <RecordHeadline markets={model.markets} mode={model.headline} lead={model.lead} moment={moment} />
            <p className="max-w-prose text-sm text-pl-text-dim" data-testid="rebuilt-note">
              {nRebuilt > 0
                ? `${n0(summary.n_resolved_fixtures)} counted picks in this record, of which ${n0(
                    nRebuilt,
                  )} were made at or after their own kickoff and are counted like any other. The figure beside each
                    market is the pre-kickoff subset.`
                : preKickoff?.n_resolved_fixtures
                  ? `${n0(preKickoff.n_resolved_fixtures)} of ${n0(summary.n_resolved_fixtures)} counted picks were made before their own kickoff.`
                  : `Every pick in this record is counted here. The figure beside each market is the pre-kickoff subset.`}
            </p>
          </>
        ) : (
          <p className="text-sm text-pl-text-faint">No scored predictions yet — check back once fixtures are final.</p>
        )}
      </RecordBlock>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          label={summary.current_gameweek ? `Gameweek ${summary.current_gameweek}` : "This gameweek"}
          value={
            summary.pct_correct_current_gameweek === null
              ? "—"
              : `${((summary.pct_correct_current_gameweek ?? 0) * 100).toFixed(0)}% (${Math.round(
                  summary.pct_correct_current_gameweek * summary.n_fixtures_current_gameweek,
                )}/${summary.n_fixtures_current_gameweek})`
          }
          info={GLOSSARY.trackRecordScore}
        />
      </div>

      {value_bets && <AgainstTheOdds bets={value_bets} />}

      <RecordBlock id="tr-week" title="By gameweek">
        <WeekStrip weeks={model.weeks ?? []} />
      </RecordBlock>

      <RecordBlock id="tr-reliability" title="Prediction reliability">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {(["match_result", "over_under_2_5", "exact_score", "btts"] as const).map((key) => {
            const stat = byMarket[key];
            const label = key === "match_result" ? "Match result" : key === "over_under_2_5" ? "Goals O/U 2.5" : key === "exact_score" ? "Exact score" : "BTTS";
            return (
              <StatTile
                key={key}
                label={label}
                value={stat?.n_resolved === 0 ? "—" : `${((stat?.pct_correct ?? 0) * 100).toFixed(0)}%`}
                sub={`${n0(stat?.n_resolved ?? 0)} fixture${(stat?.n_resolved ?? 0) === 1 ? "" : "s"} graded`}
              />
            );
          })}
        </div>
      </RecordBlock>

      {biggest_upsets.length > 0 && (
        <RecordBlock id="tr-upsets" title="Biggest upset">
          <MissesTable misses={biggest_upsets.slice(0, 3)} />
        </RecordBlock>
      )}

      <div className="flex flex-col gap-5">
        {gameweeks.map((group) => (
          <div key={group.gameweek ?? "none"} aria-labelledby={`tr-gw-${group.gameweek ?? "none"}-h`}>
            <h3 id={`tr-gw-${group.gameweek ?? "none"}-h`} className="mb-1 flex items-center gap-2 text-sm font-semibold text-pl-text-dim">
              {group.gameweek ? `Gameweek ${group.gameweek}` : "No gameweek data"}
              <span className="text-xs font-normal text-pl-text-dim">
                {group.pct_correct === null
                  ? "No pre-kickoff picks"
                  : `${((group.pct_correct ?? 0) * 100).toFixed(0)}% correct (${Math.round(group.pct_correct * group.n_fixtures)}/${group.n_fixtures})`}
              </span>
            </h3>
            {group.pct_correct !== null && group.pct_correct_by_market && (
              <p className="mb-2 text-xs text-pl-text-faint">
                Exact score {((group.pct_correct_by_market.exact_score ?? 0) * 100).toFixed(0)}% &middot; Goals O/U 2.5{" "}
                {((group.pct_correct_by_market.over_under_2_5 ?? 0) * 100).toFixed(0)}% &middot; BTTS {((group.pct_correct_by_market.btts ?? 0) * 100).toFixed(0)}%
              </p>
            )}
            <GameweekResultsGrid results={group.fixtures} onSelect={setSelected} />
          </div>
        ))}
      </div>

      {selected && <FixtureModal eventId={selected} onClose={() => setSelected(null)} explain={api.explain} loadContext={api.loadContext} />}

      <HowCounted moment={moment} headline={model.headline} asOf={asOf} />
    </section>
  );
}
