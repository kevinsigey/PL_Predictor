// The against-the-odds block (value-bet ledger summary on the hub track
// record), the percentage-forward headline, and the record's as-of stamp.
import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { TrackRecordPanel } from "./TrackRecordPanel";
import { HowCounted } from "../predictor-ui";
import type { TrackRecordResponse, ValueBetSummary } from "../types";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      snapshotMeta: () =>
        Promise.resolve({ generated_at: "2026-11-01T22:15:00Z", source: "public_snapshot", model_fingerprint: "abc" }),
    },
  };
});

const base: TrackRecordResponse = {
  summary: {
    n_resolved_fixtures: 3,
    n_rebuilt_fixtures: 0,
    pct_correct_overall: 2 / 3,
    current_gameweek: 5,
    pct_correct_current_gameweek: 2 / 3,
    n_fixtures_current_gameweek: 3,
    gameweek_trend: [],
    by_market: {
      exact_score: { pct_correct: 0, n_resolved: 3 },
      match_result: { pct_correct: 2 / 3, n_resolved: 3 },
      over_under_2_5: { pct_correct: 1 / 3, n_resolved: 3 },
      btts: { pct_correct: 1, n_resolved: 3 },
    },
    per_pick: [],
  },
  biggest_upsets: [],
  gameweeks: [],
};

const settled: ValueBetSummary = {
  n_flagged: 3,
  n_resolved: 2,
  n_pending: 1,
  wins: 1,
  losses: 1,
  win_rate: 50.0,
  average_odds: 2.4,
  roi_pct: 12.5,
  profit_units: 12.5,
  n_staked: 2,
  staking: "kelly",
  first_flagged_at: "2026-10-01T15:00:00",
  last_flagged_at: "2026-12-01T15:00:00",
};

const empty: ValueBetSummary = {
  ...settled,
  n_flagged: 0,
  n_resolved: 0,
  n_pending: 0,
  wins: 0,
  losses: 0,
  win_rate: null,
  average_odds: null,
  roi_pct: null,
  profit_units: null,
  n_staked: 0,
  first_flagged_at: null,
  last_flagged_at: null,
};

describe("TrackRecordPanel against the odds", () => {
  it("renders the ledger's graded record with its population labelled", () => {
    const { container } = render(<TrackRecordPanel data={{ ...base, value_bets: settled }} />);
    const block = within(container.querySelector("#tr-odds") as HTMLElement);

    expect(block.getByText("Against the odds")).toBeInTheDocument();
    expect(block.getByText("Value bets recorded")).toBeInTheDocument();
    expect(block.getByText("Strike rate")).toBeInTheDocument();
    expect(block.getByText("50%")).toBeInTheDocument();
    expect(block.getByText("1 won · 1 lost of 2 settled")).toBeInTheDocument();
    expect(block.getByText("Average odds")).toBeInTheDocument();
    expect(block.getByText("2.40")).toBeInTheDocument();
    expect(block.getByText("Simulated return")).toBeInTheDocument();
    expect(block.getByText("+12.5%")).toBeInTheDocument();
    expect(block.getByText(/Flagged since/)).toBeInTheDocument();
  });

  it("says so plainly when the ledger is empty, instead of disappearing", () => {
    render(<TrackRecordPanel data={{ ...base, value_bets: empty }} />);

    expect(screen.getByText("Against the odds")).toBeInTheDocument();
    expect(screen.getByTestId("odds-empty")).toHaveTextContent("No value bets have been flagged yet");
  });

  it("renders no block at all when the payload predates the summary", () => {
    render(<TrackRecordPanel data={base} />);

    expect(screen.queryByText("Against the odds")).not.toBeInTheDocument();
  });

  it("leads the headline tally with the percentage, count beside it", () => {
    const { container } = render(<TrackRecordPanel data={base} />);
    const tally = within(container).getByTestId("tally-match_result");

    expect(tally).toHaveTextContent("67%");
    expect(tally).toHaveTextContent("2 of 3");
    expect(tally.textContent!.indexOf("67%")).toBeLessThan(tally.textContent!.indexOf("2 of 3"));
  });

  it("stamps the record with the snapshot's generated_at", async () => {
    render(<TrackRecordPanel data={base} />);

    expect(await screen.findByText(/Record as of/)).toBeInTheDocument();
  });
});

describe("HowCounted as-of stamp", () => {
  it("renders the stamp when given a parseable time", () => {
    render(<HowCounted asOf="2026-11-01T22:15:00Z" />);
    expect(screen.getByText(/Record as of/)).toBeInTheDocument();
  });

  it("renders nothing extra without one, or with a broken one", () => {
    const { rerender } = render(<HowCounted />);
    expect(screen.queryByText(/Record as of/)).not.toBeInTheDocument();
    rerender(<HowCounted asOf="not a date" />);
    expect(screen.queryByText(/Record as of/)).not.toBeInTheDocument();
  });
});
