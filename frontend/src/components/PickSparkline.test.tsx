import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PickSparkline } from "./PickSparkline";
import type { PickHistoryPoint } from "../types";

const point = (t: string, p_home: number): PickHistoryPoint => ({
  t,
  p_home,
  p_draw: 0.2,
  p_away: 1 - p_home - 0.2,
  m_home: null,
  m_draw: null,
  m_away: null,
});

describe("PickSparkline", () => {
  it("draws nothing for a single snapshot, since one point is not a trend", () => {
    const { container } = render(<PickSparkline points={[point("2026-10-01T10:00:00Z", 0.5)]} side="home_win" label="Arsenal" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("states the first and latest chance in words, not only in the line", () => {
    render(
      <PickSparkline
        points={[point("2026-10-01T10:00:00Z", 0.4), point("2026-10-05T10:00:00Z", 0.55), point("2026-10-09T10:00:00Z", 0.62)]}
        side="home_win"
        label="Arsenal"
      />,
    );
    expect(screen.getByRole("img", { name: "Arsenal: 40% at the first snapshot, 62% at the latest" })).toBeInTheDocument();
    expect(screen.getByText("Chance of Arsenal, over time")).toBeInTheDocument();
    expect(screen.getByText("3 snapshots", { exact: false })).toBeInTheDocument();
  });

  it("follows the pick's side, not always the home win", () => {
    render(
      <PickSparkline
        points={[point("2026-10-01T10:00:00Z", 0.4), point("2026-10-09T10:00:00Z", 0.3)]}
        side="draw"
        label="Draw"
      />,
    );
    expect(screen.getByRole("img", { name: "Draw: 20% at the first snapshot, 20% at the latest" })).toBeInTheDocument();
  });
});
