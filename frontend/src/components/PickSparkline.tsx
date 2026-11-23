import type { PickHistoryPoint } from "../types";
import type { PickSide } from "../lib/pick";
import { pct } from "../predictor-ui";

const PROB_KEY: Record<PickSide, "p_home" | "p_draw" | "p_away"> = {
  home_win: "p_home",
  draw: "p_draw",
  away_win: "p_away",
};

const W = 300;
const H = 72;
const PAD = 6;

interface Props {
  points: PickHistoryPoint[];
  side: PickSide;
  // The pick's name in the same words the bar uses ("Arsenal", "Draw").
  label: string;
}

// The pick's probability across the snapshots that recorded it. X is real time,
// so a long quiet stretch draws as a long flat run. Y is padded 5 points either
// side of the data and never past 0-100%, and the caption states the first and
// latest values, so the line is never the only source of the number.
export function PickSparkline({ points, side, label }: Props) {
  if (points.length < 2) return null;
  const key = PROB_KEY[side];
  const values = points.map((point) => point[key]);
  const times = points.map((point) => Date.parse(point.t));
  const t0 = times[0];
  const span = times[times.length - 1] - t0 || 1;
  const lo = Math.max(0, Math.min(...values) - 0.05);
  const hi = Math.min(1, Math.max(...values) + 0.05);
  const range = hi - lo || 1;
  const x = (i: number) => PAD + ((times[i] - t0) / span) * (W - 2 * PAD);
  const y = (value: number) => PAD + (1 - (value - lo) / range) * (H - 2 * PAD);
  const line = values.map((value, i) => `${x(i).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const first = values[0];
  const latest = values[values.length - 1];
  const summary = `${label}: ${pct(first)} at the first snapshot, ${pct(latest)} at the latest`;

  return (
    <figure className="flex flex-col gap-1.5 rounded-pr bg-pr-panel px-3 py-2.5">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3 text-sm">
        <span className="font-semibold text-pr-text">Chance of {label}, over time</span>
        <span className="text-pr-text-dim">
          {pct(first)} → <span className="font-semibold text-pr-text">{pct(latest)}</span>
          <span className="text-pr-text-faint"> · {points.length} snapshots</span>
        </span>
      </figcaption>
      <svg role="img" aria-label={summary} viewBox={`0 0 ${W} ${H}`} className="h-16 w-full text-pr-accent">
        <polyline points={line} fill="none" stroke="currentColor" strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        <circle cx={x(points.length - 1)} cy={y(latest)} r={3} fill="currentColor" />
      </svg>
    </figure>
  );
}
