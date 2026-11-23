type Trio = { home: number; draw: number; away: number };

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** The most likely 1X2 outcome of a trio, as a label and probability. */
function top(t: Trio, home: string, away: string): { label: string; prob: number } {
  const pick = [
    { label: home, prob: t.home },
    { label: "Draw", prob: t.draw },
    { label: away, prob: t.away },
  ].reduce((a, b) => (b.prob > a.prob ? b : a));
  return pick;
}

/**
 * The market's 1X2 probability is the headline; the model's number sits beside it, smaller.
 * `market` null/undefined = no live line: the model number stays as it was, with a note. Never defaulted.
 * `showModel` false keeps the model's figure off this line where another component already prints it.
 */
export function MarketHeadline({ home, away, market, model, showModel = true }: { home: string; away: string; market?: Trio | null; model: Trio; showModel?: boolean }) {
  const m = top(model, home, away);
  if (!market) {
    return (
      <div data-testid="market-headline" className="text-center text-xs text-pl-text-faint">
        {showModel && <>Model: {m.label} {pct(m.prob)} · </>}No market line
      </div>
    );
  }
  const k = top(market, home, away);
  return (
    <div data-testid="market-headline" className="flex flex-wrap items-baseline justify-center gap-x-4 gap-y-0.5 text-center">
      <span className="font-display text-xl font-semibold tracking-wide text-pl-text">
        <span className="mr-1.5 text-xs font-medium uppercase tracking-wide text-pl-text-faint">Market</span>
        {k.label} {pct(k.prob)}
      </span>
      {showModel && (
        <span className="text-xs text-pl-text-dim">
          Model {m.label} {pct(m.prob)}
        </span>
      )}
    </div>
  );
}
