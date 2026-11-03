import { useId, useState } from "react";

interface Props {
  text: string;
  label?: string;
}

// A tap-to-expand explanation, written out under the number it explains.
// Replaces the hover-only "?" tooltip on the fixture detail, which a touch
// screen cannot reach. A real button with aria-expanded, so keyboard and
// screen-reader users get the same state a tap gives.
export function GlossaryDisclosure({ text, label = "What does this mean?" }: Props) {
  const [open, setOpen] = useState(false);
  const panelId = useId();

  return (
    <div className="text-sm">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
        className="rounded-pr py-1 text-left font-semibold text-pr-text-dim underline decoration-pr-rule underline-offset-4 transition hover:text-pr-text focus-visible:outline-2 focus-visible:outline-pr-accent"
      >
        {label}
      </button>
      {open && (
        <p id={panelId} className="mt-1 rounded-pr bg-pr-panel px-3 py-2 leading-snug text-pr-text-dim">
          {text}
        </p>
      )}
    </div>
  );
}
