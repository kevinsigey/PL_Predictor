import { describe, expect, it } from "vitest";
import { fixtureHash, fixtureIdFromHash } from "./fixtureLink";

describe("fixture deep links", () => {
  it("round-trips an event id through the hash", () => {
    const id = "backfill-Arsenal-Coventry-2026-08-21";
    expect(fixtureIdFromHash(fixtureHash(id))).toBe(id);
  });

  it("ignores hashes that are not a fixture", () => {
    expect(fixtureIdFromHash("#fixtures")).toBeNull();
    expect(fixtureIdFromHash("#teams")).toBeNull();
    expect(fixtureIdFromHash("")).toBeNull();
    expect(fixtureIdFromHash("#fixture/")).toBeNull();
  });

  it("treats a malformed escape as no fixture rather than throwing", () => {
    expect(fixtureIdFromHash("#fixture/%E0%A4%A")).toBeNull();
  });
});
