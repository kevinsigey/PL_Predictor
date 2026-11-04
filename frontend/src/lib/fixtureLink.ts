// A fixture's detail lives in the URL hash: `#fixture/<event_id>`. A path such
// as `/fixture/<id>` would need the server to answer every unknown path with
// index.html, and this one does not (api/main.py mounts StaticFiles with
// html=True and no fallback), so a hash is the link that works when shared.

const PREFIX = "fixture/";

export function fixtureHash(eventId: string): string {
  return `#${PREFIX}${encodeURIComponent(eventId)}`;
}

/** The event id in a `#fixture/<id>` hash, or null for any other hash. */
export function fixtureIdFromHash(hash: string): string | null {
  const body = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!body.startsWith(PREFIX)) return null;
  try {
    const id = decodeURIComponent(body.slice(PREFIX.length));
    return id || null;
  } catch {
    return null;
  }
}
