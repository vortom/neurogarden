// Ghost mode: `?ghost=owner/lineage[&speed=N]` opens the page on an archived life instead of
// the living garden. Pure helpers, so the URL grammar is testable without a window.
import { REPLAY_SPEED, REPLAY_SPEED_MAX, REPLAY_SPEED_MIN } from "./protocol";

export interface GhostRequest {
  owner: string;
  lineage: number;
  speed: number;
}

const OWNER_PATTERN = /^[A-Za-z0-9_.-]{1,64}$/;

/** The ghost a query string asks for, or null when it asks for the living garden. */
export function parseGhost(search: string): GhostRequest | null {
  const params = new URLSearchParams(search);
  const who = params.get("ghost");
  if (who === null) return null;
  const slash = who.lastIndexOf("/");
  if (slash <= 0) return null;
  const owner = who.slice(0, slash);
  const lineage = Number(who.slice(slash + 1));
  if (!OWNER_PATTERN.test(owner) || !Number.isInteger(lineage) || lineage < 1) return null;
  const asked = Number(params.get("speed") ?? REPLAY_SPEED);
  const speed = Number.isFinite(asked)
    ? Math.min(REPLAY_SPEED_MAX, Math.max(REPLAY_SPEED_MIN, asked))
    : REPLAY_SPEED;
  return { owner, lineage, speed };
}

/** The query string that opens a life as a ghost, ready to append to the page's path. */
export function ghostQuery(owner: string, lineage: number, speed?: number): string {
  const params = new URLSearchParams({ ghost: `${owner}/${lineage}` });
  if (speed !== undefined && speed !== REPLAY_SPEED) params.set("speed", String(speed));
  return `?${params.toString()}`;
}

/** One line for the banner over a ghost's garden. */
export function ghostBanner(
  ghost: { owner: string; name: string; lineage: number; born_tick: number; died_tick: number | null; speed: number; done: boolean },
  tick: number | null,
): string {
  const who = `${ghost.owner}'s ${ghost.name} #${ghost.lineage}`;
  if (ghost.done) {
    const end = ghost.died_tick === null ? "has caught up with the living" : "has faded";
    return `👻 ${who} ${end}`;
  }
  const of = ghost.died_tick === null ? "" : ` of ${ghost.died_tick}`;
  const at = tick === null ? "" : ` · tick ${tick}${of}`;
  return `👻 ${who} · ${ghost.speed}×${at}`;
}
