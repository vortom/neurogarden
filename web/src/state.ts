// The view model: everything the page knows, and the pure functions that turn it into pixels.
import type {
  AgentView,
  Died,
  Frame,
  Joined,
  Observation,
  ServerMessage,
  Welcome,
  WorldMap,
} from "./types";

export const LOG_LINES = 8;

export interface Garden {
  welcome: Welcome | null;
  world: WorldMap | null;
  frame: Frame | null;
  chronicle: string[];
  /** The human's own fly, when playing. */
  me: { owner: string; joined: Joined | null; observation: Observation | null; died: Died | null };
  notice: string | null;
}

export function emptyGarden(): Garden {
  return {
    welcome: null,
    world: null,
    frame: null,
    chronicle: [],
    me: { owner: "", joined: null, observation: null, died: null },
    notice: null,
  };
}

/** Apply a message from the spectator socket. Returns true if something visible changed. */
export function applySpectatorMessage(garden: Garden, message: ServerMessage): boolean {
  switch (message.type) {
    case "welcome":
      garden.welcome = message.payload;
      return true;
    case "world":
      garden.world = message.payload;
      return true;
    case "frame":
      garden.frame = message.payload;
      return true;
    case "chronicle":
      garden.chronicle.push(message.payload.text);
      if (garden.chronicle.length > LOG_LINES) garden.chronicle.splice(0, garden.chronicle.length - LOG_LINES);
      return true;
    case "error":
      garden.notice = `${message.payload.code}: ${message.payload.message}`;
      return true;
    default:
      return false;
  }
}

/** Apply a message from the agent socket (the human's fly). */
export function applyAgentMessage(garden: Garden, message: ServerMessage): boolean {
  switch (message.type) {
    case "joined":
      garden.me.joined = message.payload;
      garden.me.died = null;
      garden.me.observation = null;
      return true;
    case "observation":
      garden.me.observation = message.payload;
      return true;
    case "died":
      garden.me.died = message.payload;
      garden.me.joined = null;
      return true;
    case "error":
      if (message.payload.fatal) garden.notice = `${message.payload.code}: ${message.payload.message}`;
      return true;
    default:
      return false;
  }
}

export type Draw =
  | { kind: "tile"; x: number; y: number; terrain: number }
  | { kind: "fruit"; x: number; y: number; bites: number }
  | { kind: "fly"; x: number; y: number; facing: number; agentId: number; mine: boolean; away: boolean }
  | { kind: "bubble"; x: number; y: number; mood: string }
  | { kind: "say"; x: number; y: number; text: string };

/** What to draw, in order, for the latest frame. Pure, so it is testable without a canvas. */
export function drawList(garden: Garden): Draw[] {
  const out: Draw[] = [];
  const world = garden.world;
  if (world === null) return out;
  for (let y = 0; y < world.height; y++) {
    const row = world.terrain[y] ?? [];
    for (let x = 0; x < world.width; x++) out.push({ kind: "tile", x, y, terrain: row[x] ?? 0 });
  }
  const frame = garden.frame;
  if (frame === null) return out;
  for (const r of frame.resources) out.push({ kind: "fruit", x: r.x, y: r.y, bites: r.amount });
  const mine = garden.me.joined?.agent_id ?? -1;
  for (const a of livingFlies(frame)) {
    out.push({ kind: "fly", x: a.x, y: a.y, facing: a.facing, agentId: a.agent_id, mine: a.agent_id === mine, away: !a.connected });
  }
  for (const a of livingFlies(frame)) {
    if (a.mood in MOODS_WITH_BUBBLES) out.push({ kind: "bubble", x: a.x, y: a.y, mood: a.mood });
    if (a.say) out.push({ kind: "say", x: a.x, y: a.y, text: a.say });
  }
  return out;
}

const MOODS_WITH_BUBBLES: Record<string, true> = {
  content: true,
  hungry: true,
  thirsty: true,
  sleepy: true,
  desperate: true,
  dying: true,
};

export function livingFlies(frame: Frame): AgentView[] {
  return frame.agents.filter((a: AgentView) => a.alive);
}

export function myFly(garden: Garden): AgentView | null {
  const id = garden.me.joined?.agent_id;
  if (id === undefined || garden.frame === null) return null;
  return garden.frame.agents.find((a: AgentView) => a.agent_id === id && a.alive) ?? null;
}

/** 0 at full daylight, up to 0.55 at night: the alpha of the blue tint over the garden. */
export function nightAlpha(light: number): number {
  const dark = Math.max(0, Math.min(1, 1 - light / 1000));
  return Math.round(dark * 0.55 * 100) / 100;
}
