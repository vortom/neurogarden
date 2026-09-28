// The view model: everything the page knows, and the pure functions that turn it into pixels.
import type {
  AgentView,
  Died,
  Frame,
  Joined,
  Observation,
  ReplayInfo,
  ServerMessage,
  Welcome,
  WorldMap,
} from "./types";
import { RESOURCE } from "./types";

export const LOG_LINES = 8;

export interface Me {
  owner: string;
  joined: Joined | null;
  observation: Observation | null;
  died: Died | null;
  /** A `join` is on the wire: the hatch button is spent until `joined` or an error lands. */
  hatching: boolean;
}

export interface Garden {
  welcome: Welcome | null;
  world: WorldMap | null;
  frame: Frame | null;
  chronicle: string[];
  /** The human's own fly, when playing. */
  me: Me;
  /** The archived life being watched again, when the page was opened on a ghost. */
  ghost: ReplayInfo | null;
  notice: string | null;
}

export function noFly(owner = ""): Me {
  return { owner, joined: null, observation: null, died: null, hatching: false };
}

export function emptyGarden(): Garden {
  return {
    welcome: null,
    world: null,
    frame: null,
    chronicle: [],
    me: noFly(),
    ghost: null,
    notice: null,
  };
}

/** The action names a fly's body offers, in the order that gives each one its id. */
export function flyActions(garden: Garden): readonly string[] {
  return garden.welcome?.catalog.bodies["fly"]?.actions ?? [];
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
    case "replay":
      garden.ghost = message.payload;
      if (!message.payload.done) garden.frame = null; // the ghost's first frame is still to come
      return true;
    case "error":
      garden.notice = `${message.payload.code}: ${message.payload.message}`;
      return true;
    default:
      return false;
  }
}

/** The fly the page is about: the human's own, or the ghost being watched. */
export function focusId(garden: Garden): number {
  const mine = garden.me.joined?.agent_id;
  if (mine !== undefined) return mine;
  const ghost = garden.ghost;
  if (ghost === null || garden.frame === null) return -1;
  const fly = garden.frame.agents.find((a: AgentView) => a.owner === ghost.owner && a.lineage === ghost.lineage);
  return fly?.agent_id ?? -1;
}

/** Apply a message from the agent socket (the human's fly). */
export function applyAgentMessage(garden: Garden, message: ServerMessage): boolean {
  switch (message.type) {
    case "welcome":
      // The spectator socket usually gets here first; whichever arrives first wins.
      garden.welcome ??= message.payload;
      return true;
    case "joined":
      garden.me.joined = message.payload;
      garden.me.died = null;
      garden.me.observation = null;
      garden.me.hatching = false;
      return true;
    case "observation":
      garden.me.observation = message.payload;
      // Without the catalog every action would be sent as 0 (idle) and the fly would
      // never move: say so rather than look broken.
      if (flyActions(garden).length === 0) garden.notice = "server sent no actions for a fly";
      return true;
    case "died":
      garden.me.died = message.payload;
      garden.me.joined = null;
      garden.me.hatching = false;
      return true;
    case "error":
      // Non-fatal errors (a refused join, a late action) are the ones worth explaining:
      // a fatal one closes the socket, which speaks for itself.
      garden.notice = `${message.payload.code}: ${message.payload.message}`;
      garden.me.hatching = false;
      return true;
    default:
      return false;
  }
}

export type Draw =
  | { kind: "tile"; x: number; y: number; terrain: number }
  | { kind: "resource"; x: number; y: number; resource: number; amount: number }
  | { kind: "fly"; x: number; y: number; facing: number; agentId: number; mine: boolean; away: boolean }
  | { kind: "bubble"; x: number; y: number; mood: string }
  | { kind: "say"; x: number; y: number; text: string };

/** The map, which only changes when a new world arrives. Split out so it can be baked once. */
export function terrainList(world: WorldMap | null): Draw[] {
  const out: Draw[] = [];
  if (world === null) return out;
  for (let y = 0; y < world.height; y++) {
    const row = world.terrain[y] ?? [];
    for (let x = 0; x < world.width; x++) out.push({ kind: "tile", x, y, terrain: row[x] ?? 0 });
  }
  return out;
}

/** Everything that moves, in order, for the latest frame. */
export function actorList(garden: Garden): Draw[] {
  const out: Draw[] = [];
  const frame = garden.frame;
  if (garden.world === null || frame === null) return out;
  for (const r of frame.resources) {
    // A kind this client has no sprite for is better left out than drawn as an apple.
    if (r.kind !== RESOURCE.fruit) continue;
    out.push({ kind: "resource", x: r.x, y: r.y, resource: r.kind, amount: r.amount });
  }
  const mine = focusId(garden);
  for (const a of livingFlies(frame)) {
    out.push({ kind: "fly", x: a.x, y: a.y, facing: a.facing, agentId: a.agent_id, mine: a.agent_id === mine, away: !a.connected });
  }
  for (const a of livingFlies(frame)) {
    if (a.mood in MOODS_WITH_BUBBLES) out.push({ kind: "bubble", x: a.x, y: a.y, mood: a.mood });
    if (a.say) out.push({ kind: "say", x: a.x, y: a.y, text: a.say });
  }
  return out;
}

/** What to draw, in order, for the latest frame. Pure, so it is testable without a canvas. */
export function drawList(garden: Garden): Draw[] {
  return [...terrainList(garden.world), ...actorList(garden)];
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
export function nightAlpha(light: number, lightMax = DEFAULT_CONSTANTS.lightMax): number {
  const dark = Math.max(0, Math.min(1, 1 - light / Math.max(1, lightMax)));
  return Math.round(dark * 0.55 * 100) / 100;
}

export interface Constants {
  needMax: number;
  lightMax: number;
  nightLight: number;
}

/** Today's engine defaults: what to assume until (or unless) the catalog says otherwise. */
export const DEFAULT_CONSTANTS: Constants = { needMax: 1000, lightMax: 1000, nightLight: 500 };

/** The scales a number in a frame is measured against, from `welcome.catalog.constants`. */
export function constants(garden: Garden): Constants {
  const c = garden.welcome?.catalog.constants;
  return {
    needMax: c?.["need_max"] ?? DEFAULT_CONSTANTS.needMax,
    lightMax: c?.["light_max"] ?? DEFAULT_CONSTANTS.lightMax,
    nightLight: c?.["night_light_threshold"] ?? DEFAULT_CONSTANTS.nightLight,
  };
}

export interface DayPhases {
  dayLength: number;
  dawnEnd: number;
  duskStart: number;
  nightStart: number;
}

/** The shape of a day under the engine's default config, in ticks since the day began. */
export const DEFAULT_PHASES: DayPhases = {
  dayLength: 1200,
  dawnEnd: 100,
  duskStart: 700,
  nightStart: 800,
};

/** When the light turns, from `welcome.world`; a server too old to say uses the defaults. */
export function dayPhases(garden: Garden): DayPhases {
  const w = garden.welcome?.world;
  return {
    dayLength: w?.day_length ?? DEFAULT_PHASES.dayLength,
    dawnEnd: w?.dawn_end ?? DEFAULT_PHASES.dawnEnd,
    duskStart: w?.dusk_start ?? DEFAULT_PHASES.duskStart,
    nightStart: w?.night_start ?? DEFAULT_PHASES.nightStart,
  };
}
