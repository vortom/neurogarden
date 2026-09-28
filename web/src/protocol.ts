// Envelope helpers: encode outgoing frames, decode incoming ones, ignore what we do not know.
import { PROTOCOL_VERSION, type ClientMessage, type ServerMessage } from "./types";

export const SERVER_TYPES = new Set([
  "welcome",
  "joined",
  "observation",
  "died",
  "world",
  "frame",
  "chronicle",
  "error",
  "replay",
]);

/** What the server accepts as a replay speed (multiples of the world's own pace). */
export const REPLAY_SPEED = 4;
export const REPLAY_SPEED_MIN = 0.25;
export const REPLAY_SPEED_MAX = 64;

export function encode(message: ClientMessage): string {
  return JSON.stringify(message);
}

export function hello(owner: string, token: string, role: "agent" | "spectator"): ClientMessage {
  return {
    v: 1,
    type: "hello",
    payload: { protocol: PROTOCOL_VERSION, token, owner, role, client: "neurogarden-web/0.1" },
  };
}

export function join(): ClientMessage {
  return { v: 1, type: "join", payload: { body: "fly" } };
}

export function action(tick: number, id: number): ClientMessage {
  return { v: 1, type: "action", payload: { tick, action: id } };
}

export const SAY_MAX = 40;
const CONTROL_CHARS = /[\x00-\x1f\x7f]/g;

/**
 * What the server will accept in a speech bubble: it reads a control character as a
 * malformed frame and closes the connection, and rejects anything over 40 characters.
 */
export function sayText(text: string): string {
  return text.replace(CONTROL_CHARS, "").trim().slice(0, SAY_MAX);
}

export function say(text: string): ClientMessage {
  return { v: 1, type: "say", payload: { text: sayText(text) } };
}

export function leave(): ClientMessage {
  return { v: 1, type: "leave", payload: {} };
}

export function replay(owner: string, lineage: number, speed = REPLAY_SPEED): ClientMessage {
  return { v: 1, type: "replay", payload: { owner, lineage, speed } };
}

/** A server message, or null for anything this client does not understand. */
export function decode(text: string): ServerMessage | null {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof raw !== "object" || raw === null) return null;
  const envelope = raw as { v?: unknown; type?: unknown; payload?: unknown };
  if (envelope.v !== PROTOCOL_VERSION) return null;
  if (typeof envelope.type !== "string" || !SERVER_TYPES.has(envelope.type)) return null;
  if (typeof envelope.payload !== "object" || envelope.payload === null) return null;
  return envelope as ServerMessage;
}
