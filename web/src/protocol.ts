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
]);

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

export function say(text: string): ClientMessage {
  return { v: 1, type: "say", payload: { text } };
}

export function leave(): ClientMessage {
  return { v: 1, type: "leave", payload: {} };
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
