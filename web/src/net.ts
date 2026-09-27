// One WebSocket per role. Reconnecting is the page's job (reload), not this file's.
import { decode, encode, hello } from "./protocol";
import type { ClientMessage, ServerMessage } from "./types";

export interface Socket {
  send(message: ClientMessage): void;
  close(): void;
}

export interface SocketHandlers {
  onMessage(message: ServerMessage): void;
  onClose(code: number, reason: string): void;
}

/** Drop a query parameter from the address bar without reloading or adding a history entry. */
function forget(name: string): void {
  try {
    const url = new URL(location.href);
    if (!url.searchParams.has(name)) return;
    url.searchParams.delete(name);
    history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
  } catch {
    // an embedding without a usable History API: the parameter simply stays visible
  }
}

export function serverUrl(): string {
  // `?server=` points the page at another world; a dev-only escape hatch, never in the
  // bundle a stranger can link you to.
  const explicit = import.meta.env.DEV ? new URLSearchParams(location.search).get("server") : null;
  if (explicit) return explicit;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const path = import.meta.env.DEV ? "/ws" : "/";
  return `${scheme}://${location.host}${path}`;
}

let remembered: string | null = null;

/** The token from `?token=`, read once and wiped from the URL so it is not shared by copy. */
export function token(): string {
  if (remembered === null) {
    remembered = new URLSearchParams(location.search).get("token") ?? "dev";
    forget("token");
  }
  return remembered;
}

export function open(
  url: string,
  owner: string,
  role: "agent" | "spectator",
  handlers: SocketHandlers,
): Socket {
  const ws = new WebSocket(url);
  ws.addEventListener("open", () => ws.send(encode(hello(owner, token(), role))));
  ws.addEventListener("message", (event) => {
    const message = decode(String(event.data));
    if (message !== null) handlers.onMessage(message);
  });
  ws.addEventListener("close", (event) => handlers.onClose(event.code, event.reason));
  return {
    send(message) {
      if (ws.readyState === WebSocket.OPEN) ws.send(encode(message));
    },
    close() {
      ws.close(1000, "bye");
    },
  };
}
