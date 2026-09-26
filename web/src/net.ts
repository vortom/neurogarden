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

export function serverUrl(): string {
  const params = new URLSearchParams(location.search);
  const explicit = params.get("server");
  if (explicit) return explicit;
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const path = import.meta.env.DEV ? "/ws" : "/";
  return `${scheme}://${location.host}${path}`;
}

export function token(): string {
  return new URLSearchParams(location.search).get("token") ?? "dev";
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
