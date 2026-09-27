// Which agent socket still counts, and what to do with what arrives on it.
// `main.ts` owns the sockets; the decisions live here, pure, so a test needs no WebSocket.
import type { ServerMessage } from "./types";

export interface MessagePlan {
  /** The message came from a socket that is no longer ours: drop it. */
  stale: boolean;
  /** Send a `join` on this socket. */
  join: boolean;
  /** Send an action for this observation's tick; null when none is due. */
  actFor: number | null;
  /** Feed the message to the view model. */
  apply: boolean;
}

export interface ClosePlan {
  stale: boolean;
  /** What to tell the human, or null when nothing should be said. */
  notice: string | null;
}

const IGNORED: MessagePlan = { stale: true, join: false, actFor: null, apply: false };

/** The observation of the tick a fly died is its last: acting on it only earns a `no_fly`. */
function isFinal(events: readonly { type: string }[]): boolean {
  return events.some((event) => event.type === "died");
}

export function closeNotice(code: number, reason: string): string {
  const why = code === 4004 ? "another tab took over your fly" : reason || `code ${code}`;
  return `play connection closed: ${why}`;
}

/**
 * The agent socket's lifecycle. A socket gets an id when it opens; once it is not the
 * current one it is stale, and everything it still says is ignored — including its close,
 * so a Leave (or a second Play superseding the first) never looks like the line dropping.
 */
export class PlaySession {
  private current = 0;
  private issued = 0;

  /** A new agent socket is opening: from now on only its messages count. */
  start(): number {
    this.current = ++this.issued;
    return this.current;
  }

  /** We asked the current socket to go away: retire it before its close event arrives. */
  stop(): void {
    this.current = 0;
  }

  get playing(): boolean {
    return this.current !== 0;
  }

  onMessage(socketId: number, message: ServerMessage): MessagePlan {
    if (socketId !== this.current) return IGNORED;
    const plan: MessagePlan = { stale: false, join: false, actFor: null, apply: true };
    if (message.type === "welcome") plan.join = true;
    if (message.type === "observation" && !isFinal(message.payload.events)) {
      plan.actFor = message.payload.tick;
    }
    return plan;
  }

  onClose(socketId: number, code: number, reason = ""): ClosePlan {
    if (socketId !== this.current) return { stale: true, notice: null };
    this.current = 0;
    return { stale: false, notice: closeNotice(code, reason) };
  }
}
