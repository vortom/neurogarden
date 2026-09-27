import { describe, expect, it } from "vitest";
import { PlaySession, closeNotice } from "../src/play";
import type { ServerMessage } from "../src/types";

const WELCOME = {
  v: 1,
  type: "welcome",
  payload: {
    protocol: 1,
    owner: "alice",
    role: "agent",
    world: {
      name: "tiny",
      width: 3,
      height: 2,
      tps: 5,
      day_length: 1200,
      dawn_end: 100,
      dusk_start: 700,
      night_start: 800,
      rules_version: 2,
    },
    catalog: { bodies: {}, constants: {} },
    motd: "",
  },
} satisfies ServerMessage;

function observation(events: Array<{ type: string }> = [], tick = 9): ServerMessage {
  return {
    v: 1,
    type: "observation",
    payload: {
      tick,
      deadline_ms: 200,
      channels: { smell: [], vision: [], touch: [], body: [], env: [] },
      events,
      missed: 0,
    },
  };
}

describe("the play session", () => {
  it("joins on the welcome and acts on every observation", () => {
    const session = new PlaySession();
    const id = session.start();
    expect(session.playing).toBe(true);
    expect(session.onMessage(id, WELCOME)).toEqual({
      stale: false,
      join: true,
      actFor: null,
      apply: true,
    });
    expect(session.onMessage(id, observation())).toMatchObject({ join: false, actFor: 9 });
  });

  it("sends no action for the observation of the tick the fly died", () => {
    const session = new PlaySession();
    const id = session.start();
    const plan = session.onMessage(id, observation([{ type: "damaged" }, { type: "died" }]));
    expect(plan).toMatchObject({ stale: false, actFor: null, apply: true });
  });

  it("ignores everything a superseded socket still says or does", () => {
    const session = new PlaySession();
    const old = session.start();
    const fresh = session.start();
    expect(old).not.toBe(fresh);
    expect(session.onMessage(old, WELCOME)).toMatchObject({ stale: true, join: false, apply: false });
    expect(session.onClose(old, 1006, "")).toEqual({ stale: true, notice: null });
    expect(session.onMessage(fresh, WELCOME).stale).toBe(false);
  });

  it("says nothing about a close we asked for ourselves", () => {
    const session = new PlaySession();
    const id = session.start();
    session.stop(); // the human pressed Leave; the 1000 close is on its way
    expect(session.playing).toBe(false);
    expect(session.onClose(id, 1000, "bye")).toEqual({ stale: true, notice: null });
  });

  it("explains a close nobody asked for, and names the tab that stole the fly", () => {
    const session = new PlaySession();
    const id = session.start();
    expect(session.onClose(id, 4004, "superseded")).toEqual({
      stale: false,
      notice: "play connection closed: another tab took over your fly",
    });
    expect(session.playing).toBe(false);
    expect(session.onClose(id, 1006, "")).toEqual({ stale: true, notice: null }); // only once
    expect(closeNotice(1001, "server shutting down")).toMatch(/server shutting down/);
    expect(closeNotice(1006, "")).toMatch(/code 1006/);
  });
});
