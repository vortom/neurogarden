import { describe, expect, it } from "vitest";
import { SAY_MAX, decode, encode, hello, say, sayText } from "../src/protocol";
import {
  DEFAULT_CONSTANTS,
  DEFAULT_PHASES,
  LOG_LINES,
  applyAgentMessage,
  applySpectatorMessage,
  constants,
  dayPhases,
  drawList,
  emptyGarden,
  flyActions,
  myFly,
  nightAlpha,
} from "../src/state";
import type { AgentView, Frame, ServerMessage } from "../src/types";
import { timeOfDay } from "../src/hud";

function fly(overrides: Partial<AgentView> = {}): AgentView {
  return {
    agent_id: 1,
    owner: "alice",
    lineage: 1,
    name: "Dusty Wing",
    x: 2,
    y: 1,
    facing: 2,
    satiety: 700,
    hydration: 700,
    energy: 800,
    health: 1000,
    age: 3,
    alive: true,
    connected: true,
    mood: "content",
    say: "",
    ...overrides,
  };
}

function frame(agents: AgentView[], extra: Partial<Frame> = {}): ServerMessage {
  return {
    v: 1,
    type: "frame",
    payload: {
      tick: 5,
      day: 1,
      light: 1000,
      resources: [{ x: 1, y: 1, kind: 1, amount: 4 }],
      agents,
      events: [],
      scores: [],
      ...extra,
    },
  };
}

const WORLD: ServerMessage = {
  v: 1,
  type: "world",
  payload: { map_name: "tiny", width: 3, height: 2, terrain: [[2, 1, 3], [5, 1, 4]] },
};

const CHANNEL = { shape: [1], dtype: "int16", low: 0, high: 1 };

function welcome(
  extraConstants: Record<string, number> = {},
  world: Record<string, number> = {},
  actions: string[] = ["idle", "move_n"],
): ServerMessage {
  return {
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
        day_length: 200,
        dawn_end: 100,
        dusk_start: 700,
        night_start: 800,
        rules_version: 2,
        ...world,
      },
      catalog: {
        bodies: { fly: { frame: "egocentric", actions, channels: { body: CHANNEL } } },
        constants: { need_max: 500, light_max: 1000, night_light_threshold: 500, ...extraConstants },
      },
      motd: "",
    },
  };
}

function observation(events: Array<{ type: string }> = [], tick = 5): ServerMessage {
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

describe("the garden view model", () => {
  it("turns world and frame into tiles, fruit, flies and bubbles in draw order", () => {
    const garden = emptyGarden();
    expect(drawList(garden)).toEqual([]);
    applySpectatorMessage(garden, WORLD);
    expect(drawList(garden).filter((d) => d.kind === "tile")).toHaveLength(6);
    applySpectatorMessage(garden, frame([fly(), fly({ agent_id: 2, alive: false, x: 0, y: 0 })]));
    const list = drawList(garden);
    expect(list.map((d) => d.kind)).toEqual([...Array(6).fill("tile"), "resource", "fly", "bubble"]);
    expect(list.find((d) => d.kind === "fly")).toMatchObject({ x: 2, y: 1, facing: 2, mine: false, away: false });
    expect(list.find((d) => d.kind === "resource")).toMatchObject({ resource: 1, amount: 4 });
  });

  it("leaves out a resource kind it has no sprite for", () => {
    const garden = emptyGarden();
    applySpectatorMessage(garden, WORLD);
    applySpectatorMessage(garden, frame([], { resources: [{ x: 0, y: 0, kind: 7, amount: 2 }] }));
    expect(drawList(garden).some((d) => d.kind === "resource")).toBe(false);
  });

  it("marks the human's fly and draws speech bubbles", () => {
    const garden = emptyGarden();
    applySpectatorMessage(garden, WORLD);
    applyAgentMessage(garden, {
      v: 1,
      type: "joined",
      payload: { agent_id: 7, lineage: 2, name: "Amber Zip", tick: 4, reattached: false },
    });
    applySpectatorMessage(garden, frame([fly({ agent_id: 7, say: "hi", connected: false })]));
    const list = drawList(garden);
    expect(list.find((d) => d.kind === "fly")).toMatchObject({ mine: true, away: true });
    expect(list.find((d) => d.kind === "say")).toMatchObject({ text: "hi" });
    expect(myFly(garden)?.name).toBe("Dusty Wing");
  });

  it("keeps only the last log lines and records deaths", () => {
    const garden = emptyGarden();
    for (let i = 0; i < LOG_LINES + 3; i++) {
      applySpectatorMessage(garden, { v: 1, type: "chronicle", payload: { tick: i, text: `line ${i}` } });
    }
    expect(garden.chronicle).toHaveLength(LOG_LINES);
    expect(garden.chronicle[0]).toBe("line 3");
    applyAgentMessage(garden, {
      v: 1,
      type: "died",
      payload: {
        agent_id: 7,
        lineage: 2,
        name: "Amber Zip",
        tick: 90,
        causes: ["starvation"],
        stats: { lifespan: 90, days: 0, death_causes: ["starvation"], bites: 0, drinks: 1, rest_ticks: 2, bumps: 0, tiles_explored: 5, mean_wellbeing: 0.5 },
      },
    });
    expect(garden.me.died?.causes).toEqual(["starvation"]);
    expect(garden.me.joined).toBeNull();
  });

  it("tints the night and names the time of day", () => {
    expect(nightAlpha(1000)).toBe(0);
    expect(nightAlpha(500)).toBeCloseTo(0.28, 2);
    expect(nightAlpha(0)).toBe(0.55);
    expect(nightAlpha(50, 100)).toBeCloseTo(0.28, 2); // a world with another light_max
    expect(timeOfDay(0)).toBe("dawn");
    expect(timeOfDay(300)).toBe("day");
    expect(timeOfDay(750)).toBe("dusk");
    expect(timeOfDay(1000)).toBe("night");
  });

  it("takes the scales and the day's turning points from the welcome, with fallbacks", () => {
    const garden = emptyGarden();
    expect(constants(garden)).toEqual(DEFAULT_CONSTANTS);
    expect(dayPhases(garden)).toEqual(DEFAULT_PHASES);
    applySpectatorMessage(garden, welcome({ night_light_threshold: 300 }, { dawn_end: 20 }));
    expect(constants(garden)).toMatchObject({ nightLight: 300, needMax: 500 });
    expect(dayPhases(garden)).toEqual({
      dayLength: 200,
      dawnEnd: 20,
      duskStart: DEFAULT_PHASES.duskStart,
      nightStart: DEFAULT_PHASES.nightStart,
    });
    expect(timeOfDay(210, dayPhases(garden))).toBe("dawn"); // 10 ticks into the second day
  });
});

describe("the human's own fly", () => {
  it("takes the welcome from whichever socket brings it first", () => {
    const garden = emptyGarden();
    applyAgentMessage(garden, welcome());
    expect(garden.welcome?.role).toBe("agent");
    expect(flyActions(garden)).toEqual(["idle", "move_n"]);
    applySpectatorMessage(garden, welcome());
    expect(garden.welcome?.role).toBe("agent"); // the first one stays
  });

  it("says so when the catalog offers a fly no actions at all", () => {
    const garden = emptyGarden();
    applyAgentMessage(garden, observation());
    expect(garden.notice).toMatch(/no actions/);
    garden.notice = null;
    garden.welcome = null;
    applyAgentMessage(garden, welcome());
    applyAgentMessage(garden, observation());
    expect(garden.notice).toBeNull();
  });

  it("shows a non-fatal error and stops waiting for the hatch it refused", () => {
    const garden = emptyGarden();
    garden.me.hatching = true;
    applyAgentMessage(garden, {
      v: 1,
      type: "error",
      payload: { code: "no_fly", message: "join first", fatal: false },
    });
    expect(garden.notice).toBe("no_fly: join first");
    expect(garden.me.hatching).toBe(false);
  });

  it("is hatching until joined lands", () => {
    const garden = emptyGarden();
    garden.me.hatching = true;
    applyAgentMessage(garden, {
      v: 1,
      type: "joined",
      payload: { agent_id: 7, lineage: 2, name: "Amber Zip", tick: 4, reattached: false },
    });
    expect(garden.me.hatching).toBe(false);
  });
});

describe("envelopes", () => {
  it("encode hello and decode known server messages, ignoring the rest", () => {
    const wire = JSON.parse(encode(hello("alice", "dev", "agent")));
    expect(wire).toMatchObject({ v: 1, type: "hello", payload: { owner: "alice", role: "agent", protocol: 1 } });
    expect(decode('{"v": 1, "type": "chronicle", "payload": {"tick": 1, "text": "x"}}')?.type).toBe("chronicle");
    expect(decode('{"v": 1, "type": "weather", "payload": {}}')).toBeNull();
    expect(decode('{"v": 2, "type": "frame", "payload": {}}')).toBeNull();
    expect(decode("nope")).toBeNull();
    expect(decode("[]")).toBeNull();
  });

  it("strip from a say what the server would close the connection over", () => {
    expect(sayText("  fruit? 🍎  ")).toBe("fruit? 🍎");
    expect(sayText("\x1b[2Jcleared\x07")).toBe("[2Jcleared"); // the escape itself is gone
    expect(sayText("two\nlines")).toBe("twolines");
    expect(sayText("x".repeat(60))).toHaveLength(SAY_MAX);
    expect(say("hi\x00")).toMatchObject({ type: "say", payload: { text: "hi" } });
  });
});
