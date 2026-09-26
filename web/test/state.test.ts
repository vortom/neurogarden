import { describe, expect, it } from "vitest";
import { decode, encode, hello } from "../src/protocol";
import {
  LOG_LINES,
  applyAgentMessage,
  applySpectatorMessage,
  drawList,
  emptyGarden,
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

describe("the garden view model", () => {
  it("turns world and frame into tiles, fruit, flies and bubbles in draw order", () => {
    const garden = emptyGarden();
    expect(drawList(garden)).toEqual([]);
    applySpectatorMessage(garden, WORLD);
    expect(drawList(garden).filter((d) => d.kind === "tile")).toHaveLength(6);
    applySpectatorMessage(garden, frame([fly(), fly({ agent_id: 2, alive: false, x: 0, y: 0 })]));
    const list = drawList(garden);
    expect(list.map((d) => d.kind)).toEqual([...Array(6).fill("tile"), "fruit", "fly", "bubble"]);
    expect(list.find((d) => d.kind === "fly")).toMatchObject({ x: 2, y: 1, facing: 2, mine: false, away: false });
    expect(list.find((d) => d.kind === "fruit")).toMatchObject({ bites: 4 });
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
    expect(timeOfDay(0, 1200)).toBe("dawn");
    expect(timeOfDay(300, 1200)).toBe("day");
    expect(timeOfDay(750, 1200)).toBe("dusk");
    expect(timeOfDay(1000, 1200)).toBe("night");
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
});
