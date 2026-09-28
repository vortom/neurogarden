import { describe, expect, it } from "vitest";
import { ghostBanner, ghostQuery, parseGhost } from "../src/ghost";
import { replay } from "../src/protocol";
import { applySpectatorMessage, drawList, emptyGarden, focusId } from "../src/state";
import type { AgentView, ReplayInfo, ServerMessage } from "../src/types";

function info(overrides: Partial<ReplayInfo> = {}): ReplayInfo {
  return {
    owner: "alice",
    lineage: 2,
    name: "Dusty Wing",
    born_tick: 100,
    died_tick: 250,
    lifespan: 151,
    causes: ["starvation"],
    speed: 4,
    done: false,
    ...overrides,
  };
}

function fly(overrides: Partial<AgentView> = {}): AgentView {
  return {
    agent_id: 9,
    owner: "alice",
    lineage: 2,
    name: "Dusty Wing",
    x: 1,
    y: 1,
    facing: 2,
    satiety: 500,
    hydration: 500,
    energy: 500,
    health: 1000,
    age: 3,
    alive: true,
    connected: true,
    mood: "content",
    say: "",
    ...overrides,
  };
}

const WORLD: ServerMessage = {
  v: 1,
  type: "world",
  payload: { map_name: "tiny", width: 2, height: 2, terrain: [[1, 1], [1, 1]] },
};

describe("the ghost URL", () => {
  it("names an owner, a lineage and a speed within the server's bounds", () => {
    expect(parseGhost("?ghost=alice/2")).toEqual({ owner: "alice", lineage: 2, speed: 4 });
    expect(parseGhost("?ghost=a.b-c_d/12&speed=16")).toMatchObject({ owner: "a.b-c_d", lineage: 12, speed: 16 });
    expect(parseGhost("?ghost=alice/2&speed=999")?.speed).toBe(64);
    expect(parseGhost("?ghost=alice/2&speed=0")?.speed).toBe(0.25);
    expect(parseGhost("?ghost=alice/2&speed=fast")?.speed).toBe(4);
  });

  it("asks for the living garden when it makes no sense", () => {
    for (const bad of ["", "?token=x", "?ghost=", "?ghost=alice", "?ghost=/2", "?ghost=alice/0", "?ghost=alice/two", "?ghost=al ice/1", "?ghost=alice/1.5"]) {
      expect(parseGhost(bad)).toBeNull();
    }
  });

  it("round-trips through the link the hall of flies offers", () => {
    expect(ghostQuery("alice", 2)).toBe("?ghost=alice%2F2");
    expect(parseGhost(ghostQuery("alice", 2))).toEqual({ owner: "alice", lineage: 2, speed: 4 });
    expect(parseGhost(ghostQuery("bob", 7, 16))).toEqual({ owner: "bob", lineage: 7, speed: 16 });
    expect(replay("alice", 2, 8)).toEqual({ v: 1, type: "replay", payload: { owner: "alice", lineage: 2, speed: 8 } });
  });
});

describe("a ghost in the garden", () => {
  it("is the fly the page is about, and its frames are drawn like anyone's", () => {
    const garden = emptyGarden();
    applySpectatorMessage(garden, WORLD);
    applySpectatorMessage(garden, { v: 1, type: "replay", payload: info() });
    expect(garden.ghost?.name).toBe("Dusty Wing");
    expect(focusId(garden)).toBe(-1); // no frame yet
    applySpectatorMessage(garden, {
      v: 1,
      type: "frame",
      payload: { tick: 100, day: 1, light: 1000, resources: [], agents: [fly(), fly({ agent_id: 3, owner: "bob", lineage: 1 })], events: [], scores: [] },
    });
    expect(focusId(garden)).toBe(9);
    const flies = drawList(garden).filter((d) => d.kind === "fly");
    expect(flies.map((d) => d.kind === "fly" && d.mine)).toEqual([true, false]);
  });

  it("forgets the live frame when a ghost starts, and keeps the last one when it ends", () => {
    const garden = emptyGarden();
    applySpectatorMessage(garden, {
      v: 1,
      type: "frame",
      payload: { tick: 5000, day: 5, light: 1000, resources: [], agents: [], events: [], scores: [] },
    });
    applySpectatorMessage(garden, { v: 1, type: "replay", payload: info() });
    expect(garden.frame).toBeNull();
    applySpectatorMessage(garden, {
      v: 1,
      type: "frame",
      payload: { tick: 250, day: 1, light: 1000, resources: [], agents: [], events: [], scores: [] },
    });
    applySpectatorMessage(garden, { v: 1, type: "replay", payload: info({ done: true }) });
    expect(garden.frame?.tick).toBe(250);
    expect(garden.ghost?.done).toBe(true);
  });

  it("has a banner that says where it is and when it has faded", () => {
    expect(ghostBanner(info(), 130)).toBe("👻 alice's Dusty Wing #2 · 4× · tick 130 of 250");
    expect(ghostBanner(info(), null)).toBe("👻 alice's Dusty Wing #2 · 4×");
    expect(ghostBanner(info({ died_tick: null, lifespan: null }), 130)).toBe("👻 alice's Dusty Wing #2 · 4× · tick 130");
    expect(ghostBanner(info({ done: true }), 250)).toBe("👻 alice's Dusty Wing #2 has faded");
    expect(ghostBanner(info({ done: true, died_tick: null }), 250)).toBe("👻 alice's Dusty Wing #2 has caught up with the living");
  });
});
