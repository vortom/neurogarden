import { describe, expect, it } from "vitest";
import { Keys, actionId } from "../src/input";

const ACTIONS = ["idle", "move_n", "move_e", "move_s", "move_w", "consume", "rest"];

describe("keys", () => {
  it("map arrows, wasd, space and r; the last held key wins and repeats", () => {
    const keys = new Keys();
    expect(keys.nextAction()).toBe("idle");
    keys.down("ArrowUp");
    expect(keys.nextAction()).toBe("move_n");
    expect(keys.nextAction()).toBe("move_n"); // still held: repeats
    keys.down("ArrowRight");
    expect(keys.nextAction()).toBe("move_e");
    keys.up("ArrowRight");
    expect(keys.nextAction()).toBe("move_n");
    keys.up("ArrowUp");
    keys.down(" ");
    expect(keys.nextAction()).toBe("consume");
    keys.down("r");
    expect(keys.nextAction()).toBe("rest");
    keys.clear();
    expect(keys.nextAction()).toBe("idle");
  });

  it("keeps a quick tap for exactly one tick", () => {
    const keys = new Keys();
    keys.down("ArrowLeft");
    keys.up("ArrowLeft"); // released before the next observation arrived
    expect(keys.nextAction()).toBe("move_w");
    expect(keys.nextAction()).toBe("idle");
    keys.down("ArrowUp");
    keys.up("ArrowUp");
    keys.down("ArrowDown");
    keys.up("ArrowDown");
    expect(keys.nextAction()).toBe("move_s"); // the latest tap wins
  });

  it("ignore keys that are not game keys", () => {
    const keys = new Keys();
    keys.down("Escape");
    expect(keys.nextAction()).toBe("idle");
    expect(keys.isGameKey("a")).toBe(true);
    expect(keys.isGameKey("Escape")).toBe(false);
  });

  it("resolve names through the catalog's action order, unknown names to idle", () => {
    expect(actionId(ACTIONS, "move_w")).toBe(4);
    expect(actionId(ACTIONS, "consume")).toBe(5);
    expect(actionId(ACTIONS, "fly")).toBe(0);
    expect(actionId(["idle", "rest", "move_n"], "rest")).toBe(1);
  });
});
