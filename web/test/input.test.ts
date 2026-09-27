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

  it("fold Shift and CapsLock away so a key never sticks down", () => {
    const keys = new Keys();
    expect(keys.isGameKey("W")).toBe(true);
    keys.down("w");
    expect(keys.nextAction()).toBe("move_n");
    keys.up("W"); // Shift went down mid-press: the browser reports the upper case
    expect(keys.nextAction()).toBe("idle");
    keys.down("R"); // CapsLock on: R still rests
    expect(keys.nextAction()).toBe("rest");
  });

  it("let a tap beat a key held since before it, for one tick only", () => {
    const keys = new Keys();
    keys.down("ArrowUp"); // held down the whole time
    expect(keys.nextAction()).toBe("move_n");
    keys.down(" ");
    keys.up(" ");
    expect(keys.nextAction()).toBe("consume"); // the tap is newer than the hold
    expect(keys.nextAction()).toBe("move_n"); // and the hold takes over again
  });

  it("let a key held since after a tap win over it", () => {
    const keys = new Keys();
    keys.down(" ");
    keys.up(" ");
    keys.down("ArrowUp"); // pressed after the tap
    expect(keys.nextAction()).toBe("move_n");
    expect(keys.nextAction()).toBe("move_n");
  });

  it("treat key auto-repeat as one press, so it cannot swallow a tap", () => {
    const keys = new Keys();
    keys.down("ArrowUp");
    keys.down(" ");
    keys.up(" ");
    for (let i = 0; i < 5; i++) keys.down("ArrowUp"); // the OS repeating the held key
    expect(keys.nextAction()).toBe("consume");
    expect(keys.nextAction()).toBe("move_n");
  });

  it("resolve names through the catalog's action order, unknown names to idle", () => {
    expect(actionId(ACTIONS, "move_w")).toBe(4);
    expect(actionId(ACTIONS, "consume")).toBe(5);
    expect(actionId(ACTIONS, "fly")).toBe(0);
    expect(actionId(["idle", "rest", "move_n"], "rest")).toBe(1);
  });
});
