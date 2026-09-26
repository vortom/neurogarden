import { describe, expect, it } from "vitest";
import {
  BUBBLE,
  FLY_FRAMES,
  MOOD_BUBBLES,
  TERRAIN_SPRITES,
  TILE,
  allSprites,
  fruitSprite,
  groundVariant,
  validateSprite,
} from "../src/sprites";

const CATALOG_TERRAIN = { void: 0, ground: 1, rock: 2, water: 3, tree: 4, nest: 5 };
const CATALOG_MOODS = ["content", "hungry", "thirsty", "sleepy", "desperate", "dying"];

describe("sprites", () => {
  it("are all the right size and use only palette colours", () => {
    for (const [name, sprite, size] of allSprites()) {
      expect(validateSprite(sprite, size), name).toBeNull();
    }
    expect(allSprites().length).toBeGreaterThan(12);
  });

  it("cover every terrain the catalog can send", () => {
    for (const id of Object.values(CATALOG_TERRAIN)) expect(TERRAIN_SPRITES[id]).toBeDefined();
  });

  it("have a bubble for every mood that gets one, at bubble size", () => {
    for (const mood of CATALOG_MOODS) {
      expect(MOOD_BUBBLES[mood], mood).toBeDefined();
      expect(validateSprite(MOOD_BUBBLES[mood]!, BUBBLE)).toBeNull();
    }
  });

  it("pick a fruit sprite by bites left and scatter flowers deterministically", () => {
    expect(fruitSprite(4)).not.toBe(fruitSprite(2));
    expect(fruitSprite(2)).not.toBe(fruitSprite(1));
    expect(fruitSprite(1)).toBe(fruitSprite(0));
    expect(groundVariant(0, 0)).not.toBe(groundVariant(1, 0));
    expect(groundVariant(3, 5)).toBe(groundVariant(3, 5));
    const flowery = [];
    for (let y = 0; y < 24; y++) for (let x = 0; x < 32; x++) if (groundVariant(x, y) !== groundVariant(1, 0)) flowery.push([x, y]);
    expect(flowery.length).toBeGreaterThan(20);
    expect(flowery.length).toBeLessThan(200);
  });

  it("validateSprite reports what is wrong", () => {
    expect(validateSprite(["....".repeat(4)], TILE)).toMatch(/rows/);
    expect(validateSprite(Array(16).fill("..."), TILE)).toMatch(/columns/);
    expect(validateSprite(Array(16).fill("?".repeat(16)), TILE)).toMatch(/unknown colour/);
    expect(FLY_FRAMES.length).toBe(2);
  });
});
