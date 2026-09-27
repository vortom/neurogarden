import { describe, expect, it } from "vitest";
import {
  BUBBLE,
  FLY_FRAMES,
  MOOD_BUBBLES,
  RESOURCE_SPRITES,
  TERRAIN_SPRITES,
  TILE,
  WATER_FRAMES,
  allSprites,
  fruitSprite,
  groundVariant,
  terrainSprite,
  validateSprite,
} from "../src/sprites";
import { RESOURCE, TERRAIN } from "../src/types";

const CATALOG_TERRAIN = { void: 0, ground: 1, rock: 2, water: 3, tree: 4, nest: 5 };
const CATALOG_RESOURCES = { none: 0, fruit: 1 };
const CATALOG_MOODS = ["content", "hungry", "thirsty", "sleepy", "desperate", "dying"];

describe("sprites", () => {
  it("are all the right size and use only palette colours", () => {
    for (const [name, sprite, size] of allSprites()) {
      expect(validateSprite(sprite, size), name).toBeNull();
    }
    expect(allSprites().length).toBeGreaterThan(12);
  });

  it("cover every terrain the catalog can send", () => {
    expect(TERRAIN).toEqual(CATALOG_TERRAIN);
    for (const id of Object.values(CATALOG_TERRAIN)) expect(TERRAIN_SPRITES[id]).toBeDefined();
  });

  it("cover every resource kind the catalog can send but `none`", () => {
    expect(RESOURCE).toEqual(CATALOG_RESOURCES);
    for (const [name, id] of Object.entries(CATALOG_RESOURCES)) {
      if (name === "none") {
        expect(RESOURCE_SPRITES[id], name).toBeUndefined();
        continue;
      }
      const pick = RESOURCE_SPRITES[id];
      expect(pick, name).toBeDefined();
      expect(validateSprite(pick!(3), TILE), name).toBeNull();
    }
  });

  it("animate water and scatter ground, and fall back to rock for an unknown tile", () => {
    expect(terrainSprite(TERRAIN.water, 0, 0, 0)).toBe(WATER_FRAMES[0]);
    expect(terrainSprite(TERRAIN.water, 0, 0, 1)).toBe(WATER_FRAMES[1]);
    expect(terrainSprite(TERRAIN.ground, 0, 0, 0)).toBe(groundVariant(0, 0));
    expect(terrainSprite(TERRAIN.tree, 0, 0, 0)).toBe(TERRAIN_SPRITES[TERRAIN.tree]);
    expect(terrainSprite(99, 0, 0, 0)).toBe(TERRAIN_SPRITES[TERRAIN.rock]);
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
