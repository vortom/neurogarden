// Every sprite is 16 rows of 16 palette characters (bubbles are 8x8). Authored by hand.
import { isPaletteChar } from "./palette";

export const TILE = 16;
export const BUBBLE = 8;

export type Sprite = readonly string[];

const GROUND: Sprite = [
  "gggggggggggggggg",
  "gggggGgggggggggg",
  "ggggggggggggdggg",
  "gggggggggggggggg",
  "ggdggggggggggggg",
  "gggggggggGgggggg",
  "gggggggggggggggg",
  "gggggggggggggggg",
  "ggggggggggggggGg",
  "gggGgggggggggggg",
  "gggggggggggggggg",
  "ggggggggdggggggg",
  "gggggggggggggggg",
  "gggggggggggggggg",
  "gGgggggggggggggg",
  "ggggggggggggggdg",
];

const FLOWERS: Sprite = [
  "gggggggggggggggg",
  "ggggggggggggpggg",
  "gggggggggggpypgg",
  "ggggggggggggpggg",
  "gggggggggggggggg",
  "ggpggggggggggggg",
  "gpypgggggggggggg",
  "ggpggggggggggggg",
  "gggggggggggggggg",
  "gggggggggggggggg",
  "gggggggggggggggg",
  "gggggggggggggggg",
  "gggggggggggpgggg",
  "ggggggggggpypggg",
  "gggggggggggpgggg",
  "gggggggggggggggg",
];

const ROCK: Sprite = [
  "gggggggggggggggg",
  "ggggggRRRRgggggg",
  "ggggRRRRRRRRgggg",
  "gggRRRrrrrRRRggg",
  "ggRRrrrrrrrrRRgg",
  "ggRrrrrrrrrrrRgg",
  "ggrrrrrrrrrrrrgg",
  "ggrrrrrrrrrrrrgg",
  "ggrrrrrrrrrrrsgg",
  "ggrrrrrrrrrrrsgg",
  "ggsrrrrrrrrrrsgg",
  "ggssrrrrrrrrssgg",
  "gggsssrrrrsssggg",
  "ggggssssssssgggg",
  "ggggggssssgggggg",
  "gggggggggggggggg",
];

const WATER_A: Sprite = [
  "wwwwwwwwwwwwwwww",
  "wwWWwwwwwwwwwwww",
  "wwwwwwwwwwWWWwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwbwwwwwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwwwwwwwwWWw",
  "wWWWwwwwwwwwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwwwwbwwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwWWwwwwwwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwwwwwwWWWww",
  "wwbwwwwwwwwwwwww",
  "wwwwwwwwwwwwwwww",
];

const WATER_B: Sprite = [
  "wwwwwwwwwwwwwwww",
  "wwwWWwwwwwwwwwww",
  "wwwwwwwwwwwWWWww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwwbwwwwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwwwwwwwWWww",
  "wwWWWwwwwwwwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwwwwwbwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwWWwwwwwwwww",
  "wwwwwwwwwwwwwwww",
  "wwwwwwwwwwwwWWWw",
  "wwwbwwwwwwwwwwww",
  "wwwwwwwwwwwwwwww",
];

const TREE: Sprite = [
  "ggggggccccgggggg",
  "gggggcCCCCcggggg",
  "ggggcCCCCCCcgggg",
  "gggcCCCcccCCcggg",
  "gggcCCcccccCcggg",
  "ggcCCccccccCCcgg",
  "ggcCcccccccccdgg",
  "ggccccccccccccgg",
  "gggcccccccccdggg",
  "ggggcccccccdgggg",
  "gggggccttccggggg",
  "ggggggggTtgggggg",
  "ggggggggTtgggggg",
  "gggggggtTttggggg",
  "ggggggttttttgggg",
  "gggggggggggggggg",
];

const NEST: Sprite = [
  "gggggggggggggggg",
  "gggggnnnnnnggggg",
  "gggnnNNNNNNnnggg",
  "ggnNNnnnnnnNNngg",
  "gnNnnmmmmmmnnNng",
  "gnNnmmmmmmmmnNng",
  "gnnnmmmmmmmmnnng",
  "gnnnmmmmmmmmnnng",
  "gnnnmmmmmmmmnnng",
  "gnNnmmmmmmmmnNng",
  "gnNnnmmmmmmnnNng",
  "ggnNNnnnnnnNNngg",
  "gggnnNNNNNNnnggg",
  "ggggggnnnngggggg",
  "gggggggggggggggg",
  "gggggggggggggggg",
];

const FRUIT_FULL: Sprite = [
  "................",
  "........l.......",
  ".......ll.......",
  ".....aaaaaa.....",
  "....aaAaaaaa....",
  "...aaAAaaaaaa...",
  "...aaAaaaaaaa...",
  "...aaaaaaaaaa...",
  "...aaaaaaaaaa...",
  "...aaaaaaaaaa...",
  "....aaaaaaaa....",
  ".....aaaaaa.....",
  "................",
  "................",
  "................",
  "................",
];

const FRUIT_HALF: Sprite = [
  "................",
  "........l.......",
  ".......ll.......",
  "......aaaa......",
  ".....aAaaaa.....",
  ".....aAaaaaa....",
  ".....aaaaaaa....",
  ".....aaaaaaa....",
  "......aaaaa.....",
  ".......aaa......",
  "................",
  "................",
  "................",
  "................",
  "................",
  "................",
];

const FRUIT_BITE: Sprite = [
  "................",
  "................",
  "........l.......",
  ".......aaa......",
  "......aAaaa.....",
  "......aaaaa.....",
  ".......aaa......",
  "................",
  "................",
  "................",
  "................",
  "................",
  "................",
  "................",
  "................",
  "................",
];

// The fly faces south (down): head and eyes at the bottom. Other facings rotate this sprite.
const FLY_A: Sprite = [
  "................",
  "..VVV......VVV..",
  ".VvvvV....VvvvV.",
  ".VvvvvV..VvvvvV.",
  "..VvvvvKKvvvvV..",
  "...VvvKkkKvvV...",
  "....VKkkkkKV....",
  ".....kkkkkk.....",
  ".....kkkkkk.....",
  ".....KkkkkK.....",
  "....KkkkkkkK....",
  "....keekkeek....",
  "....keekkeek....",
  ".....kkkkkk.....",
  "......k..k......",
  "................",
];

const FLY_B: Sprite = [
  "................",
  "................",
  "....VV....VV....",
  "...VvvV..VvvV...",
  "...VvvvKKvvvV...",
  "....VvKkkKvV....",
  ".....VKkkKV.....",
  ".....kkkkkk.....",
  ".....kkkkkk.....",
  ".....KkkkkK.....",
  "....KkkkkkkK....",
  "....keekkeek....",
  "....keekkeek....",
  ".....kkkkkk.....",
  "......k..k......",
  "................",
];

const BUBBLE_CONTENT: Sprite = [
  "........",
  "...h....",
  "..hhh...",
  ".hhhhh..",
  "..hhh...",
  "...h....",
  "........",
  "........",
];
const BUBBLE_HUNGRY: Sprite = [
  "...l....",
  "..aaaa..",
  ".aAaaaa.",
  ".aaaaaa.",
  ".aaaaaa.",
  "..aaaa..",
  "........",
  "........",
];
const BUBBLE_THIRSTY: Sprite = [
  "...u....",
  "...u....",
  "..uuu...",
  ".uuuuu..",
  ".uWuuu..",
  ".uuuuu..",
  "..uuu...",
  "........",
];
const BUBBLE_SLEEPY: Sprite = [
  "........",
  ".zzzz...",
  "...z....",
  "..z.....",
  ".zzzz...",
  ".....zz.",
  "......z.",
  ".....zz.",
];
const BUBBLE_DESPERATE: Sprite = [
  "...qq...",
  "...qq...",
  "...qq...",
  "...qq...",
  "...qq...",
  "........",
  "...qq...",
  "...qq...",
];
const BUBBLE_DYING: Sprite = [
  "..oooo..",
  ".oooooo.",
  ".oxooxo.",
  ".oooooo.",
  "..oooo..",
  "..o.o...",
  "........",
  "........",
];

export const TERRAIN_SPRITES: Record<number, Sprite> = {
  0: ROCK, // VOID never appears in a spectator's world; rock keeps the map opaque if it did
  1: GROUND,
  2: ROCK,
  3: WATER_A,
  4: TREE,
  5: NEST,
};
export const GROUND_FLOWERS = FLOWERS;
export const WATER_FRAMES: readonly Sprite[] = [WATER_A, WATER_B];
export const FRUIT_SPRITES: readonly Sprite[] = [FRUIT_BITE, FRUIT_HALF, FRUIT_FULL]; // by bites left
export const FLY_FRAMES: readonly Sprite[] = [FLY_A, FLY_B];
export const MOOD_BUBBLES: Record<string, Sprite> = {
  content: BUBBLE_CONTENT,
  hungry: BUBBLE_HUNGRY,
  thirsty: BUBBLE_THIRSTY,
  sleepy: BUBBLE_SLEEPY,
  desperate: BUBBLE_DESPERATE,
  dying: BUBBLE_DYING,
};

export function fruitSprite(bitesLeft: number): Sprite {
  if (bitesLeft <= 1) return FRUIT_BITE;
  if (bitesLeft <= 2) return FRUIT_HALF;
  return FRUIT_FULL;
}

/** Ground tiles get flowers on a fixed, scattered subset so the meadow is not a flat green. */
export function groundVariant(x: number, y: number): Sprite {
  return (x * 7 + y * 13) % 11 === 0 ? FLOWERS : GROUND;
}

export function validateSprite(sprite: Sprite, size: number): string | null {
  if (sprite.length !== size) return `has ${sprite.length} rows, expected ${size}`;
  for (const [index, row] of sprite.entries()) {
    if (row.length !== size) return `row ${index} has ${row.length} columns, expected ${size}`;
    for (const ch of row) if (!isPaletteChar(ch)) return `row ${index} has unknown colour ${ch}`;
  }
  return null;
}

export function allSprites(): Array<[string, Sprite, number]> {
  const out: Array<[string, Sprite, number]> = [];
  for (const [id, sprite] of Object.entries(TERRAIN_SPRITES)) out.push([`terrain ${id}`, sprite, TILE]);
  out.push(["flowers", FLOWERS, TILE]);
  WATER_FRAMES.forEach((s, i) => out.push([`water ${i}`, s, TILE]));
  FRUIT_SPRITES.forEach((s, i) => out.push([`fruit ${i}`, s, TILE]));
  FLY_FRAMES.forEach((s, i) => out.push([`fly ${i}`, s, TILE]));
  for (const [mood, sprite] of Object.entries(MOOD_BUBBLES)) out.push([`bubble ${mood}`, sprite, BUBBLE]);
  return out;
}
