// One letter per colour. "." is transparent. Kept small on purpose: a Game Boy-ish garden.
export const PALETTE: Record<string, string> = {
  g: "#4f9d3a", // grass
  G: "#63b04a", // grass, light
  d: "#3d7f2e", // grass, dark
  p: "#f28bb8", // flower petal
  y: "#ffe066", // flower heart / sun
  r: "#7a7f86", // rock
  R: "#a0a5ab", // rock, light
  s: "#545860", // rock, shadow
  w: "#3b7bd4", // water
  W: "#7fb2ef", // water, light
  b: "#2a5fb0", // water, deep
  t: "#6b4a2b", // trunk
  T: "#8a6238", // trunk, light
  c: "#2f7a2f", // canopy
  C: "#4faa4f", // canopy, light
  n: "#c9a45c", // nest straw
  N: "#e2c283", // nest straw, light
  m: "#a67c3a", // nest straw, dark
  a: "#d9333a", // apple
  A: "#ff7b7b", // apple, highlight
  l: "#3f8f3f", // leaf
  k: "#2b2b2b", // fly body
  K: "#4a4a4a", // fly body, light
  e: "#e04040", // fly eye
  v: "#cfd8e3", // wing
  V: "#eef3f8", // wing, light
  o: "#ffffff", // white
  x: "#1b1b1b", // near black
  h: "#ff6f91", // heart / sparkle pink
  z: "#8fb8ff", // sleepy blue
  u: "#9ad0ff", // droplet light
  q: "#ffb347", // warning orange
};

export function isPaletteChar(ch: string): boolean {
  return ch === "." || Object.prototype.hasOwnProperty.call(PALETTE, ch);
}
