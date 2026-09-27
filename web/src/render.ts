// Canvas 2D: rasterise the sprites once, then draw the draw list every animation frame.
import { PALETTE } from "./palette";
import {
  BUBBLE,
  FLY_FRAMES,
  MOOD_BUBBLES,
  RESOURCE_SPRITES,
  TILE,
  terrainSprite,
  type Sprite,
} from "./sprites";
import { type Draw, type Garden, actorList, constants, nightAlpha, terrainList } from "./state";
import type { WorldMap } from "./types";

const ANIMATION_MS = 250;
const PHASES = 2; // water ripples and wings beat on this two-frame clock
const MAX_SCALE = 3;

/** Integer pixel scale: as many whole screen pixels per world pixel as `room` allows. */
export function scaleFor(worldWidth: number, room: number): number {
  return Math.max(1, Math.min(MAX_SCALE, Math.floor(room / (worldWidth * TILE))));
}

function rasterise(sprite: Sprite, size: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  sprite.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const colour = PALETTE[row[x] ?? "."];
      if (colour === undefined) continue;
      ctx.fillStyle = colour;
      ctx.fillRect(x, y, 1, 1);
    }
  });
  return canvas;
}

export class Renderer {
  private readonly ctx: CanvasRenderingContext2D;
  /** Rasterised sprites, by sprite and by the size they were rasterised at. */
  private readonly cache = new Map<Sprite, Map<number, HTMLCanvasElement>>();
  private width = 0;
  private height = 0;
  private scale: number;
  /** The terrain, baked once per world: one layer per animation phase. */
  private layers: HTMLCanvasElement[] = [];
  private baked: WorldMap | null = null;
  private lastPhase = -1;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    scale = 2,
  ) {
    this.scale = scale;
    this.ctx = canvas.getContext("2d")!;
    this.ctx.imageSmoothingEnabled = false;
  }

  setScale(scale: number): void {
    this.scale = scale;
  }

  private image(sprite: Sprite, size: number): HTMLCanvasElement {
    let bySize = this.cache.get(sprite);
    if (bySize === undefined) {
      bySize = new Map();
      this.cache.set(sprite, bySize);
    }
    let image = bySize.get(size);
    if (image === undefined) {
      image = rasterise(sprite, size);
      bySize.set(size, image);
    }
    return image;
  }

  /** Size the canvas for the world; the CSS scale keeps pixels crisp. */
  fit(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.canvas.width = width * TILE;
    this.canvas.height = height * TILE;
    this.canvas.style.width = `${width * TILE * this.scale}px`;
    this.canvas.style.height = `${height * TILE * this.scale}px`;
    this.ctx.imageSmoothingEnabled = false;
    this.lastPhase = -1; // resizing clears the canvas: the next draw is not optional
  }

  /** The terrain only changes when a new map arrives, so draw it once per world, not per frame. */
  private bake(world: WorldMap): void {
    this.layers = [];
    for (let phase = 0; phase < PHASES; phase++) {
      const layer = document.createElement("canvas");
      layer.width = world.width * TILE;
      layer.height = world.height * TILE;
      const ctx = layer.getContext("2d")!;
      ctx.imageSmoothingEnabled = false;
      for (const item of terrainList(world)) {
        if (item.kind !== "tile") continue;
        const sprite = terrainSprite(item.terrain, item.x, item.y, phase);
        ctx.drawImage(this.image(sprite, TILE), item.x * TILE, item.y * TILE);
      }
      this.layers.push(layer);
    }
    this.baked = world;
  }

  /**
   * Draw the garden. Nothing moves between ticks except the water and the wings, so a frame
   * with neither a new message (`dirty`) nor a new animation phase is skipped entirely.
   */
  draw(garden: Garden, now: number, dirty = true): void {
    const ctx = this.ctx;
    const phase = Math.floor(now / ANIMATION_MS) % PHASES;
    const world = garden.world;
    if (world === null) {
      if (!dirty && phase === this.lastPhase) return;
      this.lastPhase = phase;
      ctx.fillStyle = "#1b1b1b";
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      return;
    }
    if (world.width !== this.width || world.height !== this.height) this.fit(world.width, world.height);
    if (world !== this.baked) this.bake(world);
    else if (!dirty && phase === this.lastPhase) return;
    this.lastPhase = phase;
    ctx.drawImage(this.layers[phase]!, 0, 0);
    for (const item of actorList(garden)) this.drawOne(item, phase);
    const alpha = nightAlpha(garden.frame?.light ?? 1000, constants(garden).lightMax);
    if (alpha > 0) {
      ctx.fillStyle = `rgba(20, 30, 90, ${alpha})`;
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  private drawOne(item: Draw, phase: number): void {
    const ctx = this.ctx;
    const px = item.x * TILE;
    const py = item.y * TILE;
    switch (item.kind) {
      case "tile": // baked into the terrain layer
        return;
      case "resource": {
        const sprite = RESOURCE_SPRITES[item.resource]?.(item.amount);
        if (sprite === undefined) return;
        ctx.drawImage(this.image(sprite, TILE), px, py);
        return;
      }
      case "fly": {
        const sprite = FLY_FRAMES[item.away ? 0 : phase]!;
        ctx.save();
        ctx.translate(px + TILE / 2, py + TILE / 2);
        ctx.rotate(((item.facing + 2) % 4) * (Math.PI / 2)); // sprite faces south (2)
        if (item.mine) {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
          ctx.lineWidth = 1;
          ctx.strokeRect(-TILE / 2 + 1.5, -TILE / 2 + 1.5, TILE - 3, TILE - 3);
        }
        if (item.away) ctx.globalAlpha = 0.6;
        ctx.drawImage(this.image(sprite, TILE), -TILE / 2, -TILE / 2);
        ctx.restore();
        return;
      }
      case "bubble": {
        const sprite = MOOD_BUBBLES[item.mood];
        if (sprite === undefined) return;
        // A fly on the top row has no sky above it: keep its bubble on the canvas.
        ctx.drawImage(this.image(sprite, BUBBLE), px + TILE - BUBBLE + 2, Math.max(0, py - BUBBLE + 4));
        return;
      }
      case "say": {
        ctx.font = "7px monospace";
        const text = item.text.slice(0, 24);
        const width = ctx.measureText(text).width + 4;
        const bx = Math.max(0, Math.min(this.canvas.width - width, px + TILE / 2 - width / 2));
        const by = Math.max(0, py - 14);
        ctx.fillStyle = "rgba(255,255,255,0.92)";
        ctx.fillRect(bx, by, width, 9);
        ctx.fillStyle = "#1b1b1b";
        ctx.fillText(text, bx + 2, by + 7);
        return;
      }
    }
  }
}
