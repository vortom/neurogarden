// Canvas 2D: rasterise the sprites once, then draw the draw list every animation frame.
import { PALETTE } from "./palette";
import {
  BUBBLE,
  FLY_FRAMES,
  MOOD_BUBBLES,
  TERRAIN_SPRITES,
  TILE,
  WATER_FRAMES,
  fruitSprite,
  groundVariant,
  type Sprite,
} from "./sprites";
import { type Draw, type Garden, drawList, nightAlpha } from "./state";

const ANIMATION_MS = 250;

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
  private readonly cache = new Map<Sprite, HTMLCanvasElement>();
  private width = 0;
  private height = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    this.ctx.imageSmoothingEnabled = false;
  }

  private image(sprite: Sprite, size: number): HTMLCanvasElement {
    let image = this.cache.get(sprite);
    if (image === undefined) {
      image = rasterise(sprite, size);
      this.cache.set(sprite, image);
    }
    return image;
  }

  /** Size the canvas for the world; the CSS scale keeps pixels crisp. */
  fit(width: number, height: number, scale: number): void {
    this.width = width;
    this.height = height;
    this.canvas.width = width * TILE;
    this.canvas.height = height * TILE;
    this.canvas.style.width = `${width * TILE * scale}px`;
    this.canvas.style.height = `${height * TILE * scale}px`;
    this.ctx.imageSmoothingEnabled = false;
  }

  draw(garden: Garden, now: number): void {
    const ctx = this.ctx;
    if (garden.world === null) {
      ctx.fillStyle = "#1b1b1b";
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      return;
    }
    if (garden.world.width !== this.width || garden.world.height !== this.height) {
      this.fit(garden.world.width, garden.world.height, Number(this.canvas.dataset.scale ?? 2));
    }
    const phase = Math.floor(now / ANIMATION_MS) % 2;
    for (const item of drawList(garden)) this.drawOne(item, phase);
    const alpha = nightAlpha(garden.frame?.light ?? 1000);
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
      case "tile": {
        let sprite = TERRAIN_SPRITES[item.terrain] ?? TERRAIN_SPRITES[2]!;
        if (item.terrain === 1) sprite = groundVariant(item.x, item.y);
        if (item.terrain === 3) sprite = WATER_FRAMES[phase]!;
        ctx.drawImage(this.image(sprite, TILE), px, py);
        return;
      }
      case "fruit":
        ctx.drawImage(this.image(fruitSprite(item.bites), TILE), px, py);
        return;
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
        ctx.drawImage(this.image(sprite, BUBBLE), px + TILE - BUBBLE + 2, py - BUBBLE + 4);
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
