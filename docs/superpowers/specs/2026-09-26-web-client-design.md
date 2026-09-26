# Sub-project 3 — Web client

Date: 2026-09-26
Status: approved ("go"); implemented on branch feat/web-client.
Parent: `2026-09-20-neurogarden-architecture-design.md`; builds on sub-project 2
(`2026-09-26-protocol-server-sdk-design.md`), whose protocol it speaks unchanged.

## 1. Goal

A browser page, served by the world server itself, that shows the live garden
as a small pixel-art RPG and lets a human play a fly through the same agent
protocol every other brain uses.

**Done when:** `uv run neurogarden serve` then opening `http://127.0.0.1:8765/`
shows Drosoville with pixel sprites, day and night, every fly with its mood
bubble, the naturalist's log and the leaderboard; "Play" hatches a fly you
drive with the arrow keys; dying shows an obituary and "hatch again"; the
Python and vitest suites are green; a Chrome screenshot was checked.

**Out of scope:** accounts, sound, mobile touch controls, persistence, a level
editor, anything that changes the protocol beyond additive fields.

## 2. Layout

```text
web/                          Vite + TypeScript + vitest; no runtime framework
  index.html
  package.json, vite.config.ts, tsconfig.json
  scripts/gen-types.mjs       protocol/v1/neurogarden.schema.json -> src/wire.d.ts
  src/
    wire.d.ts                 generated TypeScript types (committed)
    net.ts                    WebSocket envelopes: spectator and agent sockets
    state.ts                  view model: world + latest frame + chronicle + own fly
    palette.ts, sprites.ts    16x16 pixel maps authored in code; atlas built at start-up
    render.ts                 canvas 2D: tiles, water animation, flies, bubbles, tint
    hud.ts                    DOM: roster, log, scoreboard, play panel, obituary
    input.ts                  keys -> action ids
    main.ts                   wiring
  test/*.test.ts
src/neurogarden/server/static/   the built page (committed; `npm run build` refreshes it)
src/neurogarden/server/static_files.py   serve it on plain HTTP GET
```

`npm run types` regenerates `src/wire.d.ts` with json-schema-to-typescript;
`npm run build` writes the bundle into the Python package so
`uv run neurogarden serve` needs no Node at run time. Both outputs are
committed; a Python test checks the bundle exists and `index.html` references
only files that exist.

## 3. Server: static files

`Gateway.process_request` answers any request without `Upgrade: websocket`:
`/` and `/index.html` → the page, `/assets/*` → the bundle, `/healthz` stays,
anything else → 404. Paths are resolved inside the static directory (traversal
rejected with 404), content types by extension, `Cache-Control: no-cache`.
`neurogarden serve --no-web` disables it.

## 4. Rendering

- World canvas 32×24 tiles × 16 px = 512×384, drawn at integer scale (2× on a
  laptop, chosen from the window size) with `image-rendering: pixelated`.
- Sprites are palette-indexed pixel maps in `sprites.ts` (strings of 16
  characters per row); the atlas is rasterised once into an offscreen canvas.
  Terrain: ground (two variants), rock, water (2 animation frames), tree, nest.
  Objects: fruit (by remaining bites: 3 sizes). Flies: 2 wing frames × 4
  facings; the own fly gets a light outline. Bubbles: one 8×8 glyph per mood
  (`✨ 🍎 💧 💤 ❗ ☠️`) drawn above the fly; `say` text as a small speech box.
- Day/night: a tint overlay whose alpha follows `frame.light` (0 → deep blue at
  0.55, 1000 → none) plus a sun/moon in the status bar.
- Frames arrive 5×/s; the renderer redraws on `requestAnimationFrame` and
  interpolates nothing (pixel games step), but animates water and wings on
  their own 250 ms clock so the world breathes between ticks.

## 5. Play mode

- The page always holds a spectator socket (world, frames, chronicle).
- "Play" asks for an owner name (remembered in `localStorage`) and opens an
  agent socket: `hello(role=agent)` → `join`. Keys: arrows = move, space =
  consume, R = rest; the current key state is sent as the `action` for every
  `observation` tick (held key = repeated moves; nothing held = idle). The HUD
  shows the own fly's needs, mood and `missed`; "Say" sends a speech bubble.
- `died` → obituary card (name, lineage, lifespan, days, causes, bites/drinks)
  and "Hatch again" (a new `join`). Closing the tab leaves the fly idling; the
  same owner name reattaches on return.
- One owner = one live fly (server rule); a second tab with the same name
  supersedes the first (close 4004 shown as "another tab took over").

## 6. Tests

- vitest: every terrain/resource/mood in the catalog has a sprite and every
  sprite is 16×16 with palette-only characters; `input` maps keys to the
  catalog's action ids; the view model turns `world` + `frame` into a draw list
  (tiles, fruit, flies, bubbles) and keeps the last 8 log lines; envelope
  decoding ignores unknown types.
- Python: static serving (index, asset, 404, traversal, `--no-web`); the
  committed bundle is present and consistent.
- Manual: Chrome screenshot of the live page with two NPCs and one human.

## 7. Known limits

No reconnect on network loss (reload the page); one world per page; the
protocol's event `data` is untyped in TypeScript (`Record<string, unknown>`);
the bundle is a build artefact checked into the Python package.
