import { Hud } from "./hud";
import { Keys, actionId } from "./input";
import { open, serverUrl, type Socket } from "./net";
import { PlaySession } from "./play";
import { action, join, leave, say } from "./protocol";
import { Renderer, scaleFor } from "./render";
import { applyAgentMessage, applySpectatorMessage, emptyGarden, flyActions, noFly } from "./state";
import type { WorldMap } from "./types";

const OWNER_KEY = "neurogarden.owner";
const OWNER_PATTERN = /^[A-Za-z0-9_.-]{1,64}$/;
const CONTROL_CHARS = /[\x00-\x1f\x7f]/g; // the server closes a connection that sends these
const SAY_MAX = 40;
const SIDEBAR = 360; // the aside in index.html; whatever is left over is the garden's

/** Storage is blocked in some windows and profiles; the page must run without it. */
function recall(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function remember(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // no storage: the name is simply not remembered for next time
  }
}

const garden = emptyGarden();
const canvas = document.querySelector<HTMLCanvasElement>("#garden")!;
const renderer = new Renderer(canvas);
const hud = new Hud(document.body);
const keys = new Keys();
const session = new PlaySession();

const nameInput = document.querySelector<HTMLInputElement>("#owner")!;
const playButton = document.querySelector<HTMLButtonElement>("#play")!;
const leaveButton = document.querySelector<HTMLButtonElement>("#leave")!;
const sayInput = document.querySelector<HTMLInputElement>("#say")!;
const hatchButton = document.querySelector<HTMLButtonElement>("#hatch")!;

nameInput.value = recall(OWNER_KEY);

let dirty = true;
const url = serverUrl();
open(url, "web-watcher", "spectator", {
  onMessage(message) {
    if (applySpectatorMessage(garden, message)) dirty = true;
  },
  onClose(code, reason) {
    const why = reason ? `${reason} (${code})` : `code ${code}`;
    garden.notice = `spectator connection closed: ${why} — reload the page`;
    dirty = true;
  },
});

let agent: Socket | null = null;

function play(): void {
  const owner = nameInput.value.trim();
  if (!OWNER_PATTERN.test(owner)) {
    garden.notice = "pick a name: letters, digits, _ . - (up to 64)";
    dirty = true;
    return;
  }
  remember(OWNER_KEY, owner);
  garden.me = noFly(owner);
  garden.notice = null;
  agent?.close(); // its id is retired by start(), so its close is not news
  const id = session.start();
  let socket: Socket | null = null;
  socket = open(url, owner, "agent", {
    onMessage(message) {
      const plan = session.onMessage(id, message);
      if (plan.stale) return;
      if (plan.join) {
        socket?.send(join());
        garden.me.hatching = true;
      }
      if (plan.actFor !== null) {
        socket?.send(action(plan.actFor, actionId(flyActions(garden), keys.nextAction())));
      }
      if (plan.apply && applyAgentMessage(garden, message)) dirty = true;
    },
    onClose(code, reason) {
      const plan = session.onClose(id, code, reason);
      if (plan.stale) return;
      garden.notice = plan.notice;
      garden.me.joined = null;
      garden.me.hatching = false;
      agent = null;
      dirty = true;
    },
  });
  agent = socket;
  dirty = true;
}

function stop(): void {
  session.stop();
  agent?.send(leave());
  agent?.close();
  agent = null;
  garden.me = noFly();
  keys.clear();
  dirty = true;
}

function updateButtons(): void {
  const playing = agent !== null;
  playButton.hidden = playing;
  leaveButton.hidden = !playing;
  sayInput.hidden = !playing;
  hatchButton.hidden = !(playing && garden.me.died !== null && !garden.me.hatching);
  nameInput.disabled = playing;
}

playButton.addEventListener("click", play);
leaveButton.addEventListener("click", stop);
hatchButton.addEventListener("click", () => {
  if (agent === null || garden.me.hatching) return;
  agent.send(join());
  garden.me.hatching = true;
  dirty = true;
});
nameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") play();
  event.stopPropagation();
});
sayInput.addEventListener("keydown", (event) => {
  event.stopPropagation();
  if (event.key !== "Enter") return;
  const text = sayInput.value.replace(CONTROL_CHARS, "").trim().slice(0, SAY_MAX);
  agent?.send(say(text));
  sayInput.value = "";
});
window.addEventListener("keydown", (event) => {
  if (agent === null || !keys.isGameKey(event.key)) return;
  keys.down(event.key);
  event.preventDefault();
});
window.addEventListener("keyup", (event) => keys.up(event.key));
window.addEventListener("blur", () => keys.clear());

let sized: WorldMap | null = null;

function fitToWorld(): void {
  const world = garden.world;
  if (world === null) return;
  renderer.setScale(scaleFor(world.width, window.innerWidth - SIDEBAR));
  renderer.fit(world.width, world.height);
  sized = world;
  dirty = true;
}
window.addEventListener("resize", fitToWorld);

function loop(now: number): void {
  if (garden.world !== null && garden.world !== sized) fitToWorld();
  if (dirty) {
    hud.update(garden);
    updateButtons();
  }
  renderer.draw(garden, now, dirty);
  dirty = false;
  requestAnimationFrame(loop);
}
updateButtons();
requestAnimationFrame(loop);
