import { Hud } from "./hud";
import { Keys, actionId } from "./input";
import { open, serverUrl, type Socket } from "./net";
import { action, join, leave, say } from "./protocol";
import { Renderer } from "./render";
import { applyAgentMessage, applySpectatorMessage, emptyGarden } from "./state";

const garden = emptyGarden();
const canvas = document.querySelector<HTMLCanvasElement>("#garden")!;
const scale = Math.max(1, Math.min(3, Math.floor((window.innerWidth - 360) / 512)));
canvas.dataset.scale = String(scale);
const renderer = new Renderer(canvas);
const hud = new Hud(document.body);
const keys = new Keys();

const nameInput = document.querySelector<HTMLInputElement>("#owner")!;
const playButton = document.querySelector<HTMLButtonElement>("#play")!;
const leaveButton = document.querySelector<HTMLButtonElement>("#leave")!;
const sayInput = document.querySelector<HTMLInputElement>("#say")!;
const hatchButton = document.querySelector<HTMLButtonElement>("#hatch")!;

nameInput.value = localStorage.getItem("neurogarden.owner") ?? "";

let dirty = true;
const url = serverUrl();
open(url, "web-watcher", "spectator", {
  onMessage(message) {
    if (applySpectatorMessage(garden, message)) dirty = true;
  },
  onClose(code, reason) {
    garden.notice = `spectator connection closed (${code}) ${reason}`.trim();
    dirty = true;
  },
});

let agent: Socket | null = null;

function play(): void {
  const owner = nameInput.value.trim();
  if (!/^[A-Za-z0-9_.-]{1,64}$/.test(owner)) {
    garden.notice = "pick a name: letters, digits, _ . - (up to 64)";
    dirty = true;
    return;
  }
  localStorage.setItem("neurogarden.owner", owner);
  garden.me = { owner, joined: null, observation: null, died: null };
  garden.notice = null;
  agent?.close();
  agent = open(url, owner, "agent", {
    onMessage(message) {
      if (message.type === "welcome") {
        agent?.send(join());
      } else if (message.type === "observation") {
        const actions = garden.welcome?.catalog.bodies["fly"]?.actions ?? [];
        agent?.send(action(message.payload.tick, actionId(actions, keys.nextAction())));
      }
      if (applyAgentMessage(garden, message)) dirty = true;
    },
    onClose(code, reason) {
      const why = code === 4004 ? "another tab took over your fly" : reason || `code ${code}`;
      garden.notice = `play connection closed: ${why}`;
      garden.me.joined = null;
      agent = null;
      dirty = true;
      updateButtons();
    },
  });
  updateButtons();
}

function stop(): void {
  agent?.send(leave());
  agent?.close();
  agent = null;
  garden.me = { owner: "", joined: null, observation: null, died: null };
  keys.clear();
  dirty = true;
  updateButtons();
}

function updateButtons(): void {
  const playing = agent !== null;
  playButton.hidden = playing;
  leaveButton.hidden = !playing;
  sayInput.hidden = !playing;
  hatchButton.hidden = !(playing && garden.me.died !== null);
  nameInput.disabled = playing;
}

playButton.addEventListener("click", play);
leaveButton.addEventListener("click", stop);
hatchButton.addEventListener("click", () => {
  agent?.send(join());
  hatchButton.hidden = true;
});
nameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") play();
  event.stopPropagation();
});
sayInput.addEventListener("keydown", (event) => {
  event.stopPropagation();
  if (event.key === "Enter") {
    const text = sayInput.value.trim().slice(0, 40);
    agent?.send(say(text));
    sayInput.value = "";
  }
});
window.addEventListener("keydown", (event) => {
  if (agent === null || !keys.isGameKey(event.key)) return;
  keys.down(event.key);
  event.preventDefault();
});
window.addEventListener("keyup", (event) => keys.up(event.key));
window.addEventListener("blur", () => keys.clear());

function loop(now: number): void {
  renderer.draw(garden, now);
  if (dirty) {
    hud.update(garden);
    updateButtons();
    dirty = false;
  }
  requestAnimationFrame(loop);
}
updateButtons();
requestAnimationFrame(loop);
