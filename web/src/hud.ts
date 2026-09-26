// The DOM around the canvas: status, roster, log, scores, and the play panel.
import { type Garden, livingFlies, myFly } from "./state";

const MOOD_GLYPH: Record<string, string> = {
  content: "✨",
  hungry: "🍎",
  thirsty: "💧",
  sleepy: "💤",
  desperate: "❗",
  dying: "☠️",
  dead: "✝",
};
const NEEDS = ["satiety", "hydration", "energy", "health"] as const;

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const NEED_LABEL: Record<(typeof NEEDS)[number], string> = {
  satiety: "S",
  hydration: "W",
  energy: "E",
  health: "♥",
};

function bar(need: (typeof NEEDS)[number], value: number): HTMLElement {
  const wrap = el("div", "bar");
  wrap.appendChild(el("span", "bar-label", NEED_LABEL[need]));
  const track = el("div", "bar-track");
  const fill = el("div", `bar-fill bar-${need}`);
  fill.style.width = `${Math.max(0, Math.min(100, value / 10))}%`;
  track.appendChild(fill);
  wrap.appendChild(track);
  return wrap;
}

export function timeOfDay(tick: number, dayLength: number): string {
  const phase = tick % dayLength;
  if (phase < dayLength / 12) return "dawn";
  if (phase < (dayLength * 7) / 12) return "day";
  if (phase < (dayLength * 8) / 12) return "dusk";
  return "night";
}

export class Hud {
  readonly status: HTMLElement;
  readonly roster: HTMLElement;
  readonly log: HTMLElement;
  readonly scores: HTMLElement;
  readonly mine: HTMLElement;
  readonly notice: HTMLElement;

  constructor(root: HTMLElement) {
    this.status = root.querySelector("#status")!;
    this.roster = root.querySelector("#roster")!;
    this.log = root.querySelector("#log")!;
    this.scores = root.querySelector("#scores")!;
    this.mine = root.querySelector("#mine")!;
    this.notice = root.querySelector("#notice")!;
  }

  update(garden: Garden): void {
    const frame = garden.frame;
    const dayLength = garden.welcome?.world.day_length ?? 1200;
    if (frame === null) {
      this.status.textContent = garden.world ? "waiting for the first tick…" : "connecting…";
    } else {
      const sky = frame.light < 500 ? "🌙" : "☀️";
      this.status.textContent = `Day ${frame.day} · ${timeOfDay(frame.tick, dayLength)} ${sky} · tick ${frame.tick}`;
    }

    this.roster.replaceChildren();
    if (frame !== null) {
      const mineId = garden.me.joined?.agent_id ?? -1;
      for (const a of livingFlies(frame).sort((p, q) => p.owner.localeCompare(q.owner))) {
        const row = el("div", `fly-row${a.agent_id === mineId ? " mine" : ""}${a.connected ? "" : " away"}`);
        const head = el("div", "fly-head");
        head.appendChild(el("span", "fly-name", a.name));
        head.appendChild(el("span", "fly-owner", ` ${a.owner} #${a.lineage}`));
        head.appendChild(el("span", "fly-mood", ` ${MOOD_GLYPH[a.mood] ?? "?"}${a.connected ? "" : " (away)"}`));
        row.appendChild(head);
        const bars = el("div", "bars");
        for (const need of NEEDS) bars.appendChild(bar(need, a[need]));
        row.appendChild(bars);
        if (a.say) row.appendChild(el("div", "fly-say", `“${a.say}”`));
        this.roster.appendChild(row);
      }
      if (livingFlies(frame).length === 0) this.roster.appendChild(el("div", "muted", "the garden is empty — hatch a fly"));
    }

    this.log.replaceChildren(...garden.chronicle.map((line) => el("div", "log-line", line)));

    this.scores.replaceChildren();
    if (frame !== null) {
      for (const s of frame.scores.slice(0, 8)) {
        const row = el("div", "score-row");
        row.appendChild(el("span", "score-owner", s.owner));
        row.appendChild(el("span", "score-best", `${s.best_lifespan} ticks · ${s.lives} ${s.lives === 1 ? "life" : "lives"}${s.alive ? " · alive" : ""}`));
        this.scores.appendChild(row);
      }
    }

    this.notice.textContent = garden.notice ?? "";
    this.notice.hidden = garden.notice === null;

    this.mine.replaceChildren();
    const me = garden.me;
    if (me.died !== null) {
      const d = me.died;
      const card = el("div", "obituary");
      card.appendChild(el("h3", undefined, `${d.name} (#${d.lineage}) has died`));
      card.appendChild(el("p", undefined, `of ${d.causes.join(", ") || "unknown causes"} after ${d.stats.lifespan} ticks (${d.stats.days} days)`));
      card.appendChild(el("p", "muted", `${d.stats.bites} bites · ${d.stats.drinks} drinks · ${d.stats.tiles_explored} tiles explored`));
      this.mine.appendChild(card);
      return;
    }
    const fly = myFly(garden);
    if (me.joined !== null && fly !== null) {
      const card = el("div", "me-card");
      card.appendChild(el("h3", undefined, `${fly.name} · #${fly.lineage} · ${MOOD_GLYPH[fly.mood] ?? ""}`));
      const bars = el("div", "bars");
      for (const need of NEEDS) bars.appendChild(bar(need, fly[need]));
      card.appendChild(bars);
      const obs = me.observation;
      if (obs !== null) card.appendChild(el("p", "muted", `tick ${obs.tick} · missed ${obs.missed} · deadline ${obs.deadline_ms} ms`));
      card.appendChild(el("p", "muted", "arrows move · space eats or drinks · R rests"));
      this.mine.appendChild(card);
    } else if (me.joined !== null) {
      this.mine.appendChild(el("p", "muted", "hatching…"));
    }
  }
}
