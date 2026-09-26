// Keys -> the catalog's action ids. A held key repeats every tick; a tap counts for one tick.

const KEY_TO_ACTION: Record<string, string> = {
  ArrowUp: "move_n",
  ArrowRight: "move_e",
  ArrowDown: "move_s",
  ArrowLeft: "move_w",
  w: "move_n",
  d: "move_e",
  s: "move_s",
  a: "move_w",
  " ": "consume",
  e: "consume",
  r: "rest",
};

export class Keys {
  private held: string[] = [];
  private tapped: string | null = null;

  down(key: string): void {
    if (!(key in KEY_TO_ACTION)) return;
    this.held = this.held.filter((k) => k !== key);
    this.held.push(key);
    this.tapped = key;
  }

  up(key: string): void {
    this.held = this.held.filter((k) => k !== key);
  }

  clear(): void {
    this.held = [];
    this.tapped = null;
  }

  /**
   * The action for the next tick: the most recently pressed key still held, else a key tapped
   * since the last tick (ticks are 200 ms apart; a quick tap must not vanish), else idle.
   * Taps are consumed; held keys repeat.
   */
  nextAction(): string {
    const held = this.held[this.held.length - 1];
    const key = held ?? this.tapped;
    this.tapped = null;
    return key === undefined || key === null ? "idle" : (KEY_TO_ACTION[key] ?? "idle");
  }

  isGameKey(key: string): boolean {
    return key in KEY_TO_ACTION;
  }
}

/** The action id the catalog assigns to a name; unknown names are idle (0). */
export function actionId(actions: readonly string[], name: string): number {
  const index = actions.indexOf(name);
  return index < 0 ? 0 : index;
}
