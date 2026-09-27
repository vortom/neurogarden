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

/**
 * Shift and CapsLock change the character a key reports: `keydown` may say "W" where
 * `keyup` says "w". Fold every single-character key to lower case so a key can never
 * stick down, and so WASD keeps working with CapsLock on. Named keys ("ArrowUp") pass
 * through untouched.
 */
function normalise(key: string): string {
  return key.length === 1 ? key.toLowerCase() : key;
}

export class Keys {
  /** Key -> the press that put it down, so a later tap can beat an older hold. */
  private readonly held = new Map<string, number>();
  private tapped: { key: string; press: number } | null = null;
  private presses = 0;

  down(key: string): void {
    key = normalise(key);
    if (!(key in KEY_TO_ACTION) || this.held.has(key)) return; // auto-repeat is not a new press
    this.presses += 1;
    this.held.set(key, this.presses);
    this.tapped = { key, press: this.presses };
  }

  up(key: string): void {
    this.held.delete(normalise(key));
  }

  clear(): void {
    this.held.clear();
    this.tapped = null;
  }

  /**
   * The action for the next tick: the key pressed most recently, whether it is still held
   * (a hold repeats) or was tapped and released since the last tick (ticks are 200 ms apart;
   * a quick tap must not vanish). Taps are consumed; held keys stay.
   */
  nextAction(): string {
    const tap = this.tapped;
    this.tapped = null;
    let key: string | null = null;
    let press = 0;
    for (const [candidate, at] of this.held) {
      if (at >= press) {
        press = at;
        key = candidate;
      }
    }
    if (tap !== null && tap.press > press) key = tap.key;
    return key === null ? "idle" : (KEY_TO_ACTION[key] ?? "idle");
  }

  isGameKey(key: string): boolean {
    return normalise(key) in KEY_TO_ACTION;
  }
}

/** The action id the catalog assigns to a name; unknown names are idle (0). */
export function actionId(actions: readonly string[], name: string): number {
  const index = actions.indexOf(name);
  return index < 0 ? 0 : index;
}
