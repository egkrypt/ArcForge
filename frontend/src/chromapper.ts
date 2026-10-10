/* SPDX-License-Identifier: GPL-2.0-only
 * Source adaptations from ChroMapper 0.13.892, Caeden117 and contributors.
 * See integrations/PROVENANCE.md for original files and exact source archive.
 * SpawnParameterHelper.cs and MirrorSelection.cs algorithms ported to TypeScript.
 */

export function halfJumpBeats(njs: number, offset: number, bpm: number): number {
  if (!(njs > 0) || !(bpm > 0) || !Number.isFinite(offset)) throw new Error('Invalid jump settings');
  let half = 4;
  while (njs * (60 / bpm) * half > 17.999) half /= 2;
  return Math.max(0.25, half + offset);
}

export function jumpDistance(njs: number, offset: number, bpm: number): number {
  return halfJumpBeats(njs, offset, bpm) * (60 / bpm) * njs * 2;
}

export function offsetForJumpDistance(distance: number, njs: number, bpm: number): number {
  if (!(distance > 0) || !(njs > 0) || !(bpm > 0)) throw new Error('Invalid jump distance');
  return Math.max(0.25, distance / (njs * (60 / bpm) * 2)) - halfJumpBeats(njs, 0, bpm);
}

const mirroredDirections: Record<number, number> = {2:3, 3:2, 4:5, 5:4, 6:7, 7:6};
const mirroredRotation = (rotation: any): any => typeof rotation === 'number' ? -rotation :
  Array.isArray(rotation) ? rotation.map((v, i) => i === 1 || i === 2 ? -v : v) : rotation;

export function mirrorNote(original: any): any {
  const n = structuredClone(original);
  n.a = -(n.a || 0);
  if (n.x < 0 || n.x > 3) {
    let normalized = n.x;
    if (normalized <= -1000) normalized += 1000;
    else if (normalized >= 1000) normalized -= 1000;
    normalized = 3000 - normalized;
    n.x = normalized < 0 ? normalized - 1000 : normalized + 1000;
  } else n.x = 3 - n.x;
  if (n.c === 0 || n.c === 1) n.c = 1 - n.c;
  n.d = mirroredDirections[n.d] ?? n.d;
  if (n.customData) {
    const cd = n.customData;
    for (const key of ['coordinates', '_position']) {
      if (Array.isArray(cd[key])) cd[key][0] = -cd[key][0] - 1;
    }
    for (const key of ['cutDirection', '_cutDirection']) {
      if (typeof cd[key] === 'number') cd[key] *= -1;
    }
    for (const key of ['localRotation', '_localRotation', 'worldRotation', '_rotation']) {
      if (cd[key] !== undefined) cd[key] = mirroredRotation(cd[key]);
    }
  }
  return n;
}

// Independent utility, not a ChroMapper source adaptation: integrate v3 BPM events.
export function createBeatClock(baseBpm: number, bpmEvents: any[] = []) {
  const events = [{b:0, m:baseBpm}, ...bpmEvents.filter(e => Number.isFinite(e.b) && e.m > 0)]
    .sort((a,b) => a.b - b.b);
  const segments: {b:number,m:number,t:number}[] = [];
  for (const e of events) {
    const prev = segments[segments.length - 1];
    if (prev && prev.b === e.b) { prev.m = e.m; continue; }
    segments.push({b:e.b, m:e.m, t:prev ? prev.t + (e.b - prev.b) * 60 / prev.m : 0});
  }
  return {
    seconds(beat: number) {
      let s = segments[0];
      for (const x of segments) { if (x.b > beat) break; s = x; }
      return s.t + (beat - s.b) * 60 / s.m;
    },
    beat(seconds: number) {
      let s = segments[0];
      for (const x of segments) { if (x.t > seconds) break; s = x; }
      return s.b + (seconds - s.t) * s.m / 60;
    }
  };
}
