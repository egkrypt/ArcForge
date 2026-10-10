import assert from 'node:assert/strict';
import {halfJumpBeats,jumpDistance,offsetForJumpDistance,mirrorNote,createBeatClock} from '../frontend/src/chromapper.ts';
assert.equal(halfJumpBeats(20,0,190),2);
assert.equal(halfJumpBeats(20,-2,190),.25);
assert.equal(halfJumpBeats(10,0,120),2);
assert.ok(Math.abs(jumpDistance(20,0,190)-25.26315789473684)<1e-9);
assert.ok(Math.abs(jumpDistance(20,offsetForJumpDistance(24,20,190),190)-24)<1e-9);
assert.throws(()=>halfJumpBeats(20,0,0));
for(let d=0;d<9;d++){
  const n={b:10,x:1,y:2,c:0,d,a:15,customData:{coordinates:[-.4,2],cutDirection:30,localRotation:[10,20,30]}};
  assert.deepEqual(mirrorNote(mirrorNote(n)),n);
  assert.deepEqual(n.customData.coordinates,[-.4,2]);
}
assert.equal(mirrorNote({x:0,c:0,d:6}).d,7);
assert.equal(mirrorNote({x:0,c:0,d:6}).x,3);
for(const x of [-2500,1500,4500])assert.equal(mirrorNote(mirrorNote({x,c:1,d:0})).x,x);
const clock=createBeatClock(120,[{b:4,m:240},{b:8,m:60}]);
assert.equal(clock.seconds(4),2);assert.equal(clock.seconds(8),3);assert.equal(clock.seconds(10),5);
assert.equal(clock.beat(5),10);
console.log('ChroMapper source adaptations and BPM integration: 24 assertions passed');
