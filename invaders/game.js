// Invaders: an 8-bit Space Invaders tribute for iPhone (works on desktop too).
// Logical screen is the arcade's 224x256, scaled up in whole device pixels.
'use strict';

const W = 224, H = 256;
const STEP = 1000 / 60;                 // fixed 60 Hz update
const PLAYER_Y = 216, BUNKER_Y = 184, GROUND_Y = 240, UFO_Y = 28;
const COLS = 11, ROWS = 5, CELL = 16;

const cv = document.getElementById('screen');
const g = cv.getContext('2d');

// ---------- storage (best effort, private mode can throw) ----------
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
};

// ---------- sprites ----------
const S = {
  squid: [[
    '...##...', '..####..', '.######.', '##.##.##', '########', '..#..#..', '.#.##.#.', '#.#..#.#',
  ], [
    '...##...', '..####..', '.######.', '##.##.##', '########', '.#.##.#.', '#......#', '.#....#.',
  ]],
  crab: [[
    '..#.....#..', '...#...#...', '..#######..', '.##.###.##.', '###########', '#.#######.#', '#.#.....#.#', '...##.##...',
  ], [
    '..#.....#..', '#..#...#..#', '#.#######.#', '###.###.###', '###########', '.#########.', '..#.....#..', '.#.......#.',
  ]],
  octo: [[
    '....####....', '.##########.', '############', '###..##..###', '############', '...##..##...', '..##.##.##..', '##........##',
  ], [
    '....####....', '.##########.', '############', '###..##..###', '############', '..###..###..', '.##..##..##.', '..##....##..',
  ]],
  player: [
    '......#......', '.....###.....', '.....###.....', '.###########.', '#############', '#############', '#############', '#############',
  ],
  playerBoom: [[
    '..#.....#......', '....#.#....#...', '.#...##.#......', '...#######..#..', '#.#########.#..', '..#########....', '.#.#########.#.', '###############',
  ], [
    '#.....#.....#..', '..#.......#....', '.....#.#.....#.', '#..#.#####.#...', '...##.###.##...', '.#.#########.#.', '..###########..', '###############',
  ]],
  ufo: [
    '.....######.....', '...##########...', '..############..', '.##.##.##.##.##.', '################', '..###..##..###..', '...#........#...',
  ],
  boom: [
    '....#...#....', '.#...#.#...#.', '..#.......#..', '...#.....#...', '##.........##', '...#.....#...', '..#.#...#.#..', '.#...#.#...#.',
  ],
  shotBoom: [
    '#...#...', '..#...#.', '.######.', '########', '.######.', '..#..#..', '#..#...#',
  ],
  zig: [['.#.', '#..', '.#.', '..#', '.#.', '#..', '.#.'], ['.#.', '..#', '.#.', '#..', '.#.', '..#', '.#.']],
  plunger: [['.#.', '.#.', '.#.', '.#.', '.#.', '###', '.#.'], ['.#.', '###', '.#.', '.#.', '.#.', '.#.', '.#.']],
};

function sprite(rows) {
  const c = document.createElement('canvas');
  c.width = rows[0].length; c.height = rows.length;
  const x = c.getContext('2d');
  x.fillStyle = '#fff';
  rows.forEach((r, y) => { for (let i = 0; i < r.length; i++) if (r[i] === '#') x.fillRect(i, y, 1, 1); });
  return c;
}
const SPR = {};
for (const k in S) SPR[k] = Array.isArray(S[k][0]) ? S[k].map(sprite) : sprite(S[k]);

const ALIEN_ROWS = ['squid', 'crab', 'crab', 'octo', 'octo'];
const ALIEN_PTS = { squid: 30, crab: 20, octo: 10 };
const ALIEN_W = { squid: 8, crab: 11, octo: 12 };

// ---------- 5x7 pixel font ----------
const FONT = {
  A: [14,17,17,31,17,17,17], B: [30,17,17,30,17,17,30], C: [14,17,16,16,16,17,14], D: [30,17,17,17,17,17,30],
  E: [31,16,16,30,16,16,31], F: [31,16,16,30,16,16,16], G: [14,17,16,23,17,17,15], H: [17,17,17,31,17,17,17],
  I: [14,4,4,4,4,4,14], J: [7,2,2,2,2,18,12], K: [17,18,20,24,20,18,17], L: [16,16,16,16,16,16,31],
  M: [17,27,21,21,17,17,17], N: [17,17,25,21,19,17,17], O: [14,17,17,17,17,17,14], P: [30,17,17,30,16,16,16],
  Q: [14,17,17,17,21,18,13], R: [30,17,17,30,20,18,17], S: [15,16,16,14,1,1,30], T: [31,4,4,4,4,4,4],
  U: [17,17,17,17,17,17,14], V: [17,17,17,17,17,10,4], W: [17,17,17,21,21,21,10], X: [17,17,10,4,10,17,17],
  Y: [17,17,17,10,4,4,4], Z: [31,1,2,4,8,16,31],
  0: [14,17,19,21,25,17,14], 1: [4,12,4,4,4,4,14], 2: [14,17,1,2,4,8,31], 3: [31,2,4,2,1,17,14],
  4: [2,6,10,18,31,2,2], 5: [31,16,30,1,1,17,14], 6: [6,8,16,30,17,17,14], 7: [31,1,2,4,8,8,8],
  8: [14,17,17,14,17,17,14], 9: [14,17,17,15,1,2,12],
  '-': [0,0,0,31,0,0,0], '=': [0,0,31,0,31,0,0], '?': [14,17,1,2,4,0,4], '!': [4,4,4,4,4,0,4],
  ':': [0,12,12,0,12,12,0], '.': [0,0,0,0,0,12,12], '<': [2,4,8,16,8,4,2], '>': [8,4,2,1,2,4,8],
  '*': [0,4,21,14,21,4,0], ' ': [0,0,0,0,0,0,0],
};
const GLYPH = {};
for (const ch in FONT) GLYPH[ch] = sprite(FONT[ch].map(n => n.toString(2).padStart(5, '0').replace(/1/g, '#')));

function text(s, x, y) {
  s = String(s).toUpperCase();
  for (let i = 0; i < s.length; i++) { const gl = GLYPH[s[i]]; if (gl) g.drawImage(gl, x + i * 6, y); }
}
const textC = (s, y) => text(s, Math.round((W - String(s).length * 6 + 1) / 2), y);
const pad = (n, l = 4) => String(n).padStart(l, '0');

// ---------- sound (WebAudio, chip-style) ----------
const sfx = (() => {
  let ac = null, master = null, noiseBuf = null, ufoNode = null;
  let on = store.get('invaders.sound', true);
  function unlock() {
    if (!ac) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      ac = new AC();
      master = ac.createGain(); master.gain.value = on ? 0.5 : 0; master.connect(ac.destination);
      noiseBuf = ac.createBuffer(1, ac.sampleRate, ac.sampleRate);
      const d = noiseBuf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      // iOS wants a sound started inside the gesture
      const b = ac.createBufferSource(); b.buffer = ac.createBuffer(1, 1, 22050); b.connect(ac.destination); b.start();
    }
    if (ac.state !== 'running') ac.resume();
  }
  function tone(type, f0, f1, dur, vol = 0.3, delay = 0) {
    if (!ac) return;
    const t = ac.currentTime + delay, o = ac.createOscillator(), e = ac.createGain();
    o.type = type; o.frequency.setValueAtTime(f0, t);
    if (f1 !== f0) o.frequency.exponentialRampToValueAtTime(f1, t + dur);
    e.gain.setValueAtTime(vol, t); e.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(e); e.connect(master); o.start(t); o.stop(t + dur + 0.02);
  }
  function noise(dur, f0, f1, vol = 0.4) {
    if (!ac) return;
    const t = ac.currentTime, s = ac.createBufferSource(), f = ac.createBiquadFilter(), e = ac.createGain();
    s.buffer = noiseBuf; s.loop = true;
    f.type = 'lowpass'; f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + dur);
    e.gain.setValueAtTime(vol, t); e.gain.exponentialRampToValueAtTime(0.001, t + dur);
    s.connect(f); f.connect(e); e.connect(master); s.start(t); s.stop(t + dur + 0.02);
  }
  const MARCH = [98, 87, 78, 73];
  return {
    unlock,
    get on() { return on; },
    toggle() { on = !on; store.set('invaders.sound', on); if (master) master.gain.value = on ? 0.5 : 0; return on; },
    march(i) { tone('square', MARCH[i & 3], MARCH[i & 3] * 0.9, 0.09, 0.35); },
    shoot() { tone('square', 1400, 300, 0.12, 0.12); noise(0.08, 6000, 1500, 0.12); },
    kill() { noise(0.2, 4000, 300, 0.35); tone('square', 600, 90, 0.16, 0.12); },
    death() { noise(0.9, 2500, 80, 0.5); tone('sawtooth', 220, 40, 0.9, 0.15); },
    shotHit() { noise(0.06, 3000, 800, 0.15); },
    ufoHit() { [880, 660, 990, 740, 1100].forEach((f, i) => tone('square', f, f, 0.07, 0.15, i * 0.07)); },
    life() { [523, 659, 784, 1047].forEach((f, i) => tone('square', f, f, 0.09, 0.15, i * 0.09)); },
    waveClear() { [392, 523, 659, 784, 659, 784].forEach((f, i) => tone('triangle', f, f, 0.1, 0.3, i * 0.1)); },
    ufoStart() {
      if (!ac || ufoNode) return;
      const o = ac.createOscillator(), lfo = ac.createOscillator(), lg = ac.createGain(), e = ac.createGain();
      o.type = 'square'; o.frequency.value = 700; lfo.frequency.value = 7; lg.gain.value = 250; e.gain.value = 0.06;
      lfo.connect(lg); lg.connect(o.frequency); o.connect(e); e.connect(master); o.start(); lfo.start();
      ufoNode = { o, lfo };
    },
    ufoStop() { if (ufoNode) { ufoNode.o.stop(); ufoNode.lfo.stop(); ufoNode = null; } },
    suspend() { if (ac) ac.suspend(); },
  };
})();

const buzz = ms => { if (navigator.vibrate) try { navigator.vibrate(ms); } catch (e) {} };

// ---------- input ----------
const held = { left: false, right: false, fire: false };
const keyHeld = { left: false, right: false, fire: false };
const touchHeld = { left: false, right: false, fire: false };
let pressedFire = false;          // edge, for menus

function syncHeld() { for (const k in held) held[k] = keyHeld[k] || touchHeld[k]; }

const KEYS = { ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right', Space: 'fire', ArrowUp: 'fire', KeyW: 'fire', Enter: 'fire' };
addEventListener('keydown', e => {
  sfx.unlock();
  if (e.code === 'KeyP' || e.code === 'Escape') { togglePause(); return; }
  if (e.code === 'KeyM') { toggleSound(); return; }
  const k = KEYS[e.code]; if (!k) return;
  e.preventDefault();
  if (k === 'fire' && !e.repeat) pressedFire = true;
  keyHeld[k] = true; syncHeld();
});
addEventListener('keyup', e => { const k = KEYS[e.code]; if (k) { keyHeld[k] = false; syncHeld(); } });

// Touch pads: each pointer maps to whichever button it is over, so a thumb can slide from left to right.
const pointers = new Map();
const btns = [...document.querySelectorAll('[data-k]')];
function btnAt(x, y) { const el = document.elementFromPoint(x, y); return el && el.closest ? el.closest('[data-k]') : null; }
function refreshTouch() {
  for (const k in touchHeld) touchHeld[k] = false;
  for (const el of pointers.values()) if (el && el.dataset.k in touchHeld) touchHeld[el.dataset.k] = true;
  btns.forEach(b => b.classList.toggle('held', [...pointers.values()].includes(b) && b.dataset.k in touchHeld));
  syncHeld();
}
addEventListener('pointerdown', e => {
  sfx.unlock();
  const el = btnAt(e.clientX, e.clientY);
  if (el && el.dataset.k === 'pause') { togglePause(); return; }
  if (el && el.dataset.k === 'sound') { toggleSound(); return; }
  if (!el || el.dataset.k === 'fire' || e.target === cv) pressedFire = true;   // tap screen to start / resume
  if (el && e.target.releasePointerCapture) try { e.target.releasePointerCapture(e.pointerId); } catch (err) {}
  pointers.set(e.pointerId, el); refreshTouch();
});
addEventListener('pointermove', e => {
  if (!pointers.has(e.pointerId)) return;
  const el = btnAt(e.clientX, e.clientY);
  if (el !== pointers.get(e.pointerId)) { pointers.set(e.pointerId, el); refreshTouch(); }
});
const lift = e => { if (pointers.delete(e.pointerId)) refreshTouch(); };
addEventListener('pointerup', lift);
addEventListener('pointercancel', lift);
['touchstart', 'touchmove', 'gesturestart', 'dblclick', 'contextmenu'].forEach(t =>
  addEventListener(t, e => e.preventDefault(), { passive: false }));

const soundBtn = document.getElementById('sound');
function toggleSound() { soundBtn.classList.toggle('off', !sfx.toggle()); }
soundBtn.classList.toggle('off', !sfx.on);

// ---------- game state ----------
let state = 'title';               // title | play | dying | clear | over
let paused = false;
let frame = 0, timer = 0;
let score = 0, hi = store.get('invaders.hi', 0), lives = 3, wave = 1, bonusGiven = false;
let player, shot, bombs, aliens, fleet, ufo, booms, popups, shotsFired, ufoTimer, bombTimer;

// Bunkers live in one W x 16 pixel mask so erosion is per pixel.
const BH = 16;
const bunkerMask = new Uint8Array(W * BH);
const bunkerCv = document.createElement('canvas'); bunkerCv.width = W; bunkerCv.height = BH;
const bunkerCtx = bunkerCv.getContext('2d');
const bunkerImg = bunkerCtx.createImageData(W, BH);
let bunkerDirty = true;

function resetBunkers() {
  bunkerMask.fill(0);
  for (let b = 0; b < 4; b++) {
    const bx = 34 + b * 45;
    for (let y = 0; y < BH; y++) for (let x = 0; x < 22; x++) {
      let on = true;
      if (y < 4 && (x < 4 - y || x > 17 + y)) on = false;               // rounded top
      if (y >= 12 && x >= 6 && x <= 15) on = false;                      // arch
      if (y === 11 && x >= 7 && x <= 14) on = false;
      if (on) bunkerMask[y * W + bx + x] = 1;
    }
  }
  bunkerDirty = true;
}
function bunkerAt(x, y) {
  x |= 0; y = (y | 0) - BUNKER_Y;
  return x >= 0 && x < W && y >= 0 && y < BH && bunkerMask[y * W + x] === 1;
}
function erode(cx, cy, r = 3, chance = 0.75) {
  cx |= 0; cy = (cy | 0) - BUNKER_Y;
  for (let y = cy - r; y <= cy + r; y++) for (let x = cx - r; x <= cx + r; x++) {
    if (x < 0 || x >= W || y < 0 || y >= BH) continue;
    const d = Math.abs(x - cx) + Math.abs(y - cy);
    if (d <= r && (d < 2 || Math.random() < chance)) bunkerMask[y * W + x] = 0;
  }
  bunkerDirty = true;
}
function eraseRect(x0, y0, w, h) {
  y0 -= BUNKER_Y;
  for (let y = Math.max(0, y0); y < Math.min(BH, y0 + h); y++)
    for (let x = Math.max(0, x0 | 0); x < Math.min(W, (x0 | 0) + w); x++) if (bunkerMask[y * W + x]) { bunkerMask[y * W + x] = 0; bunkerDirty = true; }
}

function newGame() {
  score = 0; lives = 3; wave = 1; bonusGiven = false; shotsFired = 0;
  startWave();
  state = 'play';
}
function startWave() {
  aliens = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) aliens.push({ c, r, type: ALIEN_ROWS[r], alive: true });
  const drop = Math.min(wave - 1, 5) * 8;
  fleet = { x: 24, y: 48 + drop, dir: 1, frame: 0, wait: 0, note: 0, freeze: 0 };
  resetBunkers();
  resetPlayer();
  ufo = null; ufoTimer = 25 * 60; bombTimer = 60;
  sfx.ufoStop();
}
function resetPlayer() {
  player = { x: 16, dead: 0 };
  shot = null; bombs = []; booms = []; popups = [];
}

const alive = () => aliens.filter(a => a.alive);
const alienX = a => fleet.x + a.c * CELL + ((CELL - ALIEN_W[a.type]) >> 1);
const alienY = a => fleet.y + a.r * CELL;

function addScore(n) {
  score += n;
  if (!bonusGiven && score >= 1500) { bonusGiven = true; lives++; sfx.life(); }
  if (score > hi) hi = score;
}

// classic UFO table indexed by shot count: the 23rd shot and every 15th after scores 300
const UFO_PTS = [100, 50, 50, 100, 150, 100, 100, 50, 300, 100, 100, 100, 50, 150, 100];

// ---------- update ----------
function update() {
  frame++;
  if (paused) { if (pressedFire) togglePause(); pressedFire = false; return; }

  if (state === 'title' || state === 'over') {
    timer++;
    if (pressedFire && (state === 'title' || timer > 90)) newGame();
    pressedFire = false;
    return;
  }
  pressedFire = false;

  booms = booms.filter(b => --b.t > 0);
  popups = popups.filter(p => --p.t > 0);

  if (state === 'clear') {
    if (--timer <= 0) { wave++; startWave(); state = 'play'; }
    return;
  }
  if (state === 'dying') {
    if (--timer <= 0) {
      if (lives <= 0) return gameOver();
      resetPlayer(); state = 'play';
    }
    return;
  }

  // player
  if (held.left) player.x -= 1.25;
  if (held.right) player.x += 1.25;
  player.x = Math.max(8, Math.min(W - 8 - 13, player.x));
  if (held.fire && !shot) {
    shot = { x: Math.round(player.x) + 6, y: PLAYER_Y - 4 };
    shotsFired++; sfx.shoot();
  }

  updateShot();
  updateFleet();
  updateBombs();
  updateUfo();

  if (state === 'play' && alive().length === 0) {
    state = 'clear'; timer = 120; bombs = []; shot = null; sfx.ufoStop(); ufo = null; sfx.waveClear();
  }
}

function updateShot() {
  if (!shot) return;
  for (let i = 0; i < 4 && shot; i++) {       // 4 px per frame, tested per pixel
    shot.y--;
    const x = shot.x, y = shot.y;
    if (y < 24) { booms.push({ spr: SPR.shotBoom, x: x - 4, y: 24, t: 14 }); shot = null; return; }
    if (bunkerAt(x, y)) { erode(x, y, 3, 0.6); shot = null; return; }
    for (const b of bombs) if (x >= b.x && x < b.x + 3 && y >= b.y && y < b.y + 7) {
      booms.push({ spr: SPR.shotBoom, x: x - 4, y: y - 3, t: 10 }); sfx.shotHit();
      bombs.splice(bombs.indexOf(b), 1); shot = null; return;
    }
    if (ufo && x >= ufo.x && x < ufo.x + 16 && y >= UFO_Y && y < UFO_Y + 7) {
      const pts = UFO_PTS[shotsFired % 15];
      addScore(pts); popups.push({ txt: String(pts), x: Math.round(ufo.x), y: UFO_Y, t: 70 });
      sfx.ufoStop(); sfx.ufoHit(); buzz(30); ufo = null; shot = null; return;
    }
    for (const a of aliens) {
      if (!a.alive) continue;
      const ax = alienX(a), ay = alienY(a);
      if (x >= ax && x < ax + ALIEN_W[a.type] && y >= ay && y < ay + 8) {
        a.alive = false; addScore(ALIEN_PTS[a.type]);
        booms.push({ spr: SPR.boom, x: ax + ((ALIEN_W[a.type] - 13) >> 1), y: ay, t: 16 });
        fleet.freeze = 10; sfx.kill(); buzz(10); shot = null; return;
      }
    }
  }
}

function updateFleet() {
  if (fleet.freeze > 0) { fleet.freeze--; return; }
  const live = alive();
  if (!live.length) return;
  if (--fleet.wait > 0) return;

  // speed rises as the fleet thins and with each wave
  const factor = Math.max(0.55, 1 - 0.07 * (wave - 1));
  const interval = Math.max(1, Math.round(live.length * 0.9 * factor));
  fleet.wait = interval;

  let minX = W, maxX = 0, maxY = 0;
  for (const a of live) {
    const ax = alienX(a);
    minX = Math.min(minX, ax); maxX = Math.max(maxX, ax + ALIEN_W[a.type]); maxY = Math.max(maxY, alienY(a) + 8);
  }
  const dx = fleet.dir * (live.length === 1 && fleet.dir > 0 ? 3 : 2);
  if (minX + dx < 4 || maxX + dx > W - 4) { fleet.y += 8; fleet.dir = -fleet.dir; }
  else fleet.x += dx;
  fleet.frame ^= 1;

  // march beat, throttled so the fast endgame does not turn into a buzz
  fleet.beat = (fleet.beat || 0) + interval;
  if (fleet.beat >= 6) { fleet.beat = 0; sfx.march(fleet.note++); }

  for (const a of live) {
    const ay = alienY(a);
    if (ay + 8 > BUNKER_Y) eraseRect(alienX(a), ay, ALIEN_W[a.type], 8);
    if (ay + 8 >= PLAYER_Y) { invaded(); return; }
  }
}

function updateBombs() {
  const live = alive();
  const speed = 1.5 + Math.min(1.5, (wave - 1) * 0.15) + (live.length < 8 ? 0.5 : 0);
  const maxBombs = Math.min(3, 1 + Math.floor((wave + 1) / 2));
  if (--bombTimer <= 0 && live.length && bombs.length < maxBombs) {
    bombTimer = Math.max(18, 70 - wave * 5 - (55 - live.length)) + (Math.random() * 30 | 0);
    // half the time drop on the player, otherwise a random column
    const px = player.x + 6;
    let col;
    if (Math.random() < 0.5) {
      col = live.reduce((best, a) => Math.abs(alienX(a) + 5 - px) < Math.abs(alienX(best) + 5 - px) ? a : best).c;
    } else col = live[Math.random() * live.length | 0].c;
    const shooter = live.filter(a => a.c === col).reduce((b, a) => a.r > b.r ? a : b);
    bombs.push({ x: alienX(shooter) + (ALIEN_W[shooter.type] >> 1) - 1, y: alienY(shooter) + 8, kind: Math.random() < 0.5 ? 'zig' : 'plunger', fy: 0 });
  }
  for (let i = bombs.length - 1; i >= 0; i--) {
    const b = bombs[i];
    b.fy += speed;
    let hit = false;
    while (b.fy >= 1 && !hit) {
      b.fy--; b.y++;
      const tipY = b.y + 7, cx = b.x + 1;
      if (bunkerAt(cx, tipY) || bunkerAt(b.x, tipY) || bunkerAt(b.x + 2, tipY)) { erode(cx, tipY + 1, 3, 0.7); hit = true; }
      else if (tipY >= GROUND_Y) { booms.push({ spr: SPR.shotBoom, x: b.x - 2, y: GROUND_Y - 7, t: 12 }); hit = true; }
      else if (state === 'play' && tipY >= PLAYER_Y && tipY < PLAYER_Y + 8 && cx >= player.x && cx < player.x + 13) { hit = true; killPlayer(); }
    }
    if (hit) { bombs.splice(i, 1); if (state !== 'play') return; }
  }
}

function updateUfo() {
  if (ufo) {
    ufo.x += ufo.dir * 0.75;
    if (ufo.x < -16 || ufo.x > W) { ufo = null; sfx.ufoStop(); }
    return;
  }
  if (--ufoTimer <= 0) {
    ufoTimer = (20 + Math.random() * 10) * 60;
    if (alive().length >= 8) {
      const dir = shotsFired % 2 ? -1 : 1;
      ufo = { x: dir > 0 ? -16 : W, dir };
      sfx.ufoStart();
    }
  }
}

function killPlayer() {
  lives--; state = 'dying'; timer = 130; player.dead = 1;
  bombs.length = 0; shot = null; sfx.ufoStop(); ufo = null; sfx.death(); buzz([60, 40, 120]);
}
function invaded() { lives = 0; killPlayer(); }
function gameOver() {
  state = 'over'; timer = 0; bombs = [];
  store.set('invaders.hi', hi);
}

function togglePause() {
  if (state === 'title' || state === 'over') return;
  paused = !paused;
  if (paused) { sfx.ufoStop(); } else { sfx.unlock(); if (ufo) sfx.ufoStart(); }
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { if (!paused && (state === 'play' || state === 'dying' || state === 'clear')) togglePause(); sfx.suspend(); store.set('invaders.hi', hi); }
});

// ---------- draw ----------
function draw() {
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);

  // HUD
  text('SCORE<1>', 8, 4); text('HI-SCORE', W - 8 - 47, 4);
  text(pad(score), 14, 14); text(pad(hi), W - 8 - 41, 14);

  if (state === 'title') drawTitle();
  else drawPlayfield();

  // the arcade's colored cellophane strips
  g.globalCompositeOperation = 'multiply';
  g.fillStyle = '#ff3c3c'; g.fillRect(0, 24, W, 16);
  g.fillStyle = '#3cff5a'; g.fillRect(0, 176, W, H - 176);
  g.globalCompositeOperation = 'source-over';
}

function drawTitle() {
  const blink = (frame >> 5) & 1;
  textC('PLAY', 56);
  textC('SPACE  INVADERS', 76);
  textC('*SCORE ADVANCE TABLE*', 104);
  const f = (frame >> 5) & 1;
  const rows = [[SPR.ufo, 16, '=? MYSTERY'], [SPR.squid[f], 8, '=30 POINTS'], [SPR.crab[f], 11, '=20 POINTS'], [SPR.octo[f], 12, '=10 POINTS']];
  rows.forEach(([spr, w, label], i) => { const y = 118 + i * 14; g.drawImage(spr, 72 - (w >> 1), y); text(label, 86, y); });
  if (blink) textC('TAP FIRE TO START', 196);
  textC('< > MOVE    FIRE SHOOT', 214);
  drawGround();
}

function drawGround() {
  g.fillStyle = '#fff'; g.fillRect(0, GROUND_Y, W, 1);
}

function drawPlayfield() {
  // bunkers
  if (bunkerDirty) {
    const d = bunkerImg.data;
    for (let i = 0; i < bunkerMask.length; i++) { const v = bunkerMask[i] ? 255 : 0; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 255; d[i * 4 + 3] = v; }
    bunkerCtx.putImageData(bunkerImg, 0, 0); bunkerDirty = false;
  }
  g.drawImage(bunkerCv, 0, BUNKER_Y);

  for (const a of aliens) if (a.alive) g.drawImage(SPR[a.type][fleet.frame], alienX(a), alienY(a));
  if (ufo) g.drawImage(SPR.ufo, Math.round(ufo.x), UFO_Y);
  for (const b of booms) g.drawImage(b.spr, b.x, b.y);
  for (const p of popups) text(p.txt, p.x, p.y);

  if (state === 'dying') g.drawImage(SPR.playerBoom[(frame >> 3) & 1], Math.round(player.x) - 1, PLAYER_Y);
  else g.drawImage(SPR.player, Math.round(player.x), PLAYER_Y);

  g.fillStyle = '#fff';
  if (shot) g.fillRect(shot.x, shot.y, 1, 4);
  for (const b of bombs) g.drawImage(SPR[b.kind][(frame >> 2) & 1], b.x, b.y);

  drawGround();
  text(Math.max(0, lives), 8, 244);
  for (let i = 0; i < Math.min(lives - 1, 5); i++) g.drawImage(SPR.player, 22 + i * 16, 243);
  text('WAVE ' + wave, W - 8 - 41 + (wave > 9 ? -6 : 0), 244);

  if (state === 'clear' && (frame >> 4) & 1) textC('WAVE ' + wave + ' CLEARED', 120);
  if (state === 'over') {
    g.fillStyle = '#000'; g.fillRect(40, 100, W - 80, 52);
    textC('GAME OVER', 110);
    if (timer > 90 && (frame >> 5) & 1) textC('TAP TO PLAY AGAIN', 132);
  }
  if (paused) {
    g.fillStyle = '#000'; g.fillRect(40, 100, W - 80, 52);
    textC('PAUSED', 110);
    if ((frame >> 5) & 1) textC('TAP TO RESUME', 132);
  }
}

// ---------- sizing: whole device pixels per logical pixel when possible ----------
const box = document.getElementById('screenbox'), crt = document.getElementById('crt');
function fit() {
  const dpr = window.devicePixelRatio || 1;
  const bw = box.clientWidth, bh = box.clientHeight;
  let s = Math.floor(Math.min(bw * dpr / W, bh * dpr / H));
  s = s >= 1 ? s / dpr : Math.min(bw / W, bh / H);
  crt.style.width = W * s + 'px'; crt.style.height = H * s + 'px';
  crt.style.setProperty('--px', s + 'px');
}
new ResizeObserver(fit).observe(box);
addEventListener('orientationchange', () => setTimeout(fit, 200));
fit();

// ---------- loop ----------
let last = performance.now(), acc = 0;
function loop(now) {
  acc += Math.min(100, now - last); last = now;
  while (acc >= STEP) { update(); acc -= STEP; }
  draw();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
