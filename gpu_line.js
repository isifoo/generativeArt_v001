// V1 LINE on the GPU (WebGPU). Port of engine/line_engine.py.
// Whole pipeline stays on the GPU: video -> CLAHE darkness -> spacing field R -> differential growth -> draw.
const LAND = new URLSearchParams(location.search).has('land');   // V3 100$ runs in its own landscape page mode
const SW = LAND ? 960 : 540, SH = LAND ? 540 : 960, OW = 1080, OH = 1920, SCALE = 2, MAXN = 262144, MARGIN = 6;
const CS = 4.16, GW = Math.floor(SW / CS) + 1, GH = Math.floor(SH / CS) + 1;
const NGW = 70, NGH = 121;                 // low-res noise grid (8 sim px per cell)
const WARM_STEPS = 150;                    // V1 stops by itself at ~125
const FPS = 30;
const $ = (s) => document.querySelector(s);
const msg = (m) => { $('#msg').textContent = m; };

if (!navigator.gpu) { msg('WebGPU is not available in this browser.'); throw new Error('no webgpu'); }
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) { msg('No GPU adapter.'); throw new Error('no adapter'); }
const hasTS = adapter.features.has('timestamp-query');
const device = await adapter.requestDevice({
  requiredFeatures: hasTS ? ['timestamp-query'] : [],
  requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage) },
});
device.addEventListener('uncapturederror', (e) => msg('GPU error: ' + e.error.message));
device.lost.then((i) => msg('GPU device lost: ' + i.message));
const adInfo = adapter.info || {};
const GPU_NAME = [adInfo.vendor, adInfo.architecture, adInfo.description].filter(Boolean).join(' ') || 'unknown GPU';

// ---------------------------------------------------------------- WGSL
const COMMON = /* wgsl */`
struct U { time:f32, rmin:f32, rmax:f32, gain:f32, noise:f32, fullWake:f32, tension:f32, kr:f32,
           key:f32, newW:f32, clip:f32, p0:f32, p1:f32, p2:f32, p3:f32, p4:f32 };
const SW:u32 = ${SW}u; const SH:u32 = ${SH}u; const MAXN:u32 = ${MAXN}u;
const GW:i32 = ${GW}; const GH:i32 = ${GH}; const CS:f32 = ${CS}; const MARGIN:f32 = ${MARGIN}.0;
fn pcg(v:u32)->u32 { let s = v*747796405u + 2891336289u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn rnd(i:u32, s:u32)->f32 { return f32(pcg(i ^ pcg(s*2654435769u + 1u))) * (1.0/4294967296.0); }
fn rndn(i:u32, s:u32)->f32 { return (rnd(i,s) + rnd(i,s+101u) + rnd(i,s+202u) + rnd(i,s+303u) - 2.0) * 1.7320508; }
fn pidx(p:vec2f)->u32 { return u32(clamp(i32(p.y),0,i32(SH)-1))*SW + u32(clamp(i32(p.x),0,i32(SW)-1)); }
`;
const K = {};   // kernels

const WGSL = {
prep: /* wgsl */`
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> gray: array<f32>;
@group(0) @binding(3) var<storage, read_write> keyr: array<f32>;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) g:vec3u) {
  if (g.x >= SW || g.y >= SH) { return; }
  let c = textureLoad(src, vec2i(g.xy), 0).rgb;
  let i = g.y*SW + g.x;
  gray[i] = floor((0.299*c.r + 0.587*c.g + 0.114*c.b)*255.0 + 0.5);
  let mx = max(c.r, max(c.g, c.b)); let mn = min(c.r, min(c.g, c.b)); let d = mx - mn;
  var h = 0.0;
  if (d > 1e-6) {
    if (mx == c.r) { h = (c.g - c.b)/d; } else if (mx == c.g) { h = 2.0 + (c.b - c.r)/d; } else { h = 4.0 + (c.r - c.g)/d; }
    h = h*60.0; if (h < 0.0) { h += 360.0; }
  }
  let s = select(0.0, d/mx*255.0, mx > 1e-6);
  let hv = h*0.5;
  keyr[i] = select(0.0, 1.0, ((hv < 8.0) || (hv > 165.0)) && s > 130.0 && u.key > 0.5);
}`,
blur: /* wgsl */`
struct B { sigma:f32, horiz:f32, rad:f32, pre:f32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> b:B;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) g:vec3u) {
  if (g.x >= SW || g.y >= SH) { return; }
  var acc = 0.0; var ws = 0.0; let r = i32(b.rad);
  for (var k = -r; k <= r; k++) {
    let w = exp(-f32(k*k)/(2.0*b.sigma*b.sigma));
    var x = i32(g.x); var y = i32(g.y);
    if (b.horiz > 0.5) { x = clamp(x + k, 0, i32(SW)-1); } else { y = clamp(y + k, 0, i32(SH)-1); }
    var v = src[u32(y)*SW + u32(x)];
    if (b.pre > 0.5) { v = smoothstep(0.3, 0.7, v); }
    acc += w*v; ws += w;
  }
  dst[g.y*SW + g.x] = acc/ws;
}`,
dilate: /* wgsl */`
struct B { sigma:f32, horiz:f32, rad:f32, pre:f32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> b:B;
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) g:vec3u) {
  if (g.x >= SW || g.y >= SH) { return; }
  var m = 0.0; let r = i32(b.rad);
  for (var k = -r; k <= r; k++) {
    var x = i32(g.x); var y = i32(g.y);
    if (b.horiz > 0.5) { x = clamp(x + k, 0, i32(SW)-1); } else { y = clamp(y + k, 0, i32(SH)-1); }
    m = max(m, src[u32(y)*SW + u32(x)]);
  }
  dst[g.y*SW + g.x] = m;
}`,
tilehist: /* wgsl */`
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> lut: array<f32>;
var<workgroup> hist: array<atomic<u32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg:vec3u, @builtin(local_invocation_index) t:u32) {
  atomicStore(&hist[t], 0u);
  workgroupBarrier();
  let tw = SW/6u; let th = SH/6u;
  let x0 = wg.x*tw; let y0 = wg.y*th;
  for (var k = t; k < tw*th; k += 256u) {
    let x = x0 + k % tw; let y = y0 + k / tw;
    atomicAdd(&hist[u32(gray[y*SW + x])], 1u);
  }
  workgroupBarrier();
  if (t == 0u) {
    let tot = f32(tw*th);
    let limit = u32(max(u.clip*tot/256.0, 1.0));
    var excess = 0u;
    for (var i = 0u; i < 256u; i++) { let v = atomicLoad(&hist[i]); if (v > limit) { excess += v - limit; atomicStore(&hist[i], limit); } }
    let batch = excess/256u; var resid = excess - batch*256u;
    for (var i = 0u; i < 256u; i++) { atomicAdd(&hist[i], batch); }
    if (resid > 0u) {
      let step = max(256u/resid, 1u); var i = 0u;
      loop { if (i >= 256u || resid == 0u) { break; } atomicAdd(&hist[i], 1u); resid--; i += step; }
    }
    var sum = 0u; let sc = 255.0/tot; let base = (wg.y*6u + wg.x)*256u;
    for (var i = 0u; i < 256u; i++) { sum += atomicLoad(&hist[i]); lut[base + i] = clamp(floor(f32(sum)*sc + 0.5), 0.0, 255.0); }
  }
}`,
claheapply: /* wgsl */`
@group(0) @binding(0) var<storage, read> gray: array<f32>;
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<storage, read> keyc: array<f32>;
@group(0) @binding(3) var<storage, read_write> cl: array<f32>;
@group(0) @binding(4) var<storage, read_write> hist2: array<atomic<u32>>;
fn L(tx:i32, ty:i32, v:u32)->f32 { return lut[(u32(ty)*6u + u32(tx))*256u + v]; }
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) g:vec3u) {
  if (g.x >= SW || g.y >= SH) { return; }
  let i = g.y*SW + g.x; let v = u32(gray[i]);
  let txf = f32(g.x)/f32(SW/6u) - 0.5; let tyf = f32(g.y)/f32(SH/6u) - 0.5;
  var tx1 = i32(floor(txf)); var tx2 = tx1 + 1; let xa = txf - f32(tx1);
  var ty1 = i32(floor(tyf)); var ty2 = ty1 + 1; let ya = tyf - f32(ty1);
  tx1 = max(tx1, 0); tx2 = min(tx2, 5); ty1 = max(ty1, 0); ty2 = min(ty2, 5);
  let r = (L(tx1,ty1,v)*(1.0-xa) + L(tx2,ty1,v)*xa)*(1.0-ya) + (L(tx1,ty2,v)*(1.0-xa) + L(tx2,ty2,v)*xa)*ya;
  let q = floor(r + 0.5);
  cl[i] = q;
  if (keyc[i] < 0.5) { atomicAdd(&hist2[u32(clamp(q, 0.0, 255.0))], 1u); }
}`,
percentile: /* wgsl */`
@group(0) @binding(0) var<storage, read> hist2: array<u32>;
@group(0) @binding(1) var<storage, read_write> stretch: array<f32>;
@compute @workgroup_size(1)
fn main() {
  var total = 0u;
  for (var i = 0u; i < 256u; i++) { total += hist2[i]; }
  if (total == 0u) { stretch[0] = 0.0; stretch[1] = 1.0; return; }
  let rlo = 0.02*f32(total - 1u); let rhi = 0.97*f32(total - 1u);
  var cum = 0u; var lo = 0.0; var hi = 255.0; var gotLo = false;
  for (var i = 0u; i < 256u; i++) {
    cum += hist2[i];
    if (!gotLo && f32(cum) > rlo) { lo = f32(i); gotLo = true; }
    if (f32(cum) > rhi) { hi = f32(i); break; }
  }
  stretch[0] = lo/255.0; stretch[1] = hi/255.0;
}`,
darkprep: /* wgsl */`
@group(0) @binding(0) var<storage, read> cl: array<f32>;
@group(0) @binding(1) var<storage, read> stretch: array<f32>;
@group(0) @binding(2) var<storage, read> keyc: array<f32>;
@group(0) @binding(3) var<storage, read_write> dg: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) {
  let i = g.x; if (i >= SW*SH) { return; }
  let lo = stretch[0]; let hi = stretch[1];
  let v = pow(clamp((cl[i]/255.0 - lo)/max(hi - lo, 1e-3), 0.0, 1.0), 1.25);
  let k = keyc[i];
  dg[i] = v*(1.0 - k) + k;
}`,
rfield: /* wgsl */`
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var<storage, read> dgb: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<storage, read_write> Rb: array<f32>;
@group(0) @binding(4) var<storage, read_write> wake: array<f32>;
@group(0) @binding(5) var<storage, read> na: array<f32>;
@group(0) @binding(6) var<storage, read> nb: array<f32>;
fn bil(which:u32, x:f32, y:f32)->f32 {
  let gx = clamp(x/8.0, 0.0, ${NGW - 1}.0 - 0.001); let gy = clamp(y/8.0, 0.0, ${NGH - 1}.0 - 0.001);
  let x0 = u32(gx); let y0 = u32(gy); let fx = gx - f32(x0); let fy = gy - f32(y0);
  let w = ${NGW}u;
  var a00 = na[y0*w + x0]; var a10 = na[y0*w + x0 + 1u]; var a01 = na[(y0+1u)*w + x0]; var a11 = na[(y0+1u)*w + x0 + 1u];
  if (which == 1u) { a00 = nb[y0*w + x0]; a10 = nb[y0*w + x0 + 1u]; a01 = nb[(y0+1u)*w + x0]; a11 = nb[(y0+1u)*w + x0 + 1u]; }
  return mix(mix(a00, a10, fx), mix(a01, a11, fx), fy);
}
@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) g:vec3u) {
  if (g.x >= SW || g.y >= SH) { return; }
  let i = g.y*SW + g.x;
  let d = 1.0 - dgb[i];
  let s = state[i]*(1.0 - u.newW) + d*u.newW;
  state[i] = s;
  let dd = clamp(s*u.gain - 0.08, 0.0, 1.0);
  var R = u.rmax*pow(u.rmin/u.rmax, sqrt(dd));
  let fx = f32(g.x); let fy = f32(g.y);
  R = R*(1.0 + u.noise*(bil(0u, fx, fy)*cos(u.time*0.6) + bil(1u, fx, fy)*sin(u.time*0.6)));
  R = clamp(R, u.rmin*0.9, u.rmax*1.3);
  let old = Rb[i]; Rb[i] = R;
  wake[i] = select(0.0, 1.0, u.fullWake > 0.5 || abs(R - old)/R > 0.03);
}`,
keymix: /* wgsl */`
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> keyc: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) {
  let i = g.x; if (i >= SW*SH) { return; }
  let m = smoothstep(0.35, 0.65, mask[i]);
  var k = keyc[i];
  if (u.p0 > 0.5 && u.p0 < 1.5) { k = max(k, 1.0 - m); } else if (u.p0 > 1.5) { k = max(k, m); }
  keyc[i] = k;
}`,
gridclear: /* wgsl */`
@group(0) @binding(0) var<storage, read_write> head: array<atomic<i32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) { if (i32(g.x) < GW*GH) { atomicStore(&head[g.x], -1); } }`,
gridbuild: /* wgsl */`
@group(0) @binding(0) var<storage, read> mt: array<u32>;
@group(0) @binding(1) var<storage, read> pos: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> head: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> nxt: array<i32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) {
  let i = g.x; if (i >= mt[0]) { return; }
  let p = pos[i];
  let cx = clamp(i32(p.x/CS), 0, GW-1); let cy = clamp(i32(p.y/CS), 0, GH-1);
  nxt[i] = atomicExchange(&head[cy*GW + cx], i32(i));
}`,
forces: /* wgsl */`
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var<storage, read> mt: array<u32>;
@group(0) @binding(2) var<storage, read> pos: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> head: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> nxt: array<i32>;
@group(0) @binding(5) var<storage, read> R: array<f32>;
@group(0) @binding(6) var<storage, read> wake: array<f32>;
@group(0) @binding(7) var<storage, read_write> mv: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) {
  let N = mt[0]; let i = g.x; if (i >= N) { return; }
  let salt = mt[1]*16u;
  let p = pos[i];
  let pi = pidx(p);
  let Rn = R[pi];
  let act = wake[pi] > 0.5;
  var F = vec2f(0.0); var crowd = 1e9;
  if (act) {
    let ip = (i + N - 1u) % N; let inx = (i + 1u) % N;
    F = ((pos[ip] + pos[inx])*0.5 - p)*u.tension;
    let si = 1.3*Rn; let rc = i32(si/CS) + 1;
    let cx = clamp(i32(p.x/CS), 0, GW-1); let cy = clamp(i32(p.y/CS), 0, GH-1);
    for (var yy = max(cy - rc, 0); yy <= min(cy + rc, GH-1); yy++) {
      for (var xx = max(cx - rc, 0); xx <= min(cx + rc, GW-1); xx++) {
        var j = atomicLoad(&head[yy*GW + xx]);
        while (j >= 0) {
          let ju = u32(j);
          if (ju != i) {
            let pj = pos[ju]; let v = pj - p; let d2 = dot(v, v);
            if (d2 < si*si) {
              var gap = select(i - ju, ju - i, ju > i);
              gap = min(gap, N - gap);
              if (gap > 3u) {
                let dist = sqrt(d2) + 1e-6;
                let rij = 0.5*(Rn + R[pidx(pj)]);
                if (gap > 10u) { crowd = min(crowd, dist/rij); }
                if (dist < rij) { F -= v*((rij - dist)/rij*u.kr/dist); }
              }
            }
          }
          j = nxt[ju];
        }
      }
    }
    F += vec2f(rndn(i, salt), rndn(i, salt + 1u))*0.05;
    let mag = length(F);
    F *= min(1.0, (u.rmin*0.22)/(mag + 1e-9));
  }
  var np = p + F;
  np = clamp(np, vec2f(MARGIN), vec2f(f32(SW) - MARGIN, f32(SH) - MARGIN));
  mv[i] = vec4f(np, crowd, select(0.0, 1.0, act));
}`,
decide: /* wgsl */`
@group(0) @binding(0) var<storage, read> mt: array<u32>;
@group(0) @binding(1) var<storage, read> mv: array<vec4f>;
@group(0) @binding(2) var<storage, read> R: array<f32>;
@group(0) @binding(3) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(4) var<uniform> u:U;
fn isKill(j:u32, salt:u32)->bool { let m = mv[j]; return m.w > 0.5 && m.z < 0.6 && rnd(j, salt + 7u) < 0.15; }
fn isShort(j:u32, N:u32)->bool {
  let a = mv[j]; let b = mv[(j + 1u) % N];
  return a.w > 0.5 && distance(a.xy, b.xy) < 0.15*R[pidx(a.xy)];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) {
  let N = mt[0]; let i = g.x; if (i >= N) { return; }
  let salt = mt[1]*16u;
  let ip = (i + N - 1u) % N; let ipp = (i + N - 2u) % N; let inx = (i + 1u) % N;
  if ((isKill(i, salt) && !isKill(ip, salt)) || (isShort(ip, N) && !isShort(ipp, N))) { cnt[i] = 0u; return; }
  let m = mv[i]; let p = m.xy; let Rn = R[pidx(p)];
  let L = distance(p, mv[inx].xy);
  let nextDropped = (isKill(inx, salt) && !isKill(i, salt)) || (isShort(i, N) && !isShort(ip, N));
  let edge = min(min(p.x - MARGIN, f32(SW) - MARGIN - p.x), min(p.y - MARGIN, f32(SH) - MARGIN - p.y));
  let free = m.w > 0.5 && m.z > 1.25 && edge > Rn;
  let pg = min(0.5, 0.12*sqrt(u.rmax/Rn));
  var grow = (L > 0.45*Rn) || (free && rnd(i, salt + 3u) < pg && L > 0.2*Rn);
  if (nextDropped || N + 20000u > MAXN) { grow = false; }
  cnt[i] = select(1u, 2u, grow);
}`,
scan1: /* wgsl */`
@group(0) @binding(0) var<storage, read> mt: array<u32>;
@group(0) @binding(1) var<storage, read> cnt: array<u32>;
@group(0) @binding(2) var<storage, read_write> scanOut: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSum: array<u32>;
var<workgroup> sh: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg:vec3u, @builtin(local_invocation_index) t:u32) {
  let N = mt[0]; let base = wg.x*512u; let i0 = base + 2u*t; let i1 = i0 + 1u;
  let a = select(0u, cnt[i0], i0 < N); let b = select(0u, cnt[i1], i1 < N);
  sh[t] = a + b;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var add = 0u; if (t >= off) { add = sh[t - off]; }
    workgroupBarrier(); sh[t] += add; workgroupBarrier();
  }
  let ex = sh[t] - (a + b);
  if (i0 < N) { scanOut[i0] = ex; }
  if (i1 < N) { scanOut[i1] = ex + a; }
  if (t == 255u) { blockSum[wg.x] = sh[255]; }
}`,
scan2: /* wgsl */`
@group(0) @binding(0) var<storage, read_write> mt: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSum: array<u32>;
var<workgroup> sh: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t:u32) {
  let nb = (mt[0] + 511u)/512u;
  let i0 = 2u*t; let i1 = i0 + 1u;
  let a = select(0u, blockSum[i0], i0 < nb); let b = select(0u, blockSum[i1], i1 < nb);
  sh[t] = a + b;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off << 1u) {
    var add = 0u; if (t >= off) { add = sh[t - off]; }
    workgroupBarrier(); sh[t] += add; workgroupBarrier();
  }
  let ex = sh[t] - (a + b);
  if (i0 < nb) { blockSum[i0] = ex; }
  if (i1 < nb) { blockSum[i1] = ex + a; }
  if (t == 255u) { mt[2] = sh[255]; }
}`,
compact: /* wgsl */`
@group(0) @binding(0) var<storage, read> mt: array<u32>;
@group(0) @binding(1) var<storage, read> mv: array<vec4f>;
@group(0) @binding(2) var<storage, read> cnt: array<u32>;
@group(0) @binding(3) var<storage, read> scanOut: array<u32>;
@group(0) @binding(4) var<storage, read> blockOff: array<u32>;
@group(0) @binding(5) var<storage, read_write> pos: array<vec2f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) {
  let N = mt[0]; let i = g.x; if (i >= N) { return; }
  let c = cnt[i]; if (c == 0u) { return; }
  let salt = mt[1]*16u;
  let base = scanOut[i] + blockOff[i/512u];
  let p = mv[i].xy; pos[base] = p;
  if (c == 2u) {
    let q = mv[(i + 1u) % N].xy; let d = q - p;
    let nrm = vec2f(-d.y, d.x)/(length(d) + 1e-9);
    pos[base + 1u] = (p + q)*0.5 + nrm*rndn(i, salt + 11u)*0.4;
  }
}`,
finalize: /* wgsl */`
@group(0) @binding(0) var<storage, read_write> mt: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let n = mt[2]; mt[0] = n; mt[1] = mt[1] + 1u;
  args[0] = (n + 63u)/64u; args[1] = 1u; args[2] = 1u;
  args[4] = (n + 511u)/512u; args[5] = 1u; args[6] = 1u;
  args[8] = 12u; args[9] = n; args[10] = 0u; args[11] = 0u;
}`,
smooth: /* wgsl */`
@group(0) @binding(0) var<storage, read> mt: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> b: array<vec2f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g:vec3u) {
  let N = mt[0]; let i = g.x; if (i >= N) { return; }
  b[i] = a[i]*0.5 + (a[(i + N - 1u) % N] + a[(i + 1u) % N])*0.25;
}`,
};

const DRAW = /* wgsl */`
struct DU { a:vec4f, col:vec4f, b:vec4f };
@group(0) @binding(0) var<uniform> du:DU;
@group(0) @binding(1) var<storage, read> pts: array<vec2f>;
@group(0) @binding(2) var<storage, read> mt: array<u32>;
fn snap(p:vec2f)->vec2f { return select(p, floor(p/du.b.y + 0.5)*du.b.y, du.b.x > 0.5); }
@vertex fn vs(@builtin(vertex_index) vi:u32, @builtin(instance_index) ii:u32) -> @builtin(position) vec4f {
  let N = mt[0];
  let q = vi/6u; let c = array<u32,6>(0u,1u,2u,2u,1u,3u)[vi%6u];
  let P0 = snap(pts[ii]); let P1 = snap(pts[(ii + 1u) % N]);
  // mode 1: two axis-aligned legs per segment, elbow at (P1.x, P0.y)
  let E = select(P1, vec2f(P1.x, P0.y), du.b.x > 0.5);
  let A = select(P0, E, q == 1u)*du.a.x;
  let B = select(E, P1, q == 1u)*du.a.x;
  let d = B - A; let l = length(d);
  var hw = du.a.y; if (l < 1e-4) { hw = 0.0; }
  let dir = d/max(l, 1e-4); let nrm = vec2f(-dir.y, dir.x);
  let e = select(A, B, c >= 2u);
  let along = select(-1.0, 1.0, c >= 2u);
  let side = select(-1.0, 1.0, (c & 1u) == 1u);
  let p = e + dir*hw*along + nrm*hw*side;
  return vec4f(p.x/du.a.z*2.0 - 1.0, 1.0 - p.y/du.a.w*2.0, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return du.col; }
`;


const DRAW8 = /* wgsl */`
struct DU { a:vec4f, col:vec4f, b:vec4f };
@group(0) @binding(0) var<uniform> du:DU;
@group(0) @binding(1) var<storage, read> pts: array<vec2f>;
@group(0) @binding(2) var<storage, read> mt: array<u32>;
@group(0) @binding(3) var<storage, read> st: array<f32>;
struct VO { @builtin(position) pos:vec4f, @location(0) @interpolate(flat) tone:f32 };
@vertex fn vs(@builtin(vertex_index) vi:u32, @builtin(instance_index) ii:u32) -> VO {
  var o: VO; o.tone = 0.0; o.pos = vec4f(0.0, 0.0, 0.0, 1.0);
  if (vi >= 6u) { return o; }
  let N = mt[0];
  let c = array<u32,6>(0u,1u,2u,2u,1u,3u)[vi];
  let p0 = pts[ii]; let p1 = pts[(ii + 1u) % N];
  let A = p0*du.a.x; let B = p1*du.a.x;
  let d = B - A; let l = max(length(d), 1e-4); let dir = d/l; let nrm = vec2f(-dir.y, dir.x);
  let hw = du.a.y;
  let e = select(A, B, c >= 2u);
  let along = select(-1.0, 1.0, c >= 2u);
  let side = select(-1.0, 1.0, (c & 1u) == 1u);
  let p = e + dir*hw*along + nrm*hw*side;
  o.pos = vec4f(p.x/du.a.z*2.0 - 1.0, 1.0 - p.y/du.a.w*2.0, 0.0, 1.0);
  let q = (p0 + p1)*0.5;
  let ix = u32(clamp(i32(q.x), 0, ${SW - 1})); let iy = u32(clamp(i32(q.y), 0, ${SH - 1}));
  o.tone = st[iy*${SW}u + ix];
  return o;
}
@fragment fn fs(v:VO) -> @location(0) vec4f {
  let t = clamp(v.tone*1.6 - 0.08, 0.0, 1.0);
  // Game Boy greens: faint in light areas, darkest where the image is dark
  var c = vec3f(0.545, 0.675, 0.059);
  if (t > 0.62) { c = vec3f(0.059, 0.220, 0.059); } else if (t > 0.30) { c = vec3f(0.188, 0.384, 0.188); }
  return vec4f(c, 1.0);
}
`;

const bilFn = (name, buf) => `fn ${name}(p:vec2f)->f32 {
  let x = clamp(p.x - 0.5, 0.0, ${SW - 1}.0 - 0.001); let y = clamp(p.y - 0.5, 0.0, ${SH - 1}.0 - 0.001);
  let x0 = u32(x); let y0 = u32(y); let fx = x - f32(x0); let fy = y - f32(y0); let i = y0*${SW}u + x0;
  return mix(mix(${buf}[i], ${buf}[i + 1u], fx), mix(${buf}[i + ${SW}u], ${buf}[i + ${SW + 1}u], fx), fy);
}`;
const DRAWV2 = /* wgsl */`
struct V2 { a:vec4f, b:vec4f, c:vec4f, d:vec4f, e:vec4f };
@group(0) @binding(0) var<uniform> q:V2;
@group(0) @binding(1) var<storage, read> st: array<f32>;
@group(0) @binding(2) var<storage, read> hs: array<f32>;
@group(0) @binding(3) var<storage, read> hl: array<f32>;
@group(0) @binding(4) var<storage, read> mk: array<f32>;
${bilFn('bst', 'st')}
${bilFn('bhs', 'hs')}
${bilFn('bhl', 'hl')}
${bilFn('bmk', 'mk')}

@vertex fn vs(@builtin(vertex_index) i:u32) -> @builtin(position) vec4f {
  let p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) f:vec4f) -> @location(0) vec4f {
  let S = q.a.z; let uv = f.xy;
  let bill = q.c.w > 0.5;
  // bill: the canvas is bill shaped and the portrait frame sits in a panel (e.xy = origin, e.z = canvas px per sim px)
  var p = uv*0.5;
  if (bill) { p = (uv - q.e.xy)/q.e.z; }
  let inb = select(0.0, 1.0, p.x > 0.0 && p.y > 0.0 && p.x < ${SW}.0 && p.y < ${SH}.0);
  let d = bst(p);
  let dd = clamp(d*1.6 - 0.08, 0.0, 1.0);
  let zone = max(q.b.w, 0.01);
  let t = smoothstep(0.0, zone, dd);                  // 0 in the brightest areas, 1 once lines run continuous
  var fr = mix(q.b.y, q.b.z, pow(dd, 0.8));           // line weight as a fraction of the spacing: thin in light, thick in dark
  if (bill) { fr = mix(max(fr, 0.14), fr, t); }       // brightest areas: points
  var v = uv.y/S;
  if (q.b.x < 0.5) { v = (uv.y + q.a.w*bhl(p))/S; }   // relief: rows ride up over the subject
  let fw = max(fwidth(v), 1e-4);
  var a = clamp((fr*0.5 - abs(fract(v) - 0.5))/fw + 0.5, 0.0, 1.0);
  a = min(a, clamp(fr/fw, 0.0, 1.0));
  if (bill) {
    let un = uv.x/q.d.w;                              // position along the row, in dash periods
    let fx = max(fwidth(un), 1e-4);
    let row = floor(v);
    let u = un + fract(row*0.618034);                 // stagger neighbouring rows
    let onf = mix(0.14*S/q.d.w, 1.0, t);              // dash length: a dot at the brightest, whole line at the edge of the zone
    let da = clamp((onf*0.5 - abs(fract(u) - 0.5))/fx + 0.5, 0.0, 1.0);
    a = a*select(da, 1.0, onf > 0.999);
  }
  // outline: stroke on the subject silhouette plus a paper gap around it (works with relief or straight lines)
  let ol = select(0.0, 1.0, !bill);
  let hm = bhs(p);
  let gx = bhs(p + vec2f(1.0, 0.0)) - bhs(p - vec2f(1.0, 0.0));
  let gy = bhs(p + vec2f(0.0, 1.0)) - bhs(p - vec2f(0.0, 1.0));
  let sd = clamp((hm - 0.5)/max(length(vec2f(gx, gy))*0.5, 1e-4)*2.0, -300.0, 300.0);   // signed distance to the silhouette, output px
  let ow = q.b.w*ol; let gap = q.d.w*ol;
  let ao = clamp(ow*0.5 - abs(sd) + 0.5, 0.0, 1.0)*step(0.5, ow);
  let hal = clamp(ow*0.5 + gap - abs(sd) + 0.5, 0.0, 1.0)*ol;
  a = max(a*(1.0 - hal), ao);
  if (bill) {                                         // only the subject gets lines, the rest stays transparent
    let cover = smoothstep(0.3, 0.7, bmk(p))*inb;          // follows the mask itself (soft matte), not the blurred shape
    return vec4f(mix(q.d.rgb, q.c.rgb, a)*cover, cover);   // opaque subject in the bill colour, lines on it
  }
  return vec4f(mix(q.d.rgb, q.c.rgb, a), 1.0);
}
`;


const DRAWV4 = /* wgsl */`
struct V4 { a:vec4f, b:vec4f, c:vec4f, d:vec4f, e:vec4f, f:vec4f };
@group(0) @binding(0) var<uniform> q:V4;
@group(0) @binding(1) var<storage, read> st: array<f32>;
@group(0) @binding(2) var<storage, read> mk: array<f32>;
${bilFn('bst', 'st')}
${bilFn('bmk', 'mk')}
@vertex fn vs(@builtin(vertex_index) i:u32) -> @builtin(position) vec4f {
  let p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  return vec4f(p[i], 0.0, 1.0);
}
const H3 = 0.8660254;
struct Cell { kind:i32, ci:i32, cj:i32, u:f32, v:f32, o:vec2f, e1:vec2f, e2:vec2f, gu:f32, gv:f32 };
fn lat(i:i32, j:i32, M:f32) -> vec2f { return vec2f(f32(i)*H3*M, (f32(i)*0.5 + f32(j))*M); }
// rhombille tiling of the triangular lattice: every triangle has exactly one omitted edge, the two triangles around it make one rhombus
fn cellAt(p:vec2f, M:f32) -> Cell {
  let h = H3*M;
  let al = p.x/h; let be = (p.y - al*M*0.5)/M;
  let fi = floor(al); let fj = floor(be);
  let i = i32(fi); let j = i32(fj);
  let upper = (al - fi) + (be - fj) >= 1.0;
  let m = ((i - j) % 3 + 3) % 3;
  var kind = 2; var ci = i; var cj = j;
  if (!upper) { if (m == 0) { kind = 0; } else if (m == 1) { kind = 1; } }
  else { if (m == 1) { kind = 0; cj = j + 1; } else if (m == 0) { kind = 1; ci = i + 1; } }
  let A = vec2f(h, M*0.5); let B = vec2f(0.0, M); let D = A - B;
  var c: Cell; c.kind = kind; c.ci = ci; c.cj = cj;
  if (kind == 1) { c.o = lat(ci - 1, cj + 1, M); c.e1 = D; c.e2 = A; }
  else if (kind == 0) { c.o = lat(ci, cj, M); c.e1 = B; c.e2 = D; }
  else { c.o = lat(ci, cj, M); c.e1 = A; c.e2 = B; }
  let det = c.e1.x*c.e2.y - c.e1.y*c.e2.x;
  let r = p - c.o;
  c.u = clamp((r.x*c.e2.y - r.y*c.e2.x)/det, 0.0, 1.0);
  c.v = clamp((c.e1.x*r.y - c.e1.y*r.x)/det, 0.0, 1.0);
  c.gu = length(vec2f(c.e2.y, -c.e2.x))/abs(det);
  c.gv = length(vec2f(-c.e1.y, c.e1.x))/abs(det);
  return c;
}
fn hash2(a:i32, b:i32) -> f32 {
  var x = u32(a)*374761393u + u32(b)*668265263u + 12345u;
  x = (x ^ (x >> 13u))*1274126177u; x = x ^ (x >> 16u);
  return f32(x & 65535u)/65535.0;
}
fn toCanvas(pp:vec2f) -> vec2f { return vec2f(pp.x, -pp.y); }
fn levelAt(pp:vec2f) -> f32 {
  let sp = toCanvas(pp)*0.5;
  let dd = clamp(bst(sp)*1.6 - 0.08, 0.0, 1.0);
  let L = q.b.x;
  let lv = min(floor(dd*L), L - 1.0);
  return select(q.c.w, lv, bmk(sp) > 0.5);
}
fn lvn(c:Cell, i:i32, j:i32, N:f32, L:f32) -> f32 {
  return levelAt(c.o + (c.e1*(f32(i) + 0.5) + c.e2*(f32(j) + 0.5))/N)/max(L - 1.0, 1.0);
}
// centre of the big cube (hexagon) containing pp: nearest point of the sublattice (i - j) mod 3 == k0
fn hexCtr(pp:vec2f, M:f32, k0:i32) -> vec2f {
  let h = H3*M;
  let al = pp.x/h; let be = (pp.y - al*M*0.5)/M;
  let i0 = i32(floor(al)); let j0 = i32(floor(be));
  var best = vec2f(0.0); var bd = 1e20;
  for (var di = -3; di <= 3; di++) {
    for (var dj = -3; dj <= 3; dj++) {
      let i = i0 + di; let j = j0 + dj;
      if ((((i - j) % 3) + 3) % 3 != k0) { continue; }
      let p = lat(i, j, M); let d = dot(pp - p, pp - p);
      if (d < bd) { bd = d; best = p; }
    }
  }
  return best;
}
// distance in px from the hexagon outline, p relative to its centre
fn hexEdge(p:vec2f, M:f32) -> f32 {
  let a = vec2f(H3*M, 1.5*M); let b = vec2f(H3*M, -1.5*M); let c = vec2f(2.0*H3*M, 0.0);
  let la = length(a);
  let da = la*0.5 - abs(dot(p, a))/la;
  let db = la*0.5 - abs(dot(p, b))/la;
  let dc = la*0.5 - abs(dot(p, c))/la;
  return min(da, min(db, dc));
}
fn darkAt(pp:vec2f) -> f32 {
  let sp = toCanvas(pp)*0.5;
  let dd = clamp(bst(sp)*1.6 - 0.08, 0.0, 1.0);
  return select(q.c.w/max(q.b.x - 1.0, 1.0), dd, bmk(sp) > 0.5);
}
// mean darkness over a uv rectangle of a face, from 5 samples
fn regionDark(c:Cell, r:vec4f) -> f32 {
  var t = 0.0;
  for (var k = 0; k < 5; k++) {
    let fx = select(select(0.25, 0.75, (k & 1) == 1), 0.5, k == 4);
    let fy = select(select(0.25, 0.75, (k & 2) == 2), 0.5, k == 4);
    let uu = mix(r.x, r.z, fx); let vv = mix(r.y, r.w, fy);
    t += darkAt(c.o + c.e1*uu + c.e2*vv);
  }
  return t*0.2;
}
fn cov(d:f32, w:f32) -> f32 { return clamp(w*0.5 - d + 0.5, 0.0, 1.0); }
@fragment fn fs(@builtin(position) f:vec4f) -> @location(0) vec4f {
  let s = q.a.z; let N = q.a.w; let M = s*N;
  let pp = vec2f(f.x, -f.y);
  let c = cellAt(pp, M);
  let L = q.b.x;
  let warp = q.e.x;
  let n = i32(N);
  if (q.e.y > 4.5) {
    // volume: layers of cubes of growing size over the finest grid. Inside the subject a random share of the cubes of every size exists (random sizes),
    // outside the subject the cubes follow the light but sit shifted off their lattice position, overlapping like a honeycomb pushed around on one layer
    let s0 = q.a.z;
    let top = i32(q.e.z);
    var rcol = vec3f(0.0);
    for (var L2 = 4; L2 >= 0; L2--) {
      if (L2 > top) { continue; }
      let Mk = s0*pow(q.f.x, f32(L2));
      let off = lat(3*L2 + 3, 3*L2, s0);
      let pk = pp - off;
      let h = H3*Mk;
      let al = pk.x/h; let be = (pk.y - al*Mk*0.5)/Mk;
      let i0 = i32(floor(al)); let j0 = i32(floor(be));
      var bestH = -1.0; var bestC = vec2f(0.0);
      for (var di = -2; di <= 2; di++) {
        for (var dj = -2; dj <= 2; dj++) {
          let i = i0 + di; let j = j0 + dj;
          if ((((i - j) % 3) + 3) % 3 != 0) { continue; }
          let lp = lat(i, j, Mk);
          let ca0 = off + lp;
          let sp0 = toCanvas(ca0)*0.5;
          let inner = bmk(sp0);
          var jit = vec2f(0.0);
          if (L2 > 0) {
            jit = vec2f(hash2(i*13 + L2*7, j*17 + 3), hash2(i*19 + 5, j*11 + L2*9)) - vec2f(0.5);
            let ph = vec2f(hash2(i*23 + 1, j*29 + L2), hash2(i*31 + 7, j*37 + L2*3))*6.2831853;
            let tt = q.f.w;
            jit = jit*1.2 + vec2f(sin(tt*1.3 + ph.x), cos(tt*1.1 + ph.y))*0.6;
            // the subject pushes the cubes around it away: the mask gradient at a ring around the cube
            let rg = Mk*1.5;
            let gx = bmk(toCanvas(ca0 + vec2f(rg, 0.0))*0.5) - bmk(toCanvas(ca0 - vec2f(rg, 0.0))*0.5);
            let gy = bmk(toCanvas(ca0 + vec2f(0.0, rg))*0.5) - bmk(toCanvas(ca0 - vec2f(0.0, rg))*0.5);
            jit = jit - vec2f(gx, gy)*1.6;
            jit = jit*Mk*q.f.y*(1.0 - inner);
          }
          let p2c = pk - lp - jit;
          if (hexEdge(p2c, Mk) < 0.0) { continue; }
          let hp = hash2(i*3 + L2*101, j*5 + 77);
          if (L2 > 0) {
            var ex = true;
            if (inner > 0.5 && q.f.z > 0.001) {
              ex = hash2(i*3 + 1 + L2*101, j*5 + 2 + L2*57) < q.f.z*0.5/(1.0 + 0.3*f32(L2));
            } else {
              let ca = ca0 + jit;
              let rr = Mk*0.5;
              let dm = (darkAt(ca) + darkAt(ca + vec2f(rr, 0.0)) + darkAt(ca - vec2f(rr, 0.0)) + darkAt(ca + vec2f(0.0, rr)) + darkAt(ca - vec2f(0.0, rr)))*0.2;
              let jt2 = (hash2(i*3 + 1 + L2*101, j*5 + 2 + L2*57) - 0.5)*0.2;
              let thr = (f32(L2)/(f32(top) + 1.0))*(1.6 - 1.2*q.e.w);
              ex = !(1.0 - dm < thr + jt2);
            }
            if (!ex) { continue; }
          }
          if (hp > bestH) { bestH = hp; bestC = lp + jit; }
        }
      }
      if (bestH < 0.0) { continue; }
      let ctr = bestC;
      let id = vec2i(i32(round(ctr.x/(H3*Mk))), i32(round(ctr.y/(Mk*0.5))));
      let p2 = pk - ctr;
      let c2 = cellAt(p2, Mk);
      let cen2 = off + ctr + c2.o + (c2.e1 + c2.e2)*0.5;
      let dk = darkAt(cen2);
      var rtone = 0.80;
      if (c2.kind == 0) { rtone = 0.60; } else if (c2.kind == 2) { rtone = 0.94; }
      if (c2.kind != 1 && hash2(id.x*7 + L2, id.y*11 + c2.kind) < q.b.z) { rtone = 1.54 - rtone; }
      let rval = clamp(rtone - q.b.y*dk, 0.0, 1.0);
      let cc = mix(q.c.rgb, q.d.rgb, rval);
      let lw2 = q.d.w;
      let ru = min(c2.u, 1.0 - c2.u)/c2.gu; let rv = min(c2.v, 1.0 - c2.v)/c2.gv;
      var rink = max(cov(ru, lw2*1.5), cov(rv, lw2*1.5));
      rink = max(rink, cov(hexEdge(p2, Mk), lw2*2.0));
      rcol = mix(cc, q.c.rgb, clamp(rink, 0.0, 1.0));
      break;
    }
    return vec4f(rcol, 1.0);
  }
  if (q.e.y > 3.5) {
    // small to big: the finest cube grid covers everything, then layers of bigger cubes (each one a set ratio bigger than the one below) are laid over it.
    // On every layer a random share of the cubes exists; the biggest layer wins, so big cubes cover the small ones around them
    let s0 = q.a.z;
    let top = i32(q.e.z);
    var rcol = vec3f(0.0);
    for (var L2 = 4; L2 >= 0; L2--) {
      if (L2 > top) { continue; }
      let Mk = s0*pow(q.f.x, f32(L2));
      let off = lat(3*L2 + 3, 3*L2, s0*1.0);
      let pk = pp - off;
      let ctr = hexCtr(pk, Mk, 0);
      let id = vec2i(i32(round(ctr.x/(H3*Mk))), i32(round(ctr.y/(Mk*0.5))));
      if (L2 > 0) {
        // size follows the feed: a bigger cube only exists where the picture under it is light, so light areas grow big cubes,
        // dark areas keep the finest ones, and the sizes move with the video
        let ca = off + ctr;
        let rr = Mk*0.5;
        let dm = (darkAt(ca) + darkAt(ca + vec2f(rr, 0.0)) + darkAt(ca - vec2f(rr, 0.0)) + darkAt(ca + vec2f(0.0, rr)) + darkAt(ca - vec2f(0.0, rr)))*0.2;
        let jit = (hash2(id.x*3 + 1 + L2*101, id.y*5 + 2 + L2*57) - 0.5)*0.2;
        let thr = (f32(L2)/(f32(top) + 1.0))*(1.6 - 1.2*q.e.w);
        if (1.0 - dm < thr + jit) { continue; }
      }
      let p2 = pk - ctr;
      let c2 = cellAt(p2, Mk);
      let cen2 = off + ctr + c2.o + (c2.e1 + c2.e2)*0.5;
      let dk = darkAt(cen2);
      var rtone = 0.80;
      if (c2.kind == 0) { rtone = 0.60; } else if (c2.kind == 2) { rtone = 0.94; }
      if (c2.kind != 1 && hash2(id.x*7 + L2, id.y*11 + c2.kind) < q.b.z) { rtone = 1.54 - rtone; }
      let rval = clamp(rtone - q.b.y*dk, 0.0, 1.0);
      var cc = mix(q.c.rgb, q.d.rgb, rval);
      let lw2 = q.d.w;
      let ru = min(c2.u, 1.0 - c2.u)/c2.gu; let rv = min(c2.v, 1.0 - c2.v)/c2.gv;
      var rink = max(cov(ru, lw2*1.5), cov(rv, lw2*1.5));
      rink = max(rink, cov(hexEdge(p2, Mk), lw2*2.0));
      rcol = mix(cc, q.c.rgb, clamp(rink, 0.0, 1.0));
      break;
    }
    return vec4f(rcol, 1.0);
  }
  if (q.e.y > 2.5) {
    // random cube sizes (test, no logic): the plane is cut into big cubes (hexagons); each one is filled with the iso cube pattern at a random
    // smaller scale, anchored on its centre so the pattern lines up with the big cube's outline
    let ctr = hexCtr(pp, M, 0);
    let id = vec2i(i32(round(ctr.x/(H3*M))), i32(round(ctr.y/(M*0.5))));
    let steps = q.e.z;
    let kk = min(i32(floor(hash2(id.x*3 + 1, id.y*5 + 2)*(steps + 1.0))), i32(steps));
    let M2 = M/f32(1 << u32(kk));
    let p2 = pp - ctr;
    let c2 = cellAt(p2, M2);
    let cen2 = ctr + c2.o + (c2.e1 + c2.e2)*0.5;
    let dk = darkAt(cen2);
    var rtone = 0.80;
    if (c2.kind == 0) { rtone = 0.60; } else if (c2.kind == 2) { rtone = 0.94; }
    if (c2.kind != 1 && hash2(c2.ci + id.x*7, c2.cj*7 + c2.kind + id.y*11) < q.b.z) { rtone = 1.54 - rtone; }
    let rval = clamp(rtone - q.b.y*dk, 0.0, 1.0);
    var rcol = mix(q.c.rgb, q.d.rgb, rval);
    let lw2 = q.d.w;
    let ru = min(c2.u, 1.0 - c2.u)/c2.gu; let rv = min(c2.v, 1.0 - c2.v)/c2.gv;
    var rink = max(cov(ru, lw2*1.5), cov(rv, lw2*1.5));
    rink = max(rink, cov(hexEdge(p2, M), lw2*2.5));
    rcol = mix(rcol, q.c.rgb, clamp(rink, 0.0, 1.0));
    return vec4f(rcol, 1.0);
  }
  if (q.e.y > 1.5) {
    // golden subdivision: each face is a big square that splits into golden-ratio squares and rectangles, nested like a spiral.
    // Light areas stop early (big whole cubes), dark areas keep splitting, so sizes shrink steadily instead of sitting on one grid
    let minpx = max(s, 2.0);
    let pu = clamp(c.u, 0.0, 0.9999); let pv = clamp(c.v, 0.0, 0.9999);
    var r = vec4f(0.0, 0.0, 1.0, 1.0);
    for (var it = 0; it < 14; it++) {
      let w = r.z - r.x; let h = r.w - r.y;
      let big = max(w, h)*M;
      let sq = max(w, h) < min(w, h)*1.2;
      let t = regionDark(c, r);
      let tgt = mix(M, minpx, pow(t, 0.7));
      if (big <= minpx*1.05) { break; }
      if (sq && big <= tgt) { break; }
      let fb = hash2(c.ci*31 + i32(r.x*8192.0) + it, c.cj*17 + i32(r.y*8192.0) + c.kind*5);
      let fb2 = fract(fb*7.31);
      if (sq) {
        let xm = r.x + w*select(0.381966, 0.618034, fb > 0.5);
        let ym = r.y + h*select(0.381966, 0.618034, fb2 > 0.5);
        if (pu < xm) { r.z = xm; } else { r.x = xm; }
        if (pv < ym) { r.w = ym; } else { r.y = ym; }
      } else if (w > h) {
        // cut a square of side h off one end, the rest stays a golden rectangle
        let xm = select(r.z - h, r.x + h, fb > 0.5);
        if (pu < xm) { r.z = xm; } else { r.x = xm; }
      } else {
        let ym = select(r.w - w, r.y + w, fb > 0.5);
        if (pv < ym) { r.w = ym; } else { r.y = ym; }
      }
    }
    let gt = regionDark(c, r);
    var gtone = 0.80;
    if (c.kind == 0) { gtone = 0.60; } else if (c.kind == 2) { gtone = 0.94; }
    if (c.kind != 1 && hash2(c.ci, c.cj*7 + c.kind) < q.b.z) { gtone = 1.54 - gtone; }
    let gval = clamp(gtone - q.b.y*gt, 0.0, 1.0);
    var gcol = mix(q.c.rgb, q.d.rgb, gval);
    let gdu = min(pu - r.x, r.z - pu)/c.gu;
    let gdv = min(pv - r.y, r.w - pv)/c.gv;
    var gink = max(cov(gdu, q.d.w), cov(gdv, q.d.w));
    let gmu = min(c.u, 1.0 - c.u)/c.gu; let gmv = min(c.v, 1.0 - c.v)/c.gv;
    gink = max(gink, max(cov(gmu, q.d.w*2.0), cov(gmv, q.d.w*2.0)));
    gcol = mix(gcol, q.c.rgb, clamp(gink, 0.0, 1.0));
    return vec4f(gcol, 1.0);
  }
  if (q.e.y > 0.5 && q.e.y < 1.5) {
    // size contrast: light areas merge pixels into one big block, dark areas stay split into small ones (like line spacing in V1)
    var top = 1; var topE = 0;
    for (var k = 0; k < 5; k++) { if (top < n) { top *= 2; topE += 1; } }
    let px = min(i32(floor(clamp(c.u, 0.0, 0.9999)*N)), n - 1); let py = min(i32(floor(clamp(c.v, 0.0, 0.9999)*N)), n - 1);
    var b = top; var u0 = 0; var v0 = 0; var u1 = n; var v1 = n;
    for (var it = 0; it < 6; it++) {
      u0 = (px/b)*b; v0 = (py/b)*b; u1 = min(u0 + b, n); v1 = min(v0 + b, n);
      var mn = 99;
      for (var j = v0; j < v1; j++) {
        for (var i = u0; i < u1; i++) {
          let e = i32(floor((1.0 - lvn(c, i, j, N, L))*(f32(topE) + 0.999)));
          mn = min(mn, 1 << u32(e));
        }
      }
      if (mn >= b || b == 1) { break; }
      b = b/2;
    }
    let bcen = c.o + (c.e1*(f32(u0 + u1)*0.5) + c.e2*(f32(v0 + v1)*0.5))/N;
    let blv = levelAt(bcen);
    var btone = 0.80;
    if (c.kind == 0) { btone = 0.60; } else if (c.kind == 2) { btone = 0.94; }
    if (c.kind != 1 && hash2(c.ci, c.cj*7 + c.kind) < q.b.z) { btone = 1.54 - btone; }
    let bval = clamp(btone - q.b.y*blv/max(L - 1.0, 1.0), 0.0, 1.0);
    var bcol = mix(q.c.rgb, q.d.rgb, bval);
    let gu = c.u*N; let gv = c.v*N;
    let du = min(gu - f32(u0), f32(u1) - gu)/(N*c.gu);
    let dv = min(gv - f32(v0), f32(v1) - gv)/(N*c.gv);
    let blw = q.d.w;
    var bink = max(cov(du, blw), cov(dv, blw));
    let bmu = min(c.u, 1.0 - c.u)/c.gu; let bmv = min(c.v, 1.0 - c.v)/c.gv;
    bink = max(bink, max(cov(bmu, blw*2.0), cov(bmv, blw*2.0)));
    bcol = mix(bcol, q.c.rgb, clamp(bink, 0.0, 1.0));
    return vec4f(bcol, 1.0);
  }
  // pixel size follows its level: rows (along face axis 2) and the pixels inside a row (axis 1) get weights, the face edges stay fixed,
  // so a pixel that grows takes the space from its neighbours
  var rw: array<f32,10>;
  var rt = 0.0;
  for (var j = 0; j < n; j++) {
    var s = 0.0;
    for (var i = 0; i < n; i++) { s += lvn(c, i, j, N, L); }
    rw[j] = 1.0 + warp*s/N; rt += rw[j];
  }
  var jj = n - 1; var vfr = 0.9999; var acc = 0.0; let tv = clamp(c.v, 0.0, 1.0)*rt;
  for (var j = 0; j < n; j++) {
    if (tv < acc + rw[j]) { jj = j; vfr = (tv - acc)/rw[j]; break; }
    acc += rw[j];
  }
  var cw: array<f32,10>;
  var ct = 0.0;
  for (var i = 0; i < n; i++) { cw[i] = 1.0 + warp*lvn(c, i, jj, N, L); ct += cw[i]; }
  var ii = n - 1; var ufr = 0.9999; acc = 0.0; let tu = clamp(c.u, 0.0, 1.0)*ct;
  for (var i = 0; i < n; i++) {
    if (tu < acc + cw[i]) { ii = i; ufr = (tu - acc)/cw[i]; break; }
    acc += cw[i];
  }
  let su0 = cw[ii]/ct; let sv0 = rw[jj]/rt;
  let fu = f32(ii) + ufr; let fv = f32(jj) + vfr;
  let cen = c.o + (c.e1*(f32(ii) + 0.5) + c.e2*(f32(jj) + 0.5))/N;
  let lv = levelAt(cen);
  let su = select(-1.0, 1.0, fract(fu) > 0.5); let sv = select(-1.0, 1.0, fract(fv) > 0.5);
  let lvU = levelAt(cen + c.e1*su/N); let lvV = levelAt(cen + c.e2*sv/N);
  var tone = 0.80;
  if (c.kind == 0) { tone = 0.60; } else if (c.kind == 2) { tone = 0.94; }
  if (c.kind != 1 && hash2(c.ci, c.cj*7 + c.kind) < q.b.z) { tone = 1.54 - tone; }
  let depth = q.b.y;
  let val = clamp(tone - depth*lv/max(L - 1.0, 1.0), 0.0, 1.0);
  var col = mix(q.c.rgb, q.d.rgb, val);
  let fru = min(fract(fu), 1.0 - fract(fu))*su0/c.gu;
  let frv = min(fract(fv), 1.0 - fract(fv))*sv0/c.gv;
  let lw = q.d.w;
  let heavyU = select(q.b.w, 1.0, abs(lvU - lv) > 0.5);
  let heavyV = select(q.b.w, 1.0, abs(lvV - lv) > 0.5);
  var ink = cov(fru, lw*(0.6 + 0.8*heavyU))*heavyU;
  ink = max(ink, cov(frv, lw*(0.6 + 0.8*heavyV))*heavyV);
  let mu = min(c.u, 1.0 - c.u)/c.gu; let mv = min(c.v, 1.0 - c.v)/c.gv;
  ink = max(ink, max(cov(mu, lw*2.0), cov(mv, lw*2.0)));
  col = mix(col, q.c.rgb, clamp(ink, 0.0, 1.0));
  return vec4f(col, 1.0);
}
`;

const BLIT = /* wgsl */`
struct BU { ps:vec4f };
@group(0) @binding(0) var low: texture_2d<f32>;
@group(0) @binding(1) var<uniform> bu:BU;
@vertex fn vs(@builtin(vertex_index) i:u32) -> @builtin(position) vec4f {
  let p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@fragment fn fs(@builtin(position) f:vec4f) -> @location(0) vec4f {
  let d = vec2i(textureDimensions(low));
  return textureLoad(low, clamp(vec2i(f.xy/bu.ps.x), vec2i(0), d - vec2i(1)), 0);
}
`;

// ---------------------------------------------------------------- pipelines
function kernel(name, spec) {
  const entries = spec.map((t, i) => ({
    binding: i, visibility: GPUShaderStage.COMPUTE,
    ...(t === 'u' ? { buffer: { type: 'uniform' } } : t === 'ro' ? { buffer: { type: 'read-only-storage' } }
      : t === 'rw' ? { buffer: { type: 'storage' } } : { texture: { sampleType: 'float' } }),
  }));
  const bgl = device.createBindGroupLayout({ entries });
  const module = device.createShaderModule({ code: COMMON + WGSL[name] });
  module.getCompilationInfo().then((ci) => { for (const m of ci.messages) if (m.type === 'error') msg(`WGSL ${name}:${m.lineNum - COMMON.split('\n').length + 1} ${m.message}`); });
  K[name] = {
    bgl,
    pipe: device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: 'main' } }),
  };
}
kernel('prep', ['u', 't', 'rw', 'rw']);
kernel('blur', ['ro', 'rw', 'u']);
kernel('dilate', ['ro', 'rw', 'u']);
kernel('tilehist', ['u', 'ro', 'rw']);
kernel('claheapply', ['ro', 'ro', 'ro', 'rw', 'rw']);
kernel('percentile', ['ro', 'rw']);
kernel('darkprep', ['ro', 'ro', 'ro', 'rw']);
kernel('rfield', ['u', 'ro', 'rw', 'rw', 'rw', 'ro', 'ro']);
kernel('keymix', ['u', 'ro', 'rw']);
kernel('gridclear', ['rw']);
kernel('gridbuild', ['ro', 'ro', 'rw', 'rw']);
kernel('forces', ['u', 'ro', 'ro', 'rw', 'ro', 'ro', 'ro', 'rw']);
kernel('decide', ['ro', 'ro', 'ro', 'rw', 'u']);
kernel('scan1', ['ro', 'ro', 'rw', 'rw']);
kernel('scan2', ['rw', 'rw']);
kernel('compact', ['ro', 'ro', 'ro', 'ro', 'ro', 'rw']);
kernel('finalize', ['rw', 'rw']);
kernel('smooth', ['ro', 'ro', 'rw']);

const canvas = $('#cv');
const ctx = canvas.getContext('webgpu');
const FORMAT = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format: FORMAT, alphaMode: 'premultiplied' });
const drawMod = device.createShaderModule({ code: DRAW });
const drawPipe = device.createRenderPipeline({
  layout: 'auto',
  vertex: { module: drawMod, entryPoint: 'vs' },
  fragment: { module: drawMod, entryPoint: 'fs', targets: [{ format: FORMAT }] },
  primitive: { topology: 'triangle-list' },
  multisample: { count: 4 },
});

const m8 = device.createShaderModule({ code: DRAW8 });
const pipe8 = device.createRenderPipeline({ layout: 'auto', vertex: { module: m8, entryPoint: 'vs' },
  fragment: { module: m8, entryPoint: 'fs', targets: [{ format: FORMAT }] }, primitive: { topology: 'triangle-list' } });
const mBlit = device.createShaderModule({ code: BLIT });
const pipeBlit = device.createRenderPipeline({ layout: 'auto', vertex: { module: mBlit, entryPoint: 'vs' },
  fragment: { module: mBlit, entryPoint: 'fs', targets: [{ format: FORMAT }] }, primitive: { topology: 'triangle-list' } });
const mV2 = device.createShaderModule({ code: DRAWV2 });
mV2.getCompilationInfo().then((ci) => { for (const m of ci.messages) if (m.type === 'error') msg(`V2 shader line ${m.lineNum}: ${m.message}`); });
const v2bgl = device.createBindGroupLayout({ entries: [
  { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
  { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }] });
const pipeV2 = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [v2bgl] }), vertex: { module: mV2, entryPoint: 'vs' },
  fragment: { module: mV2, entryPoint: 'fs', targets: [{ format: FORMAT }] }, primitive: { topology: 'triangle-list' } });
const mV4 = device.createShaderModule({ code: DRAWV4 });
mV4.getCompilationInfo().then((ci) => { for (const m of ci.messages) if (m.type === 'error') msg(`V4 shader line ${m.lineNum}: ${m.message}`); });
const v4bgl = device.createBindGroupLayout({ entries: [
  { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
  { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
  { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }] });
const pipeV4 = device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [v4bgl] }), vertex: { module: mV4, entryPoint: 'vs' },
  fragment: { module: mV4, entryPoint: 'fs', targets: [{ format: FORMAT }] }, primitive: { topology: 'triangle-list' } });
let low = null;     // low-res target, rebuilt when the pixel size changes
function ensureLow(ps) {
  if (low && low.ps === ps) return low;
  const w = Math.ceil(OW / ps), h = Math.ceil(OH / ps);
  const tex = device.createTexture({ size: [w, h], format: FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const ub = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(ub, 0, new Float32Array([ps, 0, 0, 0]));
  const view = tex.createView();
  low = { ps, w, h, tex, view, blitBG: device.createBindGroup({ layout: pipeBlit.getBindGroupLayout(0), entries: [{ binding: 0, resource: view }, { binding: 1, resource: { buffer: ub } }] }) };
  return low;
}
const msaa = device.createTexture({ size: [OW, OH], sampleCount: 4, format: FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
const msaaView = msaa.createView();

// ---------------------------------------------------------------- buffers
const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CSRC = GPUBufferUsage.COPY_SRC;
const mk = (size, usage, label) => device.createBuffer({ size: Math.max(16, (size + 3) & ~3), usage: usage | GPUBufferUsage.COPY_SRC, label });
const F32 = SW * SH * 4;
const B = {
  u: mk(64, GPUBufferUsage.UNIFORM | CD),
  gray: mk(F32, S), keyr: mk(F32, S), keyA: mk(F32, S), keyB: mk(F32, S), keyC: mk(F32, S),
  lut: mk(36 * 256 * 4, S), hist2: mk(256 * 4, S | CD), stretch: mk(16, S),
  cl: mk(F32, S), dg: mk(F32, S), dtmp: mk(F32, S), dgb: mk(F32, S),
  mask: mk(F32, S | CD), hS: mk(F32, S), hL: mk(F32, S), v2u: mk(96, GPUBufferUsage.UNIFORM | CD), v4u: mk(96, GPUBufferUsage.UNIFORM | CD), state: mk(F32, S), R: mk(F32, S | CD), wake: mk(F32, S), wakeT: mk(F32, S), wakeD: mk(F32, S),
  na: mk(NGW * NGH * 4, S | CD), nb: mk(NGW * NGH * 4, S | CD),
  meta: mk(16, S | CD | CSRC), args: mk(48, S | GPUBufferUsage.INDIRECT | CD),
  pos: mk(MAXN * 8, S | CD), mv: mk(MAXN * 16, S), head: mk(GW * GH * 4, S), nxt: mk(MAXN * 4, S),
  cnt: mk(MAXN * 4, S), scanOut: mk(MAXN * 4, S), blockOff: mk(2048, S),
  s1: mk(MAXN * 8, S), s2: mk(MAXN * 8, S), s3: mk(MAXN * 8, S),
  du: mk(48, GPUBufferUsage.UNIFORM | CD), du8: mk(48, GPUBufferUsage.UNIFORM | CD),
};
const bu = (a) => { const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | CD }); device.queue.writeBuffer(b, 0, new Float32Array(a)); return b; };
const BU = {
  keyH4: bu([4, 1, 12, 0]), keyV4: bu([4, 0, 12, 0]), keyH25: bu([2.5, 1, 8, 1]), keyV25: bu([2.5, 0, 8, 0]),
  hsH: bu([6, 1, 18, 1]), hsV: bu([6, 0, 18, 0]), hlH: bu([20, 1, 60, 1]), hlV: bu([20, 0, 60, 0]),
  darkH: bu([3.5, 1, 11, 0]), darkV: bu([3.5, 0, 11, 0]), dilH: bu([0, 1, 18, 0]), dilV: bu([0, 0, 18, 0]),
};
const srcTex = device.createTexture({ size: [SW, SH], format: 'rgba8unorm',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });

function bg(k, res) {
  return device.createBindGroup({ layout: k.bgl, entries: res.map((r, i) => ({ binding: i, resource: r instanceof GPUBuffer ? { buffer: r } : r })) });
}
const G = {
  prep: bg(K.prep, [B.u, srcTex.createView(), B.gray, B.keyr]),
  keyH4: bg(K.blur, [B.keyr, B.keyA, BU.keyH4]), keyV4: bg(K.blur, [B.keyA, B.keyB, BU.keyV4]),
  keyH25: bg(K.blur, [B.keyB, B.keyA, BU.keyH25]), keyV25: bg(K.blur, [B.keyA, B.keyC, BU.keyV25]),
  keymix: bg(K.keymix, [B.u, B.mask, B.keyC]),
  tilehist: bg(K.tilehist, [B.u, B.gray, B.lut]),
  clahe: bg(K.claheapply, [B.gray, B.lut, B.keyC, B.cl, B.hist2]),
  pct: bg(K.percentile, [B.hist2, B.stretch]),
  darkprep: bg(K.darkprep, [B.cl, B.stretch, B.keyC, B.dg]),
  darkH: bg(K.blur, [B.dg, B.dtmp, BU.darkH]), darkV: bg(K.blur, [B.dtmp, B.dgb, BU.darkV]),
  rfield: bg(K.rfield, [B.u, B.dgb, B.state, B.R, B.wake, B.na, B.nb]),
  dilH: bg(K.dilate, [B.wake, B.wakeT, BU.dilH]), dilV: bg(K.dilate, [B.wakeT, B.wakeD, BU.dilV]),
  hsH: bg(K.blur, [B.mask, B.dtmp, BU.hsH]), hsV: bg(K.blur, [B.dtmp, B.hS, BU.hsV]),
  hlH: bg(K.blur, [B.mask, B.dtmp, BU.hlH]), hlV: bg(K.blur, [B.dtmp, B.hL, BU.hlV]),
  gridclear: bg(K.gridclear, [B.head]),
  gridbuild: bg(K.gridbuild, [B.meta, B.pos, B.head, B.nxt]),
  forces: bg(K.forces, [B.u, B.meta, B.pos, B.head, B.nxt, B.R, B.wakeD, B.mv]),
  decide: bg(K.decide, [B.meta, B.mv, B.R, B.cnt, B.u]),
  scan1: bg(K.scan1, [B.meta, B.cnt, B.scanOut, B.blockOff]),
  scan2: bg(K.scan2, [B.meta, B.blockOff]),
  compact: bg(K.compact, [B.meta, B.mv, B.cnt, B.scanOut, B.blockOff, B.pos]),
  finalize: bg(K.finalize, [B.meta, B.args]),
  sm1: bg(K.smooth, [B.meta, B.pos, B.s1]), sm2: bg(K.smooth, [B.meta, B.s1, B.s2]), sm3: bg(K.smooth, [B.meta, B.s2, B.s3]),
};
const drawBG = device.createBindGroup({ layout: drawPipe.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: B.du } }, { binding: 1, resource: { buffer: B.s3 } }, { binding: 2, resource: { buffer: B.meta } }] });

const drawV2BG = device.createBindGroup({ layout: v2bgl,
  entries: [{ binding: 0, resource: { buffer: B.v2u } }, { binding: 1, resource: { buffer: B.state } }, { binding: 2, resource: { buffer: B.hS } }, { binding: 3, resource: { buffer: B.hL } }, { binding: 4, resource: { buffer: B.mask } }] });
const drawV4BG = device.createBindGroup({ layout: v4bgl, entries: [{ binding: 0, resource: { buffer: B.v4u } }, { binding: 1, resource: { buffer: B.state } }, { binding: 2, resource: { buffer: B.mask } }] });
const draw8BG = device.createBindGroup({ layout: pipe8.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: B.du8 } }, { binding: 1, resource: { buffer: B.s3 } }, { binding: 2, resource: { buffer: B.meta } }, { binding: 3, resource: { buffer: B.state } }] });

// ---------------------------------------------------------------- timestamps + readback
const CAP = 480;
const qs = hasTS ? device.createQuerySet({ type: 'timestamp', count: CAP }) : null;
const qbuf = hasTS ? device.createBuffer({ size: CAP * 8, usage: GPUBufferUsage.QUERY_RESOLVE | CSRC }) : null;
const ring = Array.from({ length: 3 }, () => ({ buf: device.createBuffer({ size: CAP * 8 + 16, usage: GPUBufferUsage.MAP_READ | CD }), busy: false }));
let qn = 0, qlabels = [], timing = true;

function cpass(enc, label) {
  const d = {};
  if (hasTS && timing && qn + 2 <= CAP) {
    d.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: qn, endOfPassWriteIndex: qn + 1 };
    qlabels.push([label, qn]); qn += 2;
  }
  return enc.beginComputePass(d);
}
function run(enc, k, bgrp, groups, label) {
  const p = cpass(enc, label);
  p.setPipeline(K[k].pipe); p.setBindGroup(0, bgrp);
  if (Array.isArray(groups)) p.dispatchWorkgroups(...groups); else p.dispatchWorkgroupsIndirect(B.args, groups);
  p.end();
}
const IND_PTS = 0, IND_SCAN = 16;

// ---------------------------------------------------------------- state
const P = { rmin: 3.05, rmax: 13.75, gain: 1.75, noise: 0.22, iters: 6, stroke: 4.75, key: false, mirror: true, subject: 1, px: 6, v2s: 16, v2r: 70, v2a: 0.08, v2b: 0.9, v2m: 0, v2mode: 0, v2o: 4, v2g: 3, v3s: 7, v3r: 37, v3a: 0.035, v3b: 0.45, v3d: 8, v3z: 0.22, v3x: 0.37, v3y: 0, v3k: 1, v3w: 1, v3m: 0.6, smodel: LAND ? 'rvm' : 'mp', rvmRes: 640, rvmRatio: 0.5, v4s: 22, v4n: 4, v4l: 4, v4d: 0.75, v4f: 0.25, v4w: 1.2, v4b: 0, v4a: 0.35, v5i: [20, 20, 22], v5p: [240, 238, 234], v5ax: 0, v5col: 1, v4i: [20, 20, 22], v4p: [240, 238, 234], v3int: 1, v3p: 6, v3res: 1536, v3auto: true, v3c: [215, 212, 212],
  fg: [210, 50, 25], bgc: [242, 240, 238] };
const sim = { frame: 0, time: 0, boost: 0, inflight: 0, pending: 0, running: true, view: 'gpu', dropped: 0 };
const U = new Float32Array(16);

function writeU(fullWake, newW) {
  U.set([sim.time, P.rmin, P.rmax, P.gain, P.noise, fullWake ? 1 : 0, 0.30, 0.6, P.key ? 1 : 0, newW, 2.5, seg.ready && sim.view !== 'v2' && sim.view !== 'v3' ? P.subject : 0]);
  device.queue.writeBuffer(B.u, 0, U);
}

// seeded RNG
function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()); }

function hilbert(n, d) {
  let x = 0, y = 0, s = 1, t = d;
  while (s < n) {
    const rx = 1 & Math.floor(t / 2), ry = 1 & (t ^ rx);
    if (ry === 0) { if (rx === 1) { x = s - 1 - x; y = s - 1 - y; } [x, y] = [y, x]; }
    x += s * rx; y += s * ry; t = Math.floor(t / 4); s *= 2;
  }
  return [x, y];
}
function mooreStart() {
  const n = 32, h = []; for (let d = 0; d < n * n; d++) h.push(hilbert(n, d));
  const q = [...h.map(([x, y]) => [n - 1 - y, x]), ...h.map(([x, y]) => [n - 1 - y, x + n]),
    ...h.map(([x, y]) => [y + n, 2 * n - 1 - x]), ...h.map(([x, y]) => [y + n, n - 1 - x])];
  let minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
  for (const [x, y] of q) { minx = Math.min(minx, x); miny = Math.min(miny, y); maxx = Math.max(maxx, x); maxy = Math.max(maxy, y); }
  const pts = q.map(([x, y]) => [MARGIN + 10 + (x - minx) / (maxx - minx) * (SW - 2 * MARGIN - 20), MARGIN + 10 + (y - miny) / (maxy - miny) * (SH - 2 * MARGIN - 20)]);
  const r = mulberry(3), out = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    const k = Math.max(1, Math.floor(Math.hypot(b[0] - a[0], b[1] - a[1]) / 3));
    for (let t = 0; t < k; t++) out.push(a[0] + (b[0] - a[0]) * t / k + gauss(r) * 0.6, a[1] + (b[1] - a[1]) * t / k + gauss(r) * 0.6);
  }
  return new Float32Array(out);
}
function noiseField(seed) {
  const r = mulberry(seed), a = new Float32Array(NGW * NGH);
  for (let i = 0; i < a.length; i++) a[i] = gauss(r);
  const sg = 45 / 8, rad = Math.ceil(sg * 3), w = [];
  for (let k = -rad; k <= rad; k++) w.push(Math.exp(-k * k / (2 * sg * sg)));
  const blur = (src, horiz) => {
    const dst = new Float32Array(src.length);
    for (let y = 0; y < NGH; y++) for (let x = 0; x < NGW; x++) {
      let acc = 0, ws = 0;
      for (let k = -rad; k <= rad; k++) {
        const xx = horiz ? Math.min(Math.max(x + k, 0), NGW - 1) : x, yy = horiz ? y : Math.min(Math.max(y + k, 0), NGH - 1);
        acc += w[k + rad] * src[yy * NGW + xx]; ws += w[k + rad];
      }
      dst[y * NGW + x] = acc / ws;
    }
    return dst;
  };
  const f = blur(blur(a, true), false);
  let m = 1e-9; for (const v of f) m = Math.max(m, Math.abs(v));
  return f.map((v) => v / m);
}
device.queue.writeBuffer(B.na, 0, noiseField(11));
device.queue.writeBuffer(B.nb, 0, noiseField(12));

function resetSim() {
  const p0 = mooreStart(), n = p0.length / 2;
  device.queue.writeBuffer(B.pos, 0, p0);
  device.queue.writeBuffer(B.meta, 0, new Uint32Array([n, 0, n, 0]));
  device.queue.writeBuffer(B.args, 0, new Uint32Array([Math.ceil(n / 64), 1, 1, 0, Math.ceil(n / 512), 1, 1, 0, 12, n, 0, 0]));
  sim.frame = 0; sim.time = 0; sim.boost = 0; sim.warmed = false;
}

// ---------------------------------------------------------------- video source
const vid = $('#vid');
let camStream = null;
const cv2 = document.createElement('canvas'); cv2.width = SW; cv2.height = SH;
const c2 = cv2.getContext('2d', { willReadFrequently: false });
c2.imageSmoothingQuality = 'high';
const vmap = { au: 1 / SW, bu: 0, av: 1 / SH, bv: 0 };   // buffer pixel -> normalised video coordinates
const rsOut = new Float32Array(SW * SH);
function resampleMask(a, mw, mh) {                       // bilinear, in video space; outside the video frame the mask is 0
  const xs = new Float32Array(SW), xi = new Int32Array(SW);
  for (let x = 0; x < SW; x++) {
    const u = vmap.au * (x + 0.5) + vmap.bu;
    if (u < 0 || u >= 1) { xi[x] = -1; continue; }
    const fx = Math.min(mw - 1.001, Math.max(0, u * mw - 0.5)); xi[x] = fx | 0; xs[x] = fx - (fx | 0);
  }
  for (let y = 0; y < SH; y++) {
    const v = vmap.av * (y + 0.5) + vmap.bv, row = y * SW;
    if (v < 0 || v >= 1) { rsOut.fill(0, row, row + SW); continue; }
    const fy = Math.min(mh - 1.001, Math.max(0, v * mh - 0.5)), y0 = fy | 0, ty = fy - y0, r0 = y0 * mw, r1 = r0 + mw;
    for (let x = 0; x < SW; x++) {
      const x0 = xi[x]; if (x0 < 0) { rsOut[row + x] = 0; continue; }
      const tx = xs[x];
      rsOut[row + x] = (a[r0 + x0] * (1 - tx) + a[r0 + x0 + 1] * tx) * (1 - ty) + (a[r1 + x0] * (1 - tx) + a[r1 + x0 + 1] * tx) * ty;
    }
  }
  return rsOut;
}
// the video as it is (native aspect), downscaled for the models
const mcv = document.createElement('canvas'), mctx = mcv.getContext('2d', { willReadFrequently: true });
const still = { snap: document.createElement('canvas'), busy: false, last: 0, ms: 0, have: false };
const srcEl = () => (sim.view === 'v3' && still.have ? still.snap : vid);
const dimW = (e) => e.videoWidth || e.width, dimH = (e) => e.videoHeight || e.height;
function modelFrame(longSide, mult = 8) {
  const el = srcEl(), vw = dimW(el), vh = dimH(el), k = longSide / Math.max(vw, vh);
  const w = Math.max(mult, Math.round(vw * k / mult) * mult), h = Math.max(mult, Math.round(vh * k / mult) * mult);
  if (mcv.width !== w || mcv.height !== h) { mcv.width = w; mcv.height = h; }
  mctx.drawImage(el, 0, 0, w, h);
  return { w, h };
}
function uploadFrame() {
  const el = srcEl(), vw = dimW(el), vh = dimH(el), ar = SW / SH;
  let sx = 0, sy = 0, sw = vw, sh = vh;
  if (vw / vh > ar) { sw = vh * ar; sx = (vw - sw) / 2; } else { sh = vw / ar; sy = (vh - sh) / 2; }
  c2.save();
  const mir = !!(camStream && P.mirror);   // (a snapshot is stored unflipped too)
  if (mir) { c2.translate(SW, 0); c2.scale(-1, 1); }
  let dx, dy, dw, dh;                                   // where the whole video frame lands in the sim buffer
  if (LAND) {                                           // whole camera frame visible at zoom 1, grey fill around it
    const sc = Math.min(SW / vw, SH / vh) * P.v3w; dw = vw * sc; dh = vh * sc; dx = (SW - dw) / 2; dy = (SH - dh) / 2;
    c2.fillStyle = '#808080'; c2.fillRect(0, 0, SW, SH);
    c2.drawImage(el, 0, 0, vw, vh, dx, dy, dw, dh);
  } else { c2.drawImage(el, sx, sy, sw, sh, 0, 0, SW, SH); dw = vw * SW / sw; dh = vh * SH / sh; dx = -sx * SW / sw; dy = -sy * SH / sh; }
  c2.restore();
  vmap.au = (mir ? -1 : 1) / dw; vmap.bu = mir ? (SW - dx) / dw : -dx / dw; vmap.av = 1 / dh; vmap.bv = -dy / dh;   // buffer px -> video uv
  device.queue.copyExternalImageToTexture({ source: cv2 }, { texture: srcTex }, [SW, SH]);
  segment();
}


// ---------------------------------------------------------------- subject separation (MediaPipe selfie segmenter)
const seg = { ready: false, loading: false, obj: null, ms: 0, ts: 0, w: 0, h: 0, delegate: '' };
async function loadSegmenter() {
  if (seg.ready || seg.loading) return;
  seg.loading = true; msg('Loading segmenter...');
  try {
    const vision = await import('./vendor/vision_bundle.mjs');
    const b64 = (await import('./vendor/model_b64.js')).default;
    const modelBytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const fileset = await vision.FilesetResolver.forVisionTasks('vendor/wasm');
    for (const delegate of ['GPU', 'CPU']) {
      try {
        seg.obj = await vision.ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetBuffer: modelBytes, delegate },
          runningMode: 'VIDEO', outputConfidenceMasks: true, outputCategoryMask: false });
        seg.delegate = delegate; break;
      } catch (e) { if (delegate === 'CPU') throw e; }
    }
    seg.ready = true; msg('');
  } catch (e) { msg('Segmenter failed: ' + e.message); P.subject = 0; $('#subject').value = '0'; }
  seg.loading = false;
}
const box = { x0: 0, x1: SW, y0: 0, y1: SH, init: false };   // smoothed bounding box of the subject, sim px
function trackBox(m) {
  const cols = new Uint16Array(SW), rows = new Uint16Array(SH);
  for (let y = 0; y < SH; y += 2) for (let x = 0; x < SW; x += 2) if (m[y * SW + x] > 0.5) { cols[x]++; rows[y]++; }
  let x0 = -1, x1 = -1, y0 = -1, y1 = -1;
  for (let x = 0; x < SW; x += 2) if (cols[x] > 2) { if (x0 < 0) x0 = x; x1 = x; }
  for (let y = 0; y < SH; y += 2) if (rows[y] > 2) { if (y0 < 0) y0 = y; y1 = y; }
  if (x0 < 0 || x1 - x0 < 40 || y1 - y0 < 40) return;
  const k = box.init ? 0.12 : 1; box.init = true;
  box.x0 += (x0 - box.x0) * k; box.x1 += (x1 - box.x1) * k; box.y0 += (y0 - box.y0) * k; box.y1 += (y1 - box.y1) * k;
}
function pushMask(a) {
  if (sim.view === 'v3') { trackBox(a); } else if (sim.view === 'v2' || sim.view === 'v4') {            // temporal smoothing of the subject mask: kills the flicker
    if (!seg.cur) { seg.cur = new Float32Array(SW * SH); seg.cur.set(a); }
    const s = P.smodel === 'rvm' ? P.v3m * 0.5 : P.v3m, n = SW * SH, cur = seg.cur;   // the matting model is already temporally stable
    for (let i = 0; i < n; i++) cur[i] = cur[i] * s + a[i] * (1 - s);
    a = cur;
  }
  device.queue.writeBuffer(B.mask, 0, a);
}

// ---------------------------------------------------------------- Robust Video Matting (onnxruntime-web, recurrent, WebGPU)
const rvm = { ready: false, loading: false, busy: false, sess: null, ort: null, st: null, ms: 0, ep: '' };
async function loadRvm() {
  if (rvm.ready || rvm.loading) return;
  rvm.loading = true; msg('Loading matting model (about 37 MB, first time only)...');
  try {
    const ort = await import('./vendor/ort/ort.webgpu.min.mjs');
    ort.env.wasm.wasmPaths = new URL('./vendor/ort/', location.href).href;
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    let sess = null;
    for (const ep of ['webgpu', 'wasm']) {
      try { sess = await ort.InferenceSession.create('vendor/rvm_mobilenetv3_fp32.onnx', { executionProviders: [ep] }); rvm.ep = ep; break; }
      catch (e) { if (ep === 'wasm') throw e; }
    }
    rvm.ort = ort; rvm.sess = sess;
    rvm.zero = () => new ort.Tensor('float32', new Float32Array(1), [1, 1, 1, 1]);
    rvm.st = [rvm.zero(), rvm.zero(), rvm.zero(), rvm.zero()];
    rvm.ratio = new ort.Tensor('float32', new Float32Array([P.rvmRatio]), [1]);
    rvm.ready = true; msg('');
  } catch (e) { msg('Matting model failed: ' + e.message); P.smodel = 'mp'; $('#smodel').value = 'mp'; }
  rvm.loading = false;
}
async function runRvm(res) {                           // one matting pass on the current source; returns the resampled mask
  const { ort } = rvm;
  const { w, h } = modelFrame(res, 32);
  if (rvm.dims !== w + 'x' + h) { rvm.dims = w + 'x' + h; rvm.st = [rvm.zero(), rvm.zero(), rvm.zero(), rvm.zero()]; }
  const d = mctx.getImageData(0, 0, w, h).data, n = w * h, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { f[i] = d[i * 4] / 255; f[n + i] = d[i * 4 + 1] / 255; f[2 * n + i] = d[i * 4 + 2] / 255; }
  const o = await rvm.sess.run({ src: new ort.Tensor('float32', f, [1, 3, h, w]), r1i: rvm.st[0], r2i: rvm.st[1], r3i: rvm.st[2], r4i: rvm.st[3], downsample_ratio: rvm.ratio });
  rvm.st = [o.r1o, o.r2o, o.r3o, o.r4o];
  const p = o.pha.data ?? await o.pha.getData();
  return resampleMask(p, w, h);
}
function rvmKick() {
  if (!rvm.ready || rvm.busy) return;
  rvm.busy = true;
  const t0 = performance.now(), { ort } = rvm;
  const { w, h } = modelFrame(P.rvmRes, 32);
  if (rvm.dims !== w + 'x' + h) { rvm.dims = w + 'x' + h; rvm.st = [rvm.zero(), rvm.zero(), rvm.zero(), rvm.zero()]; }
  const d = mctx.getImageData(0, 0, w, h).data, n = w * h, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { f[i] = d[i * 4] / 255; f[n + i] = d[i * 4 + 1] / 255; f[2 * n + i] = d[i * 4 + 2] / 255; }
  rvm.sess.run({ src: new ort.Tensor('float32', f, [1, 3, h, w]), r1i: rvm.st[0], r2i: rvm.st[1], r3i: rvm.st[2], r4i: rvm.st[3], downsample_ratio: rvm.ratio })
    .then(async (o) => {
      rvm.st = [o.r1o, o.r2o, o.r3o, o.r4o];
      const p = o.pha.data ?? await o.pha.getData();
      pushMask(resampleMask(p, w, h)); rvm.ms = performance.now() - t0; seg.ms = rvm.ms;
    })
    .catch((e) => { msg('Matting run failed: ' + e.message); P.smodel = 'mp'; $('#smodel').value = 'mp'; })
    .finally(() => { rvm.busy = false; });
}
async function stillCycle() {
  const vw = vid.videoWidth, vh = vid.videoHeight;
  if (still.busy || vid.readyState < 2 || !vw) return;
  still.busy = true; still.last = performance.now();
  const t0 = performance.now(); let tMat = 0;
  try {
    if (still.snap.width !== vw || still.snap.height !== vh) { still.snap.width = vw; still.snap.height = vh; }
    still.snap.getContext('2d').drawImage(vid, 0, 0, vw, vh); still.have = true;   // freeze this frame
    uploadFrame();                                       // also fixes the video -> buffer mapping for the mask
    const t1 = performance.now();
    if (P.smodel === 'rvm' && rvm.ready) {
      rvm.st = [rvm.zero(), rvm.zero(), rvm.zero(), rvm.zero()];   // same still frame several times so the recurrent state settles
      let a = null;
      for (let i = 0; i < P.v3p; i++) a = await runRvm(P.v3res);
      box.init = false; pushMask(a);
    } else if (seg.ready) {
      modelFrame(1024);
      seg.ts = Math.max(seg.ts + 1, Math.round(performance.now()));
      const r = seg.obj.segmentForVideo(mcv, seg.ts), m = r.confidenceMasks[0];
      box.init = false; pushMask(resampleMask(m.getAsFloat32Array(), m.width, m.height));
      m.close(); r.close && r.close();
    }
    tMat = performance.now() - t1;
    sim.frame = 0;                                       // first-frame behaviour: no temporal blending of the tone field
    await tick({ full: true });
    still.ms = performance.now() - t0;
    $('#stillinfo').textContent = `Capture: ${still.ms.toFixed(0)} ms total (matting ${tMat.toFixed(0)} ms, ${P.smodel === 'rvm' ? P.v3p + ' passes at ' + P.v3res + ' px' : '1 pass'}). ${rvm.ep ? 'Model runs on ' + rvm.ep + '.' : ''}`;
  } catch (e) { msg('Capture failed: ' + e.message); }
  still.busy = false;
}
function stillTick() {
  if (still.busy || !sim.running || benchRunning) return;
  const iv = P.v3int * 1000;
  if (P.v3int > 0 && performance.now() - still.last >= iv) stillCycle();
}
$('#capnow').onclick = () => stillCycle();
function segment() {
  if (sim.view === 'v3') return;                         // V3 captures stills, see stillCycle
  if (P.smodel === 'rvm' && (sim.view === 'v2' || sim.view === 'v3' || sim.view === 'v4')) { rvmKick(); return; }
  if (!seg.ready || (P.subject === 0 && sim.view !== 'v2' && sim.view !== 'v3' && sim.view !== 'v4')) { seg.ms = 0; return; }
  const t0 = performance.now();
  seg.ts = Math.max(seg.ts + 1, Math.round(performance.now()));
  modelFrame(512);
  const r = seg.obj.segmentForVideo(mcv, seg.ts);
  const m = r.confidenceMasks[0];
  const a = resampleMask(m.getAsFloat32Array(), m.width, m.height);
  pushMask(a);
  seg.w = m.width; seg.h = m.height;
  m.close(); r.close && r.close();
  seg.ms = performance.now() - t0;
}

// ---------------------------------------------------------------- one frame
function encodeField(enc) {
  enc.clearBuffer(B.hist2);
  run(enc, 'prep', G.prep, [Math.ceil(SW / 16), Math.ceil(SH / 16)], 'field: gray + red key');
  const im = [Math.ceil(SW / 16), Math.ceil(SH / 16)];
  for (const g of ['keyH4', 'keyV4', 'keyH25', 'keyV25']) run(enc, 'blur', G[g], im, 'field: key blur');
  run(enc, 'keymix', G.keymix, [Math.ceil(SW * SH / 64)], 'field: key blur');
  run(enc, 'tilehist', G.tilehist, [6, 6], 'field: CLAHE');
  run(enc, 'claheapply', G.clahe, im, 'field: CLAHE');
  run(enc, 'percentile', G.pct, [1], 'field: stretch');
  run(enc, 'darkprep', G.darkprep, [Math.ceil(SW * SH / 64)], 'field: stretch');
  run(enc, 'blur', G.darkH, im, 'field: darkness blur'); run(enc, 'blur', G.darkV, im, 'field: darkness blur');
  run(enc, 'rfield', G.rfield, im, 'field: spacing R + wake');
  run(enc, 'dilate', G.dilH, im, 'field: spacing R + wake'); run(enc, 'dilate', G.dilV, im, 'field: spacing R + wake');
}
function encodeStep(enc) {
  run(enc, 'gridclear', G.gridclear, [Math.ceil(GW * GH / 64)], 'sim: grid');
  run(enc, 'gridbuild', G.gridbuild, IND_PTS, 'sim: grid');
  run(enc, 'forces', G.forces, IND_PTS, 'sim: forces');
  run(enc, 'decide', G.decide, IND_PTS, 'sim: prune/grow decide');
  run(enc, 'scan1', G.scan1, IND_SCAN, 'sim: scan');
  run(enc, 'scan2', G.scan2, [1], 'sim: scan');
  run(enc, 'compact', G.compact, IND_PTS, 'sim: compact + insert');
  run(enc, 'finalize', G.finalize, [1], 'sim: compact + insert');
}
function encodeDraw(enc) {
  run(enc, 'smooth', G.sm1, IND_PTS, 'draw: smooth'); run(enc, 'smooth', G.sm2, IND_PTS, 'draw: smooth'); run(enc, 'smooth', G.sm3, IND_PTS, 'draw: smooth');
  if (sim.view === 'bit8') { drawBit8(enc); return; }
  const dv = new Float32Array([SCALE, P.stroke / 2, OW, OH, P.fg[0] / 255, P.fg[1] / 255, P.fg[2] / 255, 1, 0, 3, 0, 0]);
  device.queue.writeBuffer(B.du, 0, dv);
  const d = {
    colorAttachments: [{ view: msaaView, resolveTarget: ctx.getCurrentTexture().createView(),
      clearValue: { r: P.bgc[0] / 255, g: P.bgc[1] / 255, b: P.bgc[2] / 255, a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
  };
  if (hasTS && timing && qn + 2 <= CAP) { d.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: qn, endOfPassWriteIndex: qn + 1 }; qlabels.push(['draw: render', qn]); qn += 2; }
  const p = enc.beginRenderPass(d);
  p.setPipeline(drawPipe); p.setBindGroup(0, drawBG); p.drawIndirect(B.args, 32); p.end();
}



const BW = 2373, BH = 1012;                           // V3 100$ canvas = the bill image
function billPanel() {
  if (!P.v3auto || !box.init) {
    const ph = P.v3k*BH, z = ph/SH;
    return [P.v3x*BW - SW*z/2, (BH - ph)/2 + P.v3y*BH, z, 0];
  }
  // auto fit: scale the subject's bounding box into a target area of the bill, bottom aligned, centred on X
  const th = 0.85*P.v3k*BH, tw = 0.34*P.v3k*BW;
  const bw = Math.max(box.x1 - box.x0, 40), bh = Math.max(box.y1 - box.y0, 40);
  const z = Math.min(tw/bw, th/bh);
  const cx = P.v3x*BW, bottom = BH*(0.5 + P.v3y) + th/2;
  return [cx - (box.x0 + bw/2)*z, bottom - box.y1*z, z, 0];
}
function encodeV2(enc, bill = false) {
  const im = [Math.ceil(SW / 16), Math.ceil(SH / 16)];
  run(enc, 'blur', G.hsH, im, 'v2: subject shape'); run(enc, 'blur', G.hsV, im, 'v2: subject shape');
  run(enc, 'blur', G.hlH, im, 'v2: subject shape'); run(enc, 'blur', G.hlV, im, 'v2: subject shape');
  const rgb = (c) => c.map((x) => x / 255);
  const u = bill
    ? [BW, BH, P.v3s, P.v3r, P.v2m, P.v3a, P.v3b, P.v3z, 0, 0, 0, 1, ...P.v3c.map((x) => x / 255), P.v3d, ...billPanel()]
    : [OW, OH, P.v2s, P.v2r, P.v2mode, P.v2a, P.v2b, P.v2o, ...rgb(P.fg), 0, ...rgb(P.bgc), P.v2g, 0, 0, 1, 0];
  device.queue.writeBuffer(B.v2u, 0, new Float32Array(u));
  const d = { colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: bill ? { r: 0, g: 0, b: 0, a: 0 } : { r: 1, g: 1, b: 1, a: 1 } }] };
  if (hasTS && timing && qn + 2 <= CAP) { d.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: qn, endOfPassWriteIndex: qn + 1 }; qlabels.push(['draw: render', qn]); qn += 2; }
  const p = enc.beginRenderPass(d);
  p.setPipeline(pipeV2); p.setBindGroup(0, drawV2BG); p.draw(3); p.end();
}


let v4T = 0, v4L = performance.now();
function encodeV4(enc) {
  const nowT = performance.now(); v4T += Math.min(0.1, (nowT - v4L)/1000)*(P.v4y ?? 1.5); v4L = nowT;
  const rgb = (c) => c.map((x) => x / 255);
  device.queue.writeBuffer(B.v4u, 0, new Float32Array([OW, OH, P.v4s, P.v4n, P.v4l, P.v4d, P.v4f, P.v4a, ...rgb(P.v4i), P.v4b, ...rgb(P.v4p), P.v4w, P.v4k, P.v4m, P.v4r, P.v4c, P.v4q, P.v4j, P.v4z, v4T]));
  const d = { colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 1, b: 1, a: 1 } }] };
  if (hasTS && timing && qn + 2 <= CAP) { d.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: qn, endOfPassWriteIndex: qn + 1 }; qlabels.push(['draw: render', qn]); qn += 2; }
  const p = enc.beginRenderPass(d);
  p.setPipeline(pipeV4); p.setBindGroup(0, drawV4BG); p.draw(3); p.end();
}

function drawBit8(enc) {
  const L = ensureLow(P.px);
  device.queue.writeBuffer(B.du8, 0, new Float32Array([SCALE / P.px, 0.6, L.w, L.h, 0, 0, 0, 1, 0, 0, 0, 0]));
  const d = { colorAttachments: [{ view: L.view, clearValue: { r: 0.608, g: 0.737, b: 0.059, a: 1 }, loadOp: 'clear', storeOp: 'store' }] };
  if (hasTS && timing && qn + 2 <= CAP) { d.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: qn, endOfPassWriteIndex: qn + 1 }; qlabels.push(['draw: render', qn]); qn += 2; }
  let p = enc.beginRenderPass(d);
  p.setPipeline(pipe8); p.setBindGroup(0, draw8BG); p.drawIndirect(B.args, 32); p.end();
  p = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
  p.setPipeline(pipeBlit); p.setBindGroup(0, L.blitBG); p.draw(3); p.end();
}

let tsQuant = null;   // true if timestamps look quantized to 100 us
const lastStats = { labels: {}, span: 0, sum: 0, n: 0, cpuUpload: 0, cpuSubmit: 0, quant: false };

async function tick({ full = false, steps = null, wait = false } = {}) {
  const t0 = performance.now();
  if (vid.readyState >= 2) uploadFrame();
  const tUp = performance.now() - t0;
  const first = sim.frame === 0;
  const fullWake = first || full || sim.boost > 0 || sim.frame % 15 === 0;
  writeU(fullWake, first ? 1 : 0.3);
  const t1 = performance.now();
  qn = 0; qlabels = [];
  const enc = device.createCommandEncoder();
  timing = true;
  encodeField(enc);
  if (sim.view === 'v2' || sim.view === 'v3') { encodeV2(enc, sim.view === 'v3'); return finishTick(enc, tUp, t1, false); }
  if (sim.view === 'v4') { encodeV4(enc); return finishTick(enc, tUp, t1, false); }
  const nIter = steps ?? (sim.boost > 0 ? P.iters * 2 : P.iters);
  if (!sim.warmed) {
    sim.warmed = true;
    // warm-up: untimed passes, run in a few submits so the encoder stays small
    timing = false;
    device.queue.submit([enc.finish()]);
    for (let b = 0; b < 3; b++) {
      const e2 = device.createCommandEncoder();
      for (let s = 0; s < WARM_STEPS / 3; s++) encodeStep(e2);
      device.queue.submit([e2.finish()]);
    }
    const e3 = device.createCommandEncoder(); timing = true; qn = 0; qlabels = [];
    encodeDraw(e3);
    return finishTick(e3, tUp, t1, true);
  }
  for (let s = 0; s < nIter; s++) encodeStep(enc);
  encodeDraw(enc);
  return finishTick(enc, tUp, t1, false);
}

async function finishTick(enc, tUp, t1, warm) {
  sim.frame++; sim.time += 1 / FPS; if (sim.boost > 0) sim.boost--;
  const slot = ring.find((r) => !r.busy);
  let labels = qlabels, nq = qn;
  if (hasTS && slot && nq > 0) {
    enc.resolveQuerySet(qs, 0, nq, qbuf, 0);
    enc.copyBufferToBuffer(qbuf, 0, slot.buf, 0, nq * 8);
  }
  if (slot) enc.copyBufferToBuffer(B.meta, 0, slot.buf, CAP * 8, 16);
  device.queue.submit([enc.finish()]);
  sim.pending++; device.queue.onSubmittedWorkDone().then(() => { sim.pending--; });
  const cpuSubmit = performance.now() - t1;
  if (!slot) return null;
  slot.busy = true;
  await slot.buf.mapAsync(GPUMapMode.READ);
  const raw = slot.buf.getMappedRange();
  const meta = new Uint32Array(raw.slice(CAP * 8, CAP * 8 + 16));
  const res = { n: meta[0], cpuSeg: seg.ms, cpuUpload: tUp, cpuSubmit, warm, labels: {}, span: 0, sum: 0 };
  if (hasTS && nq > 0) {
    const ts = new BigUint64Array(raw.slice(0, nq * 8));
    let lo = ts[0], hi = ts[1], q100 = true;
    for (const [lab, qi] of labels) {
      const a = ts[qi], b = ts[qi + 1];
      if (a < lo) lo = a; if (b > hi) hi = b;
      if (a % 100000n !== 0n || b % 100000n !== 0n) q100 = false;
      const ms = Number(b - a) / 1e6;
      res.labels[lab] = (res.labels[lab] || 0) + ms; res.sum += ms;
    }
    res.span = Number(hi - lo) / 1e6; res.quant = q100;
  }
  slot.buf.unmap(); slot.busy = false;
  onResult(res);
  return res;
}

// ---------------------------------------------------------------- stats panel
const ema = { labels: {}, span: 0, sum: 0, n: 0, up: 0, sub: 0, fps: 0 };
let lastTickT = 0, quantSeen = false;
function onResult(r) {
  if (r.warm) { $('#benchnote').textContent = ''; return; }
  const a = 0.12;
  for (const [k, v] of Object.entries(r.labels)) ema.labels[k] = ema.labels[k] === undefined ? v : ema.labels[k] * (1 - a) + v * a;
  const mix = (o, v) => (o === 0 ? v : o * (1 - a) + v * a);
  ema.span = mix(ema.span, r.span); ema.sum = mix(ema.sum, r.sum); ema.n = mix(ema.n, r.n);
  ema.up = mix(ema.up, r.cpuUpload - r.cpuSeg); ema.seg = mix(ema.seg || 0, r.cpuSeg); ema.sub = mix(ema.sub, r.cpuSubmit);
  if (r.quant) quantSeen = true;
  const now = performance.now();
  if (lastTickT && now - lastTickT > 1) ema.fps = mix(ema.fps, 1000 / (now - lastTickT));
  lastTickT = now;
}
function renderStats() {
  const rows = Object.entries(ema.labels).map(([k, v]) => `<tr><td>${k}</td><td>${v.toFixed(3)} ms</td></tr>`).join('');
  const budget = 1000 / FPS;
  $('#stats').innerHTML = rows +
    `<tr class="sum"><td>GPU busy (sum of passes)</td><td>${ema.sum.toFixed(3)} ms</td></tr>` +
    `<tr><td>GPU span (first to last pass)</td><td>${ema.span.toFixed(3)} ms</td></tr>` +
    `<tr><td>CPU: frame copy to GPU</td><td>${ema.up.toFixed(3)} ms</td></tr>` +
    (seg.ready && P.subject ? `<tr><td>CPU: subject separation (${seg.delegate})</td><td>${(ema.seg || 0).toFixed(2)} ms</td></tr>` : '') +
    `<tr><td>CPU: encode + submit</td><td>${ema.sub.toFixed(3)} ms</td></tr>` +
    `<tr><td>line points</td><td>${Math.round(ema.n).toLocaleString()}</td></tr>` +
    `<tr><td>frames per second</td><td>${ema.fps.toFixed(1)}</td></tr>` +
    `<tr><td>GPU</td><td>${GPU_NAME}</td></tr>` +
    `<tr><td>timestamps</td><td>${!hasTS ? 'not supported' : quantSeen ? 'quantized to 100 us' : 'fine'}</td></tr>`;
  $('#budget').style.width = Math.min(100, ema.span / budget * 100) + '%';
}
setInterval(renderStats, 250);

// ---------------------------------------------------------------- loop driver
let lastMedia = -1;
function loop() {
  requestAnimationFrame(loop);
  if (sim.view === 'v3') { stillTick(); return; }
  if (sim.view === 'v5') { if (cubes3d) cubes3d.frame(vid, P); return; }
  if (vid.readyState < 2) return;
  const t = vid.currentTime;
  if (t === lastMedia) return;                              // no new video frame yet
  if (t < lastMedia - 0.2) sim.boost = 8;                   // loop wrapped: wake everything for a few frames
  lastMedia = t;
  if (sim.view === 'src' || !sim.running || benchRunning) return;
  if (sim.pending >= 3) { sim.dropped++; return; }
  tick();
}
async function loadVideo(url) {
  stopCamera();
  vid.src = url; vid.muted = true; vid.loop = true;
  await new Promise((res, rej) => { vid.onloadeddata = res; vid.onerror = () => rej(new Error('video failed to load')); });
  await vid.play().catch(() => {});
}
function stopCamera() {
  if (!camStream) return;
  camStream.getTracks().forEach((t) => t.stop()); camStream = null;
  vid.srcObject = null; vid.style.transform = ''; $('#cam').classList.remove('on');
}
async function startCamera() {
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false });
  } catch (e) {
    camStream = null;
    msg('Camera is not available here (' + (e.name || e.message) + '). Use "Load video" to pick a file.');
    return false;
  }
  msg(''); vid.removeAttribute('src'); vid.srcObject = camStream; vid.loop = false;
  await vid.play().catch(() => {});
  vid.style.transform = P.mirror ? 'scaleX(-1)' : '';
  $('#cam').classList.add('on'); lastMedia = -1; resetSim();
  return true;
}
async function useCamera() {
  if (camStream) { stopCamera(); vid.removeAttribute('src'); msg('Camera off. Turn it on again or use "Load video".'); return; }
  await startCamera();
}
async function start() {
  resetSim();
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- UI
const versions = [
  { id: 'src', label: 'Source video' },
  { id: 'gpu', label: 'V1 Line' },
  { id: 'v2', label: 'V2 Line' },
  { id: 'v3', label: 'V3 100$' },
  { id: 'v4', label: 'V4 Iso Cubes' },
  { id: 'v5', label: 'V5 Cube Pixels' },
  { id: 'bit8', label: '8 bit' },
];
let cubes3d = null;
function setView(id) {
  sim.view = id;
  import('./cubes3d.js?b=' + (document.querySelector('meta[name=build]')?.content || 0)).then(async (m) => { cubes3d = m; await m.show(id === 'v5', $('#stage'), vid, P); }).catch((e) => msg('V5 failed: ' + e.message));
  vid.style.visibility = id === 'src' ? 'visible' : 'hidden';
  canvas.style.visibility = id !== 'src' && id !== 'v5' ? 'visible' : 'hidden';
  const land = id === 'v3';
  const cw = land ? BW : OW, ch = land ? BH : OH;
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  $('#stage').classList.toggle('land', land);
  if (id !== 'src') sim.boost = 8;
  if (id === 'v2' || id === 'v3' || id === 'v4') loadSegmenter();
  if (id === 'v4' && P.smodel === 'rvm') loadRvm();
  if (id === 'v3') { still.last = 0; if (P.smodel === 'rvm') loadRvm(); }
  document.querySelectorAll('#versions button').forEach((b) => b.classList.toggle('on', b.dataset.id === id));
  const vl = versions.find((v) => v.id === id);
  $('#setHead').textContent = vl.label + ' settings';
  $('#settings').hidden = id === 'src';
  document.querySelectorAll('[data-views]').forEach((el) => { el.hidden = !el.dataset.views.split(',').includes(id); });
}
for (const v of versions) {
  const b = document.createElement('button'); b.textContent = v.label; b.dataset.id = v.id;
  b.onclick = () => { if ((v.id === 'v3') !== LAND && v.id !== 'src') location.search = v.id === 'v3' ? '?land=1&view=v3' : '?view=' + v.id; else setView(v.id); };
  $('#versions').append(b);
}
const sliders = { rmin: 'rmin', rmax: 'rmax', gain: 'gain', noise: 'noise', iters: 'iters', stroke: 'stroke', px: 'px', v2s: 'v2s', v2r: 'v2r', v2a: 'v2a', v2b: 'v2b', v2o: 'v2o', v2g: 'v2g', v3s: 'v3s', v3r: 'v3r', v3a: 'v3a', v3b: 'v3b', v3d: 'v3d', v3z: 'v3z', v3w: 'v3w', v3m: 'v3m', v3int: 'v3int', v4s: 'v4s', v4n: 'v4n', v4l: 'v4l', v4d: 'v4d', v4f: 'v4f', v4w: 'v4w', v4b: 'v4b', v4a: 'v4a', v4k: 'v4k', v4m: 'v4m', v4r: 'v4r', v4q: 'v4q', v4j: 'v4j', v4z: 'v4z', v4y: 'v4y', v5s: 'v5s', v5f: 'v5f', v5k: 'v5k', v5t: 'v5t', v5g: 'v5g', v5r: 'v5r', v5c: 'v5c', v5m: 'v5m', v4c: 'v4c', v3p: 'v3p', v3res: 'v3res', v3x: 'v3x', v3y: 'v3y', v3k: 'v3k' };
for (const [id, key] of Object.entries(sliders)) {
  const el = $('#' + id), out = el.parentElement.querySelector('output');
  const upd = () => { P[key] = parseFloat(el.value); out.textContent = el.value; sim.boost = Math.max(sim.boost, 4); };
  el.oninput = upd; upd();
}
$('#cam').onclick = useCamera;
$('#v2m').onchange = (e) => { P.v2m = parseInt(e.target.value); };
$('#v2mode').onchange = (e) => { P.v2mode = parseInt(e.target.value); };
for (const id of Object.keys(sliders)) {              // editable min / max under every slider
  const el = $('#' + id), row = el.parentElement;
  const lim = document.createElement('div'); lim.className = 'lim';
  lim.innerHTML = '<span>min</span><input type="number" step="any"><span>max</span><input type="number" step="any">';
  row.append(lim);
  const [mi, ma] = lim.querySelectorAll('input'); mi.value = el.min; ma.value = el.max;
  const setLim = () => {
    const a = parseFloat(mi.value), b = parseFloat(ma.value);
    if (!(a < b)) { mi.value = el.min; ma.value = el.max; return; }
    el.min = a; el.max = b; el.value = Math.min(b, Math.max(a, parseFloat(el.value)));
    el.dispatchEvent(new Event('input'));
  };
  mi.onchange = setLim; ma.onchange = setLim; el.limInputs = [mi, ma];
}
$('#mirror').onchange = (e) => { P.mirror = e.target.checked; if (camStream) vid.style.transform = P.mirror ? 'scaleX(-1)' : ''; };
$('#subject').onchange = async (e) => { P.subject = parseInt(e.target.value); if (P.subject) await loadSegmenter(); sim.boost = 8; };
$('#key').onchange = (e) => { P.key = e.target.checked; sim.boost = 6; };
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
$('#smodel').onchange = async (e) => { P.smodel = e.target.value; if (P.smodel === 'rvm') await loadRvm(); };
$('#v3auto').onchange = (e) => { P.v3auto = e.target.checked; };
$('#v4i').oninput = (e) => { P.v4i = hex(e.target.value); };
$('#v4p').oninput = (e) => { P.v4p = hex(e.target.value); };
$('#v5i').oninput = (e) => { P.v5i = hex(e.target.value); };
$('#v5p').oninput = (e) => { P.v5p = hex(e.target.value); };
$('#v5col').onchange = (e) => { P.v5col = parseInt(e.target.value); };
$('#v5ax').onchange = (e) => { P.v5ax = parseInt(e.target.value); };
$('#v3c').oninput = (e) => { P.v3c = hex(e.target.value); };
$('#fg').oninput = (e) => { P.fg = hex(e.target.value); };
$('#bg').oninput = (e) => { P.bgc = hex(e.target.value); };
$('#restart').onclick = () => { resetSim(); };
$('#pick').onclick = () => $('#file').click();
$('#file').onchange = (e) => { const f = e.target.files[0]; if (f) { loadVideo(URL.createObjectURL(f)).then(resetSim); } };
const stage = $('#stage'), drop = $('#drop');
addEventListener('dragover', (e) => { e.preventDefault(); drop.style.display = 'flex'; });
addEventListener('dragleave', (e) => { if (!e.relatedTarget) drop.style.display = 'none'; });
addEventListener('drop', (e) => { e.preventDefault(); drop.style.display = 'none'; const f = e.dataTransfer.files[0]; if (f) loadVideo(URL.createObjectURL(f)).then(resetSim); });

// ---------------------------------------------------------------- benchmark
let benchRunning = false;
const pct = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
async function seekTo(t) {
  await new Promise((res) => { const h = () => { vid.removeEventListener('seeked', h); res(); }; vid.addEventListener('seeked', h); vid.currentTime = t; setTimeout(res, 1500); });
}
async function benchmark() {
  const note = $('#benchnote');
  if (vid.readyState < 2) { note.textContent = 'Benchmark needs a source: turn the camera on or load a video first.'; return; }
  const live = !!camStream || !isFinite(vid.duration);      // camera: run live frames instead of seeking a file
  benchRunning = true; $('#bench').disabled = true;
  const wasView = sim.view; if (wasView === 'src') setView('gpu');
  if (!live) vid.pause();
  const frames = live ? 210 : Math.max(1, Math.floor(vid.duration * FPS));
  note.textContent = 'Benchmark: warm-up frame...';
  resetSim();
  if (!live) await seekTo(0);
  const tw0 = performance.now();
  await tick();                                         // first frame incl. 150 warm-up steps
  await device.queue.onSubmittedWorkDone();
  const warmMs = performance.now() - tw0;
  const rec = [];
  for (let f = 1; f < frames; f++) {
    if (live) await new Promise((r) => setTimeout(r, 33)); else await seekTo(f / FPS + 0.0005);
    const r = await tick();
    if (r) rec.push(r);
    if (f % 15 === 0) note.textContent = `Benchmark: frame ${f}/${frames}`;
  }
  // worst case: everything awake, back to back, no readback
  note.textContent = 'Benchmark: worst case (all awake)...';
  const K1 = 60;
  await device.queue.onSubmittedWorkDone();
  const tw = performance.now();
  for (let i = 0; i < K1; i++) {
    writeU(true, 0.3);
    qn = 0; qlabels = []; timing = false;
    const enc = device.createCommandEncoder(); encodeField(enc); for (let s = 0; s < P.iters; s++) encodeStep(enc); encodeDraw(enc);
    device.queue.submit([enc.finish()]); sim.time += 1 / FPS;
  }
  await device.queue.onSubmittedWorkDone();
  const worstMs = (performance.now() - tw) / K1;
  timing = true;

  const labelsAll = {}; rec.forEach((r) => Object.keys(r.labels).forEach((k) => (labelsAll[k] = 1)));
  const perLabel = {};
  for (const k of Object.keys(labelsAll)) { const a = rec.map((r) => r.labels[k] || 0); perLabel[k] = { mean_ms: mean(a), p95_ms: pct(a, 0.95), max_ms: Math.max(...a) }; }
  const span = rec.map((r) => r.span), sum = rec.map((r) => r.sum);
  const result = {
    when: new Date().toISOString(), gpu: GPU_NAME, userAgent: navigator.userAgent,
    clip: { frames, live, fps: FPS, sim_size: `${SW}x${SH}`, output_size: `${OW}x${OH}` },
    settings: { ...P, warm_steps: WARM_STEPS },
    timestamps: { supported: hasTS, quantized_100us: rec.some((r) => r.quant) },
    first_frame_ms_wall: warmMs,
    per_frame_gpu_ms: { busy_mean: mean(sum), busy_p95: pct(sum, 0.95), busy_max: Math.max(...sum), span_mean: mean(span), span_p95: pct(span, 0.95), span_max: Math.max(...span) },
    realtime_budget_ms: 1000 / FPS,
    realtime_headroom_x: (1000 / FPS) / mean(span),
    per_stage_ms: perLabel,
    points: { mean: mean(rec.map((r) => r.n)), max: Math.max(...rec.map((r) => r.n)) },
    cpu_ms: { frame_copy_mean: mean(rec.map((r) => r.cpuUpload - r.cpuSeg)), subject_separation_mean: mean(rec.map((r) => r.cpuSeg)), encode_submit_mean: mean(rec.map((r) => r.cpuSubmit)) },
    worst_case_all_awake_ms_per_frame_wall: worstMs, worst_case_fps: 1000 / worstMs,
    reference_cpu_numba_2core_cloud: { note: 'measured by the earlier chat, not on this machine', first_frame_s: 8, per_frame_s: 0.5, full_7s_clip_s: 120 },
    frames_dropped_live: sim.dropped,
  };
  const name = `bench_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  let saved = false;
  try { saved = (await fetch('/save?name=' + name, { method: 'POST', body: JSON.stringify(result, null, 2) })).ok; } catch {}
  window.__bench = result;
  const fx = (v) => v.toFixed(2);
  note.innerHTML = `<b>Benchmark done</b> (${frames} frames, measured)<br>` +
    `first frame (150 warm-up steps): ${fx(warmMs)} ms wall<br>` +
    `per frame GPU span: mean ${fx(mean(span))} ms, p95 ${fx(pct(span, .95))} ms, max ${fx(Math.max(...span))} ms (budget ${fx(1000 / FPS)} ms)<br>` +
    `worst case, all points awake: ${fx(worstMs)} ms per frame = ${(1000 / worstMs).toFixed(0)} fps<br>` +
    (saved ? `saved to bench/${name}` : 'could not save to bench/ (open via serve.py); result is in window.__bench');
  if (!live) vid.play();
  setView(wasView); benchRunning = false; $('#bench').disabled = false;
  return result;
}
$('#bench').onclick = benchmark;
window.__tick = tick; window.__bench_run = benchmark; window.__state = { sim, P };

// ---------------------------------------------------------------- go
const STORE = 'lineGpuDefaults.v1';
const SAVE_IDS = ['stroke', 'rmin', 'rmax', 'gain', 'noise', 'iters', 'px', 'v2s', 'v2r', 'v2a', 'v2b', 'v2m', 'v2mode', 'v2o', 'v2g', 'v3s', 'v3r', 'v3a', 'v3b', 'v3d', 'v3z', 'v3x', 'v3y', 'v3k', 'v3w', 'v3m', 'v3int', 'v3p', 'v3res', 'v4s', 'v4n', 'v4l', 'v4d', 'v4f', 'v4w', 'v4b', 'v4a', 'v4k', 'v4m', 'v4r', 'v4q', 'v4j', 'v4z', 'v4y', 'v5s', 'v5f', 'v5k', 'v5t', 'v5g', 'v5r', 'v5c', 'v5m', 'v5ax', 'v5col', 'v5i', 'v5p', 'v4c', 'v4i', 'v4p', 'v3c', 'smodel', 'fg', 'bg', 'subject'];
const SAVE_CB = ['mirror', 'key', 'v3auto'];
function snapshotSettings() {
  const o = { view: sim.view === 'src' ? 'gpu' : sim.view, v: {}, c: {}, l: {} };
  for (const id of Object.keys(sliders)) o.l[id] = [$('#' + id).min, $('#' + id).max];
  SAVE_IDS.forEach((id) => { o.v[id] = $('#' + id).value; });
  SAVE_CB.forEach((id) => { o.c[id] = $('#' + id).checked; });
  return o;
}
function applySettings(o) {
  for (const [id, mm] of Object.entries(o.l || {})) {
    const el = $('#' + id); if (!el) continue;
    el.min = mm[0]; el.max = mm[1]; if (el.limInputs) { el.limInputs[0].value = mm[0]; el.limInputs[1].value = mm[1]; }
  }
  for (const [id, val] of Object.entries(o.v || {})) {
    const el = $('#' + id); if (!el) continue;
    el.value = val; el.dispatchEvent(new Event(el.tagName === 'SELECT' ? 'change' : 'input'));
  }
  for (const [id, val] of Object.entries(o.c || {})) {
    const el = $('#' + id); if (!el) continue;
    el.checked = val; el.dispatchEvent(new Event('change'));
  }
}
let saved = null;
try { saved = JSON.parse(localStorage.getItem(STORE) || 'null'); } catch {}
if (saved) applySettings(saved);
$('#setdef').onclick = () => {
  try { localStorage.setItem(STORE, JSON.stringify(snapshotSettings())); $('#setdef').textContent = 'Saved as default'; }
  catch { $('#setdef').textContent = 'Could not save'; }
  setTimeout(() => { $('#setdef').textContent = 'Set default'; }, 1600);
};
$('#resetdef').onclick = () => { try { localStorage.removeItem(STORE); } catch {} location.reload(); };
const qv = new URLSearchParams(location.search).get('view');
$('#smodel').value = P.smodel;
const startView = qv && versions.some((v) => v.id === qv) ? qv : saved && saved.view && versions.some((v) => v.id === saved.view) ? saved.view : 'gpu';
if ((startView === 'v3') !== LAND) location.replace(location.pathname + (startView === 'v3' ? '?land=1&view=v3' : '?view=' + startView));
setView(startView);
await start();
loadSegmenter();
await startCamera();
window.__dbg = async (name, count = 64, off = 0) => {
  const b = B[name], size = Math.min(b.size, (off + count) * 4);
  const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | CD });
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b, 0, st, 0, size); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
  return Array.from(a.slice(off, off + count));
};
window.__device = device;

// ---------------------------------------------------------------- version check
{
  const mine = document.querySelector('meta[name=build]')?.content || '';
  const badge = $('#ver');
  if (mine.startsWith('__')) badge.textContent = 'dev (local)';
  else {
    const check = async () => {
      try {
        const r = await fetch('version.json?t=' + Date.now(), { cache: 'no-store' }); const v = await r.json();
        if (v.build !== mine) { $('#newver').hidden = false; $('#newver').onclick = () => location.reload(); }
      } catch {}
    };
    check(); setInterval(check, 30000);
  }
}

window.__rvm = rvm;
