// V1 LINE on the GPU (WebGPU). Port of engine/line_engine.py.
// Whole pipeline stays on the GPU: video -> CLAHE darkness -> spacing field R -> differential growth -> draw.
const SW = 540, SH = 960, OW = 1080, OH = 1920, SCALE = 2, MAXN = 262144, MARGIN = 6;
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
ctx.configure({ device, format: FORMAT, alphaMode: 'opaque' });
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
  mask: mk(F32, S | CD), state: mk(F32, S), R: mk(F32, S | CD), wake: mk(F32, S), wakeT: mk(F32, S), wakeD: mk(F32, S),
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
const P = { rmin: 3.05, rmax: 13.75, gain: 1.75, noise: 0.22, iters: 6, stroke: 4.75, key: false, mirror: true, subject: 1, grid: 2.5, px: 6,
  fg: [210, 50, 25], bgc: [242, 240, 238] };
const sim = { frame: 0, time: 0, boost: 0, inflight: 0, pending: 0, running: true, view: 'gpu', dropped: 0 };
const U = new Float32Array(16);

function writeU(fullWake, newW) {
  U.set([sim.time, P.rmin, P.rmax, P.gain, P.noise, fullWake ? 1 : 0, 0.30, 0.6, P.key ? 1 : 0, newW, 2.5, seg.ready ? P.subject : 0]);
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
  sim.frame = 0; sim.time = 0; sim.boost = 0;
}

// ---------------------------------------------------------------- video source
const vid = $('#vid');
let camStream = null;
const cv2 = document.createElement('canvas'); cv2.width = SW; cv2.height = SH;
const c2 = cv2.getContext('2d', { willReadFrequently: false });
c2.imageSmoothingQuality = 'high';
function uploadFrame() {
  const vw = vid.videoWidth, vh = vid.videoHeight, ar = SW / SH;
  let sx = 0, sy = 0, sw = vw, sh = vh;
  if (vw / vh > ar) { sw = vh * ar; sx = (vw - sw) / 2; } else { sh = vw / ar; sy = (vh - sh) / 2; }
  c2.save();
  if (camStream && P.mirror) { c2.translate(SW, 0); c2.scale(-1, 1); }
  c2.drawImage(vid, sx, sy, sw, sh, 0, 0, SW, SH);
  c2.restore();
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
function segment() {
  if (!seg.ready || P.subject === 0) { seg.ms = 0; return; }
  const t0 = performance.now();
  seg.ts = Math.max(seg.ts + 1, Math.round(performance.now()));
  const r = seg.obj.segmentForVideo(cv2, seg.ts);
  const m = r.confidenceMasks[0];
  let a = m.getAsFloat32Array();
  if (m.width !== SW || m.height !== SH) {                 // nearest resample if the model output is a different size
    const b = new Float32Array(SW * SH);
    for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) b[y * SW + x] = a[Math.floor(y * m.height / SH) * m.width + Math.floor(x * m.width / SW)];
    a = b;
  }
  device.queue.writeBuffer(B.mask, 0, a);
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
  const dv = new Float32Array([SCALE, P.stroke / 2, OW, OH, P.fg[0] / 255, P.fg[1] / 255, P.fg[2] / 255, 1, sim.view === 'ortho' ? 1 : 0, P.grid, 0, 0]);
  device.queue.writeBuffer(B.du, 0, dv);
  const d = {
    colorAttachments: [{ view: msaaView, resolveTarget: ctx.getCurrentTexture().createView(),
      clearValue: { r: P.bgc[0] / 255, g: P.bgc[1] / 255, b: P.bgc[2] / 255, a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
  };
  if (hasTS && timing && qn + 2 <= CAP) { d.timestampWrites = { querySet: qs, beginningOfPassWriteIndex: qn, endOfPassWriteIndex: qn + 1 }; qlabels.push(['draw: render', qn]); qn += 2; }
  const p = enc.beginRenderPass(d);
  p.setPipeline(drawPipe); p.setBindGroup(0, drawBG); p.drawIndirect(B.args, 32); p.end();
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
  const nIter = steps ?? (sim.boost > 0 ? P.iters * 2 : P.iters);
  if (first) {
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
  { id: 'gpu', label: 'V1 Line, GPU live' },
  { id: 'ortho', label: '90 degree lines' },
  { id: 'bit8', label: '8 bit' },
];
function setView(id) {
  sim.view = id;
  vid.style.visibility = id === 'src' ? 'visible' : 'hidden';
  canvas.style.visibility = id !== 'src' ? 'visible' : 'hidden';
  if (id !== 'src') sim.boost = 8;
  document.querySelectorAll('#versions button').forEach((b) => b.classList.toggle('on', b.dataset.id === id));
}
for (const v of versions) {
  const b = document.createElement('button'); b.textContent = v.label; b.dataset.id = v.id;
  b.onclick = () => setView(v.id); $('#versions').append(b);
}
const sliders = { rmin: 'rmin', rmax: 'rmax', gain: 'gain', noise: 'noise', iters: 'iters', stroke: 'stroke', grid: 'grid', px: 'px' };
for (const [id, key] of Object.entries(sliders)) {
  const el = $('#' + id), out = el.parentElement.querySelector('output');
  const upd = () => { P[key] = parseFloat(el.value); out.textContent = el.value; sim.boost = Math.max(sim.boost, 4); };
  el.oninput = upd; upd();
}
$('#cam').onclick = useCamera;
$('#mirror').onchange = (e) => { P.mirror = e.target.checked; if (camStream) vid.style.transform = P.mirror ? 'scaleX(-1)' : ''; };
$('#subject').onchange = async (e) => { P.subject = parseInt(e.target.value); if (P.subject) await loadSegmenter(); sim.boost = 8; };
$('#key').onchange = (e) => { P.key = e.target.checked; sim.boost = 6; };
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
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
  await new Promise((res) => { const h = () => { vid.removeEventListener('seeked', h); res(); }; vid.addEventListener('seeked', h); vid.currentTime = t; });
}
async function benchmark() {
  benchRunning = true; $('#bench').disabled = true;
  const note = $('#benchnote');
  const wasView = sim.view; if (wasView === 'src') setView('gpu');
  vid.pause();
  const frames = Math.max(1, Math.floor(vid.duration * FPS));
  note.textContent = 'Benchmark: warm-up frame...';
  resetSim();
  await seekTo(0);
  const tw0 = performance.now();
  await tick();                                         // first frame incl. 150 warm-up steps
  await device.queue.onSubmittedWorkDone();
  const warmMs = performance.now() - tw0;
  const rec = [];
  for (let f = 1; f < frames; f++) {
    await seekTo(f / FPS + 0.0005);
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
    clip: { frames, fps: FPS, sim_size: `${SW}x${SH}`, output_size: `${OW}x${OH}` },
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
  vid.play(); setView(wasView); benchRunning = false; $('#bench').disabled = false;
  return result;
}
$('#bench').onclick = benchmark;
window.__tick = tick; window.__bench_run = benchmark; window.__state = { sim, P };

// ---------------------------------------------------------------- go
setView('gpu');
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
