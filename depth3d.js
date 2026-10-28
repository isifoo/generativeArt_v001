// V7 Depth Boxes: a depth model (Depth Anything V2 Small, onnxruntime-web on WebGPU) turns the camera into a depth map, and a grid of
// shiny blue bevel boxes (three.js) is pushed toward the viewer by the depth under each box.
let T = null, ort = null, st = null, initP = null;
const W3 = 1080, H3 = 1920;

async function init(stage) {
  T = await import('three');
  const { RoundedBoxGeometry } = await import('three/addons/geometries/RoundedBoxGeometry.js');
  const { RoomEnvironment } = await import('three/addons/environments/RoomEnvironment.js');
  const cv = document.createElement('canvas'); cv.id = 'cv7'; stage.append(cv);
  const renderer = new T.WebGLRenderer({ canvas: cv, antialias: true });
  renderer.setPixelRatio(1); renderer.setSize(W3, H3, false);
  renderer.toneMapping = T.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  const scene = new T.Scene(); scene.background = new T.Color(0x06080f);
  const pm = new T.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  const cam = new T.PerspectiveCamera(35, W3 / H3, 10, 20000); cam.position.set(0, 0, (H3 / 2) / Math.tan(17.5 * Math.PI / 180));
  const dl = new T.DirectionalLight(0xffffff, 1.4); dl.position.set(-600, 900, 1400); scene.add(dl);
  const pv = document.createElement('canvas'); pv.width = 135; pv.height = 240;
  pv.style.cssText = 'position:absolute;left:8px;bottom:8px;width:27%;height:auto;border:1px solid #fff4;display:none;z-index:5';
  stage.append(pv);
  const note = document.createElement('div');
  note.style.cssText = 'position:absolute;left:0;right:0;top:45%;text-align:center;color:#9ab;font:14px sans-serif;z-index:6;pointer-events:none';
  stage.append(note);
  st = {
    cv, renderer, scene, cam, RB: RoundedBoxGeometry, mesh: null, key: '', mat: null, pv, note, obj: new T.Object3D(),
    ic: document.createElement('canvas'), sess: null, busy: false, ready: false, loading: false, ms: 0,
    dep: null, lo: 0, hi: 1, tgt: null, cur: null, cols: 0, rows: 0, cs: 0,
  };
}

export async function show(on, stage, vid, P) {
  if (!st) { if (!on) return; initP = initP || init(stage); await initP; }
  if (!st) return;
  st.cv.style.visibility = on ? 'visible' : 'hidden';
  if (!on) { st.pv.style.display = 'none'; st.note.textContent = ''; }
  else loadModel();
}

async function loadModel() {
  if (st.ready || st.loading) return;
  st.loading = true; st.note.textContent = 'Loading depth model (50 MB, first time only)...';
  try {
    ort = await import('./vendor/ort/ort.webgpu.min.mjs');
    ort.env.wasm.wasmPaths = new URL('./vendor/ort/', location.href).href;
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    for (const ep of ['webgpu', 'wasm']) {
      try { st.sess = await ort.InferenceSession.create(new URL('./vendor/depth_anything_v2_small_fp16.onnx', import.meta.url).href, { executionProviders: [ep] }); st.ep = ep; break; }
      catch (e) { if (ep === 'wasm') throw e; }
    }
    st.ready = true; st.note.textContent = '';
  } catch (e) { st.note.textContent = 'Depth model failed: ' + e.message; }
  st.loading = false;
}

function build(P) {
  const s = Math.max(12, P.v7s ?? 36), r = P.v7r ?? 0.18;
  const cols = Math.max(2, Math.round(W3 / s)), rows = Math.max(2, Math.round(H3 / s));
  const key = [cols, rows, r].join();
  if (key === st.key) return;
  st.key = key; st.cols = cols; st.rows = rows; st.cs = W3 / cols;
  if (st.mesh) { st.scene.remove(st.mesh); st.mesh.geometry.dispose(); }
  if (!st.mat) st.mat = new T.MeshPhysicalMaterial({ color: 0x1f5bff, metalness: 0.7, roughness: 0.1, clearcoat: 1, clearcoatRoughness: 0.04, envMapIntensity: 2 });
  const mesh = new T.InstancedMesh(new st.RB(1, 1, 1, 3, Math.min(0.49, r)), st.mat, cols * rows);
  mesh.frustumCulled = false; st.mesh = mesh; st.scene.add(mesh);
  st.tgt = new Float32Array(cols * rows); st.cur = new Float32Array(cols * rows);
  if (st.dep) sampleGrid();
}

// depth model output (video space, mirrored and cover-cropped already) -> one normalised value per box
function sampleGrid() {
  const { dep, cols, rows, tgt } = st, { d, w, h } = dep, span = Math.max(1e-6, st.hi - st.lo);
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const x = Math.min(w - 1.001, Math.max(0, (i + 0.5) / cols * w - 0.5)), y = Math.min(h - 1.001, Math.max(0, (j + 0.5) / rows * h - 0.5));
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0;
    const a = d[y0 * w + x0], b = d[y0 * w + x0 + 1], c = d[(y0 + 1) * w + x0], e = d[(y0 + 1) * w + x0 + 1];
    const v = (a + (b - a) * fx) * (1 - fy) + (c + (e - c) * fx) * fy;
    tgt[j * cols + i] = Math.min(1, Math.max(0, (v - st.lo) / span));
  }
}

async function infer(vid, P) {
  st.busy = true;
  try {
    const k = Math.round(P.v7q ?? 18), W = 14 * k, H = 14 * Math.round(W * 16 / 9 / 14);
    const ic = st.ic; if (ic.width !== W || ic.height !== H) { ic.width = W; ic.height = H; }
    const x = ic.getContext('2d', { willReadFrequently: true });
    const vw = vid.videoWidth, vh = vid.videoHeight, ar = W / H;
    let cw = vw, ch = vw / ar; if (ch > vh) { ch = vh; cw = vh * ar; }
    x.save(); if (P.mirror) { x.translate(W, 0); x.scale(-1, 1); }
    x.drawImage(vid, (vw - cw) / 2, (vh - ch) / 2, cw, ch, 0, 0, W, H); x.restore();
    const px = x.getImageData(0, 0, W, H).data, n = W * H, f = new Float32Array(3 * n);
    const mean = [0.485, 0.456, 0.406], sd = [0.229, 0.224, 0.225];
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) f[c * n + i] = (px[i * 4 + c] / 255 - mean[c]) / sd[c];
    const t0 = performance.now();
    const out = await st.sess.run({ pixel_values: new ort.Tensor('float32', f, [1, 3, H, W]) });
    const o = out[Object.keys(out)[0]], dims = o.dims, oh = dims[dims.length - 2], ow = dims[dims.length - 1];
    const raw = o.data ?? await o.getData(), d = Float32Array.from(raw);
    st.ms += ((performance.now() - t0) - st.ms) * 0.2;
    // robust range (2% .. 98%) with a slow follow, so a stray pixel does not squash the whole relief
    let mn = Infinity, mx = -Infinity; for (let i = 0; i < d.length; i++) { if (d[i] < mn) mn = d[i]; if (d[i] > mx) mx = d[i]; }
    const hist = new Float32Array(256), sp = Math.max(1e-6, mx - mn);
    for (let i = 0; i < d.length; i++) hist[Math.min(255, ((d[i] - mn) / sp * 256) | 0)]++;
    let acc = 0, l = 0, hgh = 255; for (let b = 0; b < 256; b++) { acc += hist[b]; if (acc < d.length * 0.02) l = b; if (acc < d.length * 0.98) hgh = b + 1; }
    const lo = mn + l / 256 * sp, hi = mn + hgh / 256 * sp;
    if (!st.dep) { st.lo = lo; st.hi = hi; } else { st.lo += (lo - st.lo) * 0.15; st.hi += (hi - st.hi) * 0.15; }
    st.dep = { d, w: ow, h: oh };
    if (st.tgt) sampleGrid();
  } catch (e) { st.note.textContent = 'Depth run failed: ' + e.message; st.ready = false; }
  st.busy = false;
}

export function frame(vid, P) {
  if (!st || st.cv.style.visibility === 'hidden') return;
  if (st.ready && !st.busy && vid.readyState >= 2 && vid.videoWidth) infer(vid, P);
  if (st.ready && !st.dep) st.note.textContent = 'Waiting for the first depth map...'; else if (st.dep && st.note.textContent.startsWith('Waiting')) st.note.textContent = '';
  build(P);
  const { cols, rows, cs, mesh, cur, tgt, obj } = st;
  st.mat.color.set(document.getElementById('v7col')?.value || '#1f5bff');
  st.mat.metalness = P.v7m ?? 0.7; st.mat.roughness = P.v7ro ?? 0.1;
  st.scene.background.set(document.getElementById('v7bgc')?.value || '#06080f');
  const sm = Math.min(0.95, Math.max(0, P.v7sm ?? 0.5)), crv = P.v7c ?? 1.2, zr = P.v7z ?? 600, len = P.v7l ?? 1, g = P.v7g ?? 0.92, hide = P.v7bg ?? 0;
  for (let i = 0; i < cur.length; i++) {
    cur[i] += (Math.pow(tgt[i], crv) - cur[i]) * (1 - sm);
    let sc = g; if (hide > 0) { const t = Math.min(1, Math.max(0, (cur[i] - (hide - 0.1)) / 0.1)); sc *= t * t * (3 - 2 * t); }
    obj.position.set((i % cols + 0.5) * cs - W3 / 2, H3 / 2 - (((i / cols) | 0) + 0.5) * cs, cur[i] * zr);
    obj.scale.set(cs * sc, cs * sc, Math.max(0.001, cs * sc * len)); obj.updateMatrix();
    mesh.setMatrixAt(i, obj.matrix);
  }
  mesh.instanceMatrix.needsUpdate = true;
  st.renderer.render(st.scene, st.cam);
  const showPv = P.v7pv && st.dep;
  st.pv.style.display = showPv ? 'block' : 'none';
  if (showPv && (st.pvT !== st.dep)) {
    st.pvT = st.dep; const { d, w, h } = st.dep, c = st.pv.getContext('2d'), im = c.createImageData(135, 240), span = Math.max(1e-6, st.hi - st.lo);
    for (let j = 0; j < 240; j++) for (let i = 0; i < 135; i++) {
      const v = Math.min(1, Math.max(0, (d[Math.min(h - 1, (j / 240 * h) | 0) * w + Math.min(w - 1, (i / 135 * w) | 0)] - st.lo) / span)), q = (j * 135 + i) * 4;
      im.data[q] = im.data[q + 1] = im.data[q + 2] = v * 255; im.data[q + 3] = 255;
    }
    c.putImageData(im, 0, 0);
  }
}
