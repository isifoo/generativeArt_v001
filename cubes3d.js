// V5 Cube Pixels: every pixel of the camera is a cube (three.js) that turns about one axis, 0 deg = brightest, full turn = darkest
let T = null, st = null;
export async function show(on, stage, vid, P) {
  if (!st) {
    if (!on) return;
    T = await import('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js');
    const cv = document.createElement('canvas'); cv.id = 'cv3'; stage.append(cv);
    const renderer = new T.WebGLRenderer({ canvas: cv, antialias: true });
    renderer.setPixelRatio(1); renderer.setSize(1080, 1920, false);
    const scene = new T.Scene();
    const cam = new T.PerspectiveCamera(35, 1080 / 1920, 10, 20000);
    scene.add(new T.AmbientLight(0xffffff, 0.55));
    const dl = new T.DirectionalLight(0xffffff, 1.1); dl.position.set(-500, 800, 900); scene.add(dl);
    st = { cv, renderer, scene, cam, mesh: null, key: '', samp: document.createElement('canvas'), d: null, lo: 0, hi: 1 };
  }
  st.cv.style.visibility = on ? 'visible' : 'hidden';
  if (on) frame(vid, P);
}
function build(P) {
  const s = P.v5s, g = P.v5g;
  const cols = Math.max(2, Math.round(1080 / s)), rows = Math.max(2, Math.round(1920 / s));
  const key = [cols, rows, g].join();
  if (key === st.key) return;
  st.key = key; st.cols = cols; st.rows = rows;
  if (st.mesh) { st.scene.remove(st.mesh); st.mesh.geometry.dispose(); st.mesh.material.dispose(); }
  const cs = 1080 / cols;
  const mesh = new T.InstancedMesh(new T.BoxGeometry(cs * g, cs * g, cs * g), new T.MeshLambertMaterial({ color: 0xffffff }), cols * rows);
  mesh.instanceColor = new T.InstancedBufferAttribute(new Float32Array(cols * rows * 3), 3);
  st.cs = cs; st.mesh = mesh; st.scene.add(mesh);
  st.samp.width = cols; st.samp.height = rows; st.d = new Float32Array(cols * rows);
}
let obj = null;
export function frame(vid, P) {
  if (!st || st.cv.style.visibility === 'hidden' || vid.readyState < 2 || !vid.videoWidth) return;
  build(P);
  const { cols, rows, cs, mesh, samp, d } = st;
  const sx = samp.getContext('2d', { willReadFrequently: true });
  // cover-crop the video into one sample per cube
  const vw = vid.videoWidth, vh = vid.videoHeight, ar = cols / rows;
  let cw = vw, ch = vw / ar; if (ch > vh) { ch = vh; cw = vh * ar; }
  sx.save(); if (P.mirror) { sx.translate(cols, 0); sx.scale(-1, 1); }
  sx.drawImage(vid, (vw - cw) / 2, (vh - ch) / 2, cw, ch, 0, 0, cols, rows); sx.restore();
  const px = sx.getImageData(0, 0, cols, rows).data;
  let lo = 1, hi = 0;
  const raw = new Float32Array(cols * rows);
  for (let i = 0; i < raw.length; i++) {
    const l = (0.2126 * px[i * 4] + 0.7152 * px[i * 4 + 1] + 0.0722 * px[i * 4 + 2]) / 255;
    raw[i] = l; if (l < lo) lo = l; if (l > hi) hi = l;
  }
  st.lo += (lo - st.lo) * 0.1; st.hi += (hi - st.hi) * 0.1;           // slow auto range so the whole turn is used
  const span = Math.max(0.05, st.hi - st.lo), ink = P.v5i.map((x) => x / 255), pap = P.v5p.map((x) => x / 255);
  const rng = P.v5r * Math.PI / 180, sm = P.v5m, col = new T.Color();
  obj = obj || new T.Object3D();
  for (let i = 0; i < raw.length; i++) {
    const norm = Math.min(1, Math.max(0, (raw[i] - st.lo) / span));
    let dk = 1 - Math.pow(norm, P.v5c);                                    // 0 = brightest, 1 = darkest
    dk = Math.min(1, Math.max(0, dk + (dk * dk * (3 - 2 * dk) - dk) * P.v5k));   // S-curve: pushes lights and darks apart
    d[i] += (dk - d[i]) * (1 - sm);
    obj.position.set((i % cols + 0.5) * cs - 540, 960 - (Math.floor(i / cols) + 0.5) * cs, 0);
    const a = d[i] * rng;
    obj.rotation.set(P.v5ax === 1 ? a : 0, P.v5ax === 0 ? a : 0, P.v5ax === 2 ? a : 0);
    obj.updateMatrix(); mesh.setMatrixAt(i, obj.matrix);
    col.setRGB(pap[0] + (ink[0] - pap[0]) * d[i], pap[1] + (ink[1] - pap[1]) * d[i], pap[2] + (ink[2] - pap[2]) * d[i]);
    mesh.setColorAt(i, col);
  }
  mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
  // real perspective: the camera sits in front of the grid and drifts sideways by the tilt, so the sides of the cubes show
  const fov = P.v5f, dist = 960 / Math.tan(fov * Math.PI / 360), tl = P.v5t * Math.PI / 180;
  st.cam.fov = fov; st.cam.updateProjectionMatrix();
  st.cam.position.set(Math.sin(tl) * dist * 0.6, Math.sin(tl) * dist * 0.4, Math.cos(tl) * dist);
  st.cam.lookAt(0, 0, 0);
  st.renderer.setClearColor(new T.Color(pap[0], pap[1], pap[2]));
  st.renderer.render(st.scene, st.cam);
}
