// V6 Liquid Surfaces: palmdrop "Surfaces" noise (liquid-fx2-surfaces.json). The subject mask plays the mouse: it boosts the pattern's own
// frequency / warp / time where the subject is and leaves a decaying trail when it moves. The background is the same pattern, brighter.
let gl = null, st = null;
const DEF = {scale: 1.78, warpAmount: 0.935, warpIterations: 4, animSpeed: 0.011, srcFrequency: 0.3291, srcLacunarity: 5.740366659674152, srcPersistence: 0.514, srcRidge: 0.9915719678769965, srcPow: 2.240971917382968, srcOctaves: 3, angFrequency: 0.3993420240238754, angLacunarity: 2.9519116325958965, angPersistence: 0.012859637563908644, angRidge: 0.9969207974367497, angPow: 2.0034261488639253, angOctaves: 3, amtFrequency: 0.5981440617034829, amtLacunarity: 2.5704336074177316, amtPersistence: 0.07686896917051708, amtRidge: 0.828, amtPow: 1.0510829716259347, amtOctaves: 5, hue: -0.24, saturation: 0.621209118076694, brightness: 1.14, power: 1, hueSrc: -0.19973268980328918, hueAng: 0.11885348084160197, hueAmt: -0.42670525006517657, satSrc: 0.21365060616739806, satAng: 0.07407397671756077, satAmt: 0.8013619448583572, briSrc: 1, briAng: 0, briAmt: 0, brushRadius: 0.11, decay: 0.9745, bleed: 0.5, phaseRate: 0, extraFreqMult: 1.22, extraWarpMult: 1.92, brushStrength: 0.82};   // the runtime defaults the portfolio page runs on
Object.assign(DEF, { saturation: 0.838632309403537, brightness: 1.539 });   // the purple look from the JSON's settings export
const VERT = 'attribute vec2 aPos;\nvoid main() {\n  gl_Position = vec4(aPos, 0.0, 1.0);\n}';
const UPDATE = [
  'precision highp float;',
  'uniform sampler2D uPrev; uniform sampler2D uMask;',
  'uniform vec2 uResolution; uniform float uDecay, uBleed, uPhaseRate, uDt, uStrength;',
  'vec2 decode(vec4 c) { return vec2(c.r, c.g * 6.0); }',
  'vec4 encode(vec2 v) { return vec4(v.x, v.y / 6.0, 0.0, 1.0); }',
  'vec2 sampleBlur(vec2 uv, vec2 t) {',
  '  vec2 s = decode(texture2D(uPrev, uv)) * 4.0;',
  '  s += decode(texture2D(uPrev, uv + vec2(t.x, 0.0))); s += decode(texture2D(uPrev, uv - vec2(t.x, 0.0)));',
  '  s += decode(texture2D(uPrev, uv + vec2(0.0, t.y))); s += decode(texture2D(uPrev, uv - vec2(0.0, t.y)));',
  '  return s / 8.0;',
  '}',
  'void main() {',
  '  vec2 uv = gl_FragCoord.xy / uResolution; vec2 texel = 1.0 / uResolution;',
  '  vec2 state = mix(decode(texture2D(uPrev, uv)), sampleBlur(uv, texel), uBleed);',
  '  state *= uDecay;',
  '  if (state.x < 0.008) state.x = 0.0;',
  '  if (state.y < 0.008) state.y = 0.0;',
  '  state.x = max(state.x, texture2D(uMask, uv).r * uStrength);',
  '  state.x = clamp(state.x, 0.0, 1.0);',
  '  state.y = clamp(state.y + state.x * uPhaseRate * uDt, 0.0, 6.0);',
  '  gl_FragColor = encode(state);',
  '}'
].join('\n');
const DISPLAY = [
    'precision highp float;',
    'uniform sampler2D uInfluence;',
    'uniform vec2 uResolution;',
    'uniform vec3 uSourceOffset;',
    'uniform vec3 uAngleOffset;',
    'uniform vec3 uAmountOffset;',
    'uniform float uScale;',
    'uniform float uWarpAmount;',
    'uniform int   uWarpIterations;',
    'uniform float uSrcFreq, uSrcLac, uSrcPers, uSrcRidge, uSrcPow;',
    'uniform float uAngFreq, uAngLac, uAngPers, uAngRidge, uAngPow;',
    'uniform float uAmtFreq, uAmtLac, uAmtPers, uAmtRidge, uAmtPow;',
    'uniform int   uSrcOctaves, uAngOctaves, uAmtOctaves;',
    'uniform float uHue, uSaturation, uBrightness, uPower;',
    'uniform float uHueSrc, uHueAng, uHueAmt;',
    'uniform float uSatSrc, uSatAng, uSatAmt;',
    'uniform float uBriSrc, uBriAng, uBriAmt;',
    'uniform float uExtraFreqMult;',
    'uniform float uExtraWarpMult;',
    'uniform float uBgBright, uBgWhite, uScaleBg, uBlur, uShowMask;',
    'uniform sampler2D uMaskView;',
    '#define PI 3.14159265358979323846',
    '#define PHASE_MAX 6.0',
    'vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }',
    'vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }',
    'vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }',
    'vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }',
    'float snoise(vec3 v){',
    '  const vec2 C = vec2(1.0/6.0, 1.0/3.0);',
    '  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);',
    '  vec3 i  = floor(v + dot(v, C.yyy));',
    '  vec3 x0 = v - i + dot(i, C.xxx);',
    '  vec3 g = step(x0.yzx, x0.xyz);',
    '  vec3 l = 1.0 - g;',
    '  vec3 i1 = min(g.xyz, l.zxy);',
    '  vec3 i2 = max(g.xyz, l.zxy);',
    '  vec3 x1 = x0 - i1 + C.xxx;',
    '  vec3 x2 = x0 - i2 + C.yyy;',
    '  vec3 x3 = x0 - D.yyy;',
    '  i = mod289(i);',
    '  vec4 p = permute(permute(permute(',
    '            i.z + vec4(0.0, i1.z, i2.z, 1.0))',
    '          + i.y + vec4(0.0, i1.y, i2.y, 1.0))',
    '          + i.x + vec4(0.0, i1.x, i2.x, 1.0));',
    '  float n_ = 0.142857142857;',
    '  vec3 ns = n_ * D.wyz - D.xzx;',
    '  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);',
    '  vec4 x_ = floor(j * ns.z);',
    '  vec4 y_ = floor(j - 7.0 * x_);',
    '  vec4 x = x_ * ns.x + ns.yyyy;',
    '  vec4 y = y_ * ns.x + ns.yyyy;',
    '  vec4 h = 1.0 - abs(x) - abs(y);',
    '  vec4 b0 = vec4(x.xy, y.xy);',
    '  vec4 b1 = vec4(x.zw, y.zw);',
    '  vec4 s0 = floor(b0) * 2.0 + 1.0;',
    '  vec4 s1 = floor(b1) * 2.0 + 1.0;',
    '  vec4 sh = -step(h, vec4(0.0));',
    '  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;',
    '  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;',
    '  vec3 p0 = vec3(a0.xy, h.x);',
    '  vec3 p1 = vec3(a0.zw, h.y);',
    '  vec3 p2 = vec3(a1.xy, h.z);',
    '  vec3 p3 = vec3(a1.zw, h.w);',
    '  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));',
    '  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;',
    '  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);',
    '  m = m * m;',
    '  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));',
    '}',
    'float getNoise(vec3 p, float frequency, float amplitude, vec3 offset, float ridgeThreshold, float powVal, float freqMult) {',
    '  float safeFreq = min(frequency * freqMult, 150.0);',
    '  vec3 samp = p * safeFreq + offset;',
    '  float result = snoise(samp);',
    '  result = 0.5 + result * 0.5;',
    '  if (ridgeThreshold < 1.0) {',
    '    float threshold = max(0.5, ridgeThreshold);',
    '    if (result > threshold) result = threshold - (result - threshold);',
    '    result /= threshold;',
    '  }',
    '  result = max(result, 0.0);',
    '  if (powVal != 1.0) result = pow(result, powVal);',
    '  return amplitude * result;',
    '}',
    'float noiseSupplier(vec3 p, float frequency, float lacunarity, float persistence, vec3 offset, float ridgeThreshold, float powVal, float freqMult, int octaves) {',
    '  float result = 0.0;',
    '  float maxAmp = 0.0;',
    '  float freq = frequency;',
    '  float amp = 1.0;',
    '  for (int i = 0; i < 5; i++) {',
    '    if (i >= octaves) break;',
    '    result += getNoise(p, freq, amp, offset, ridgeThreshold, powVal, freqMult);',
    '    maxAmp += amp;',
    '    freq *= lacunarity;',
    '    amp *= persistence;',
    '  }',
    '  return result / maxAmp;',
    '}',
    'vec2 polarWarp(vec2 point, vec3 angleOffset, vec3 amountOffset, float warpAmountLocal, float freqMult) {',
    '  float amount = noiseSupplier(vec3(point, 0.0), uAmtFreq, uAmtLac, uAmtPers, amountOffset, uAmtRidge, uAmtPow, freqMult, uAmtOctaves) * warpAmountLocal;',
    '  float angle = noiseSupplier(vec3(point, 0.0), uAngFreq, uAngLac, uAngPers, angleOffset, uAngRidge, uAngPow, freqMult, uAngOctaves) * PI * 2.0;',
    '  return point + vec2(cos(angle), sin(angle)) * amount;',
    '}',
    'vec3 hsv2rgb(vec3 c) {',
    '  vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);',
    '  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);',
    '  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);',
    '}',
    'float component(float v, float a, float e, float cSrc, float cAng, float cAmt) {',
    '  float divider = max(1.0, cSrc + cAng + cAmt);',
    '  return clamp((v * cSrc + a * cAng + e * cAmt) / divider, 0.0, 1.0);',
    '}',
    'void main() {',
    '  float aspect = uResolution.x / uResolution.y;',
    '  vec2 uv = gl_FragCoord.xy / uResolution.xy;',
    '  vec2 auv = vec2(uv.x * aspect, uv.y);',
    '  vec2 influence = texture2D(uInfluence, uv).rg;',
    '  float intensity = 0.0;',
    '  for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) intensity += texture2D(uInfluence, uv + vec2(float(i), float(j)) * uBlur * vec2(1.0 / aspect, 1.0)).r;',
    '  intensity /= 9.0;',
    '  float phase = influence.g * PHASE_MAX;',
    '  float freqMult = mix(1.0, uExtraFreqMult, intensity);',
    '  float warpAmountLocal = uWarpAmount * mix(1.0, uExtraWarpMult, intensity);',
    '  vec3 srcOff = uSourceOffset + vec3(0.0, 0.0, phase);',
    '  vec3 angOff = uAngleOffset  + vec3(0.0, 0.0, phase);',
    '  vec3 amtOff = uAmountOffset + vec3(0.0, 0.0, phase);',
    '  vec2 p = auv * mix(uScaleBg, uScale, smoothstep(0.2, 0.8, intensity));',
    '  for (int i = 0; i < 4; i++) {',
    '    if (i >= uWarpIterations) break;',
    '    p = polarWarp(p, angOff, amtOff, warpAmountLocal, freqMult);',
    '  }',
    '  float v = noiseSupplier(vec3(p, 0.0), uSrcFreq, uSrcLac, uSrcPers, srcOff, uSrcRidge, uSrcPow, freqMult, uSrcOctaves);',
    '  float a = noiseSupplier(vec3(p, 0.0), uAmtFreq, uAmtLac, uAmtPers, amtOff, uAmtRidge, uAmtPow, freqMult, uAmtOctaves);',
    '  float e = noiseSupplier(vec3(p, 0.0), uAngFreq, uAngLac, uAngPers, angOff, uAngRidge, uAngPow, freqMult, uAngOctaves);',
    '  float h = component(v, a, e, uHueSrc, uHueAng, uHueAmt);',
    '  h = 1.0 - h;',
    '  h += uHue;',
    '  float s = component(v, a, e, uSatSrc, uSatAng, uSatAmt) * uSaturation;',
    '  float b = component(v, a, e, uBriSrc, uBriAng, uBriAmt);',
    '  b = pow(b, uPower) * uBrightness;',
    '  float lift = 1.0 - smoothstep(0.2, 0.8, intensity);',
    '  vec3 color = hsv2rgb(vec3(h, s, min(b * (1.0 + lift * (uBgBright - 1.0)), 1.6)));',
    '  color = mix(color, vec3(1.0), lift * uBgWhite);',
    '  color = mix(color, vec3(texture2D(uMaskView, uv).r), uShowMask);',
    '  gl_FragColor = vec4(color, 1.0);',
    '}'
].join('\n');
function prog(src) {
  const mk = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, VERT)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, src)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
function tex(w, h, filter) {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  if (w) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  return t;
}
export function show(on, stage, vid, P) {
  if (!st) {
    if (!on) return;
    const cv = document.createElement('canvas'); cv.id = 'cv6'; cv.width = 540; cv.height = 960; stage.append(cv);
    gl = cv.getContext('webgl', { antialias: false });
    if (!gl) throw new Error('WebGL is not available');
    const upd = prog(UPDATE), dis = prog(DISPLAY);
    const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const IW = 270, IH = 480;
    const fb = [0, 1].map(() => { const t = tex(IW, IH, gl.LINEAR), f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); return { t, f }; });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const MW = 90, MH = 160, mt = tex(0, 0, gl.LINEAR);
    const rnd = () => Math.random() * 8;
    st = { cv, upd, dis, quad, fb, IW, IH, mt, MW, MH, mc: document.createElement('canvas'), rd: 0, off: [[rnd(), rnd(), 0], [rnd(), rnd(), 0], [rnd(), rnd(), 0]], u: {}, d: {}, t: 0 };
    st.f = new Float32Array(MW * MH); st.f2 = new Float32Array(MW * MH); st.mc.width = MW; st.mc.height = MH;
    for (const n of ['uPrev', 'uMask', 'uResolution', 'uDecay', 'uBleed', 'uPhaseRate', 'uDt', 'uStrength']) st.u[n] = gl.getUniformLocation(upd, n);
    for (const n of ['uInfluence', 'uResolution', 'uSourceOffset', 'uAngleOffset', 'uAmountOffset', 'uScale', 'uWarpAmount', 'uWarpIterations', 'uSrcFreq', 'uSrcLac', 'uSrcPers', 'uSrcRidge', 'uSrcPow', 'uAngFreq', 'uAngLac', 'uAngPers', 'uAngRidge', 'uAngPow', 'uAmtFreq', 'uAmtLac', 'uAmtPers', 'uAmtRidge', 'uAmtPow', 'uSrcOctaves', 'uAngOctaves', 'uAmtOctaves', 'uHue', 'uSaturation', 'uBrightness', 'uPower', 'uHueSrc', 'uHueAng', 'uHueAmt', 'uSatSrc', 'uSatAng', 'uSatAmt', 'uBriSrc', 'uBriAng', 'uBriAmt', 'uExtraFreqMult', 'uExtraWarpMult', 'uBgBright', 'uBgWhite', 'uShowMask', 'uMaskView', 'uScaleBg', 'uBlur']) st.d[n] = gl.getUniformLocation(dis, n);
  }
  st.cv.style.visibility = on ? 'visible' : 'hidden';
}
const bindQuad = (p) => { const l = gl.getAttribLocation(p, 'aPos'); gl.bindBuffer(gl.ARRAY_BUFFER, st.quad); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 2, gl.FLOAT, false, 0, 0); };
export function frame(vid, P, mk) {
  if (!st || st.cv.style.visibility === 'hidden') return;
  const now = performance.now(), dt = Math.min(0.1, (now - (st.t || now)) / 1000); st.t = now;
  // subject mask (video space) -> screen space texture: cover crop and mirror like the other versions
  const { MW, MH, mc } = st, x = mc.getContext('2d'), im = x.createImageData(MW, MH), o = im.data;
  if (mk && vid.videoWidth) {
    const vw = vid.videoWidth, vh = vid.videoHeight, ar = MW / MH;
    let cw = vw, ch = vw / ar; if (ch > vh) { ch = vh; cw = vh * ar; }
    for (let j = 0; j < MH; j++) for (let i = 0; i < MW; i++) {
      const cx = P.mirror ? MW - 1 - i : i;
      const u = (vw - cw) / 2 + ((cx + 0.5) / MW) * cw, v = (vh - ch) / 2 + ((j + 0.5) / MH) * ch;
      const m = mk.d[Math.min(mk.h - 1, (v / vh * mk.h) | 0) * mk.w + Math.min(mk.w - 1, (u / vw * mk.w) | 0)];
      st.f[j * MW + i] = Math.min(1, Math.max(0, (m - 0.2) / 0.6));
    }
    // wide soft edge: three box blur passes, so the object fades into the background instead of ending at a contour
    const R = Math.round(P.v6bl ?? 3), f = st.f, t = st.f2;
    if (R > 0) for (let pass = 0; pass < 3; pass++) {
      for (let j = 0; j < MH; j++) for (let i = 0; i < MW; i++) { let a = 0, n = 0; for (let q = -R; q <= R; q++) { const ii = i + q; if (ii >= 0 && ii < MW) { a += f[j * MW + ii]; n++; } } t[j * MW + i] = a / n; }
      for (let j = 0; j < MH; j++) for (let i = 0; i < MW; i++) { let a = 0, n = 0; for (let q = -R; q <= R; q++) { const jj = j + q; if (jj >= 0 && jj < MH) { a += t[jj * MW + i]; n++; } } f[j * MW + i] = a / n; }
    }
    for (let k = 0; k < f.length; k++) { const g = Math.round(255 * f[k]); o[k * 4] = o[k * 4 + 1] = o[k * 4 + 2] = g; o[k * 4 + 3] = 255; }
  } else for (let k = 0; k < o.length; k += 4) o[k + 3] = 255;
  x.putImageData(im, 0, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, st.mt); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, mc);
  const sp = P.v6sp ?? 1, df = P.v6df ?? 1;   // df scales every difference between object and background: 0 = identical, 1 = full
  const mixD = (v) => 1 + (v - 1) * df;
  for (const f of st.off) f[2] += DEF.animSpeed * sp * dt * 6;
  // update pass: decay + bleed + the mask as the "mouse"
  const w = 1 - st.rd, u = st.u;
  gl.bindFramebuffer(gl.FRAMEBUFFER, st.fb[w].f); gl.viewport(0, 0, st.IW, st.IH);
  gl.useProgram(st.upd); bindQuad(st.upd);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, st.fb[st.rd].t); gl.uniform1i(u.uPrev, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, st.mt); gl.uniform1i(u.uMask, 1);
  gl.uniform2f(u.uResolution, st.IW, st.IH); gl.uniform1f(u.uDecay, P.v6d ?? 0.99); gl.uniform1f(u.uBleed, DEF.bleed);
  gl.uniform1f(u.uPhaseRate, (P.v6ph ?? 0) * df); gl.uniform1f(u.uDt, dt); gl.uniform1f(u.uStrength, P.v6s ?? 0.82);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  st.rd = w;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, st.cv.width, st.cv.height);
  gl.useProgram(st.dis); bindQuad(st.dis);
  const d = st.d;
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, st.fb[st.rd].t); gl.uniform1i(d.uInfluence, 0);
  gl.uniform2f(d.uResolution, st.cv.width, st.cv.height);
  gl.uniform3f(d.uSourceOffset, ...st.off[0]); gl.uniform3f(d.uAngleOffset, ...st.off[1]); gl.uniform3f(d.uAmountOffset, ...st.off[2]);
  gl.uniform1f(d.uScale, P.v6sc ?? DEF.scale); gl.uniform1f(d.uWarpAmount, DEF.warpAmount); gl.uniform1i(d.uWarpIterations, DEF.warpIterations);
  for (const k of ['Src', 'Ang', 'Amt']) {
    const q = k.toLowerCase();
    gl.uniform1f(d['u' + k + 'Freq'], DEF[q + 'Frequency']); gl.uniform1f(d['u' + k + 'Lac'], DEF[q + 'Lacunarity']); gl.uniform1f(d['u' + k + 'Pers'], DEF[q + 'Persistence']);
    gl.uniform1f(d['u' + k + 'Ridge'], DEF[q + 'Ridge']); gl.uniform1f(d['u' + k + 'Pow'], DEF[q + 'Pow']); gl.uniform1i(d['u' + k + 'Octaves'], DEF[q + 'Octaves']);
  }
  gl.uniform1f(d.uHue, DEF.hue); gl.uniform1f(d.uSaturation, DEF.saturation); gl.uniform1f(d.uBrightness, DEF.brightness); gl.uniform1f(d.uPower, DEF.power);
  gl.uniform1f(d.uHueSrc, DEF.hueSrc); gl.uniform1f(d.uHueAng, DEF.hueAng); gl.uniform1f(d.uHueAmt, DEF.hueAmt);
  gl.uniform1f(d.uSatSrc, DEF.satSrc); gl.uniform1f(d.uSatAng, DEF.satAng); gl.uniform1f(d.uSatAmt, DEF.satAmt);
  gl.uniform1f(d.uBriSrc, DEF.briSrc); gl.uniform1f(d.uBriAng, DEF.briAng); gl.uniform1f(d.uBriAmt, DEF.briAmt);
  gl.uniform1f(d.uExtraFreqMult, mixD(P.v6f ?? 2.5)); gl.uniform1f(d.uExtraWarpMult, mixD(P.v6w ?? 2.5));
  gl.uniform1f(d.uScaleBg, P.v6sb ?? DEF.scale); gl.uniform1f(d.uBlur, 0.003); gl.uniform1f(d.uBgBright, mixD(P.v6b ?? 1)); gl.uniform1f(d.uBgWhite, (P.v6wh ?? 0) * df);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, st.mt); gl.uniform1i(d.uMaskView, 1); gl.uniform1f(d.uShowMask, P.v6pv ? 1 : 0);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}
