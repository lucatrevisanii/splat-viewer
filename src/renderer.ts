// WebGL2 EWA splatting renderer. Splat data lives in a float texture; a single
// instanced quad is drawn per splat. The 3D covariance is projected to a 2D
// screen-space conic in the vertex shader (Zwicker et al. EWA splatting), and
// the fragment shader evaluates the Gaussian with premultiplied-alpha "over"
// blending. Splats are depth-sorted on the CPU (painter's algorithm) each time
// the camera moves.

import type { SplatData } from "./ply";
import type { Mat4 } from "./math";

const TEX_WIDTH = 1024;

const VERT = `#version 300 es
precision highp float;
precision highp int;

uniform sampler2D u_data;
uniform int u_texWidth;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_focal;
uniform vec2 u_viewport;

in vec2 a_corner;
in uint a_index;

out vec4 v_color;
out vec2 v_corner;

vec4 fetch(uint texel) {
  int t = int(texel);
  return texelFetch(u_data, ivec2(t % u_texWidth, t / u_texWidth), 0);
}

void main() {
  uint base = a_index * 4u;
  vec4 d0 = fetch(base);
  vec4 d1 = fetch(base + 1u);
  vec4 d2 = fetch(base + 2u);
  vec4 d3 = fetch(base + 3u);

  vec3 center = d0.xyz;
  float alpha = d0.w;
  vec3 rgb = d1.xyz;

  // Symmetric 3D covariance.
  mat3 Vrk = mat3(
    d2.x, d2.y, d2.z,
    d2.y, d3.x, d3.y,
    d2.z, d3.y, d3.z
  );

  vec4 cam = u_view * vec4(center, 1.0);
  if (cam.z > -0.2) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  vec4 clip = u_proj * cam;
  float lim = 1.3 * clip.w;
  if (clip.z < -lim || clip.x < -lim || clip.x > lim || clip.y < -lim || clip.y > lim) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return;
  }

  // Jacobian of the perspective projection at cam (cam.z < 0).
  mat3 J = mat3(
    u_focal.x / cam.z, 0.0, -(u_focal.x * cam.x) / (cam.z * cam.z),
    0.0, u_focal.y / cam.z, -(u_focal.y * cam.y) / (cam.z * cam.z),
    0.0, 0.0, 0.0
  );
  mat3 W = mat3(u_view);
  mat3 T = W * J;
  mat3 cov = transpose(T) * Vrk * T;

  // Low-pass filter so sub-pixel splats stay visible.
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;

  float a = cov[0][0], b = cov[0][1], c = cov[1][1];
  float mid = 0.5 * (a + c);
  float rad = sqrt(max(0.0, mid * mid - (a * c - b * b)));
  float lambda1 = mid + rad;
  float lambda2 = mid - rad;
  if (lambda2 <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  vec2 e = vec2(b, lambda1 - a);
  vec2 dir1 = (dot(e, e) < 1e-12) ? vec2(1.0, 0.0) : normalize(e);
  vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * dir1;
  vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(dir1.y, -dir1.x);

  v_color = vec4(rgb, alpha);
  v_corner = a_corner;

  vec2 centerNdc = clip.xy / clip.w;
  vec2 offset = (a_corner.x * majorAxis + a_corner.y * minorAxis) * 2.0 / u_viewport;
  gl_Position = vec4(centerNdc + offset, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

in vec4 v_color;
in vec2 v_corner;
out vec4 frag;

void main() {
  float A = -dot(v_corner, v_corner);
  if (A < -4.0) discard;
  float B = exp(A) * v_color.a;
  frag = vec4(v_color.rgb * B, B);
}`;

export class SplatRenderer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private indexBuf: WebGLBuffer;
  private dataTex: WebGLTexture;
  private uni: Record<string, WebGLUniformLocation | null> = {};

  private count = 0;
  private positions: Float32Array = new Float32Array(0);
  private scratch: SortScratch | null = null;
  private needsSort = true;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;

    this.prog = link(gl, VERT, FRAG);
    gl.useProgram(this.prog);
    for (const name of ["u_data", "u_texWidth", "u_view", "u_proj", "u_focal", "u_viewport"]) {
      this.uni[name] = gl.getUniformLocation(this.prog, name);
    }

    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);

    // Per-vertex quad corners in sigma space [-2, 2], drawn as a triangle strip.
    const cornerBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
    const aCorner = gl.getAttribLocation(this.prog, "a_corner");
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);

    this.indexBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    const aIndex = gl.getAttribLocation(this.prog, "a_index");
    gl.enableVertexAttribArray(aIndex);
    gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(aIndex, 1);

    this.dataTex = gl.createTexture()!;
    gl.bindVertexArray(null);
  }

  setSplats(data: SplatData): void {
    const gl = this.gl;
    this.count = data.count;
    this.positions = data.positions;

    const texels = data.count * 4;
    const height = Math.ceil(texels / TEX_WIDTH);
    const buf = new Float32Array(TEX_WIDTH * height * 4);
    for (let i = 0; i < data.count; i++) {
      const o = i * 16;
      buf[o] = data.positions[i * 3];
      buf[o + 1] = data.positions[i * 3 + 1];
      buf[o + 2] = data.positions[i * 3 + 2];
      buf[o + 3] = data.colors[i * 4 + 3]; // alpha
      buf[o + 4] = data.colors[i * 4];
      buf[o + 5] = data.colors[i * 4 + 1];
      buf[o + 6] = data.colors[i * 4 + 2];
      buf[o + 8] = data.cov3d[i * 6];
      buf[o + 9] = data.cov3d[i * 6 + 1];
      buf[o + 10] = data.cov3d[i * 6 + 2];
      buf[o + 12] = data.cov3d[i * 6 + 3];
      buf[o + 13] = data.cov3d[i * 6 + 4];
      buf[o + 14] = data.cov3d[i * 6 + 5];
    }

    gl.bindTexture(gl.TEXTURE_2D, this.dataTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_WIDTH, height, 0, gl.RGBA, gl.FLOAT, buf);

    this.scratch = {
      depths: new Float32Array(data.count),
      bins: new Uint16Array(data.count),
      counts: new Uint32Array(DEPTH_BINS),
      order: new Uint32Array(data.count),
    };
    this.needsSort = true;
  }

  markDirty(): void {
    this.needsSort = true;
  }

  private sort(view: Mat4): void {
    if (!this.scratch) return;
    const order = depthSortOrder(this.positions, this.count, view, this.scratch);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, order, gl.DYNAMIC_DRAW);
  }

  render(view: Mat4, proj: Mat4, focal: [number, number], viewport: [number, number]): void {
    const gl = this.gl;
    if (this.count === 0) return;
    if (this.needsSort) {
      this.sort(view);
      this.needsSort = false;
    }

    gl.viewport(0, 0, viewport[0], viewport[1]);
    gl.clearColor(0.05, 0.06, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    // Premultiplied-alpha "over": dst = src + (1 - src.a) * dst.
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dataTex);
    gl.uniform1i(this.uni.u_data, 0);
    gl.uniform1i(this.uni.u_texWidth, TEX_WIDTH);
    gl.uniformMatrix4fv(this.uni.u_view, false, view);
    gl.uniformMatrix4fv(this.uni.u_proj, false, proj);
    gl.uniform2f(this.uni.u_focal, focal[0], focal[1]);
    gl.uniform2f(this.uni.u_viewport, viewport[0], viewport[1]);

    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
  }
}

function link(gl: WebGL2RenderingContext, vsrc: string, fsrc: string): WebGLProgram {
  const vs = compile(gl, gl.VERTEX_SHADER, vsrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsrc);
  const prog = gl.createProgram()!;
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error("Program link failed: " + gl.getProgramInfoLog(prog));
  }
  return prog;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error("Shader compile failed: " + gl.getShaderInfoLog(sh));
  }
  return sh;
}

// --- depth sort --------------------------------------------------------------

const DEPTH_BINS = 65536;

export interface SortScratch {
  depths: Float32Array; // count
  bins: Uint16Array; // count
  counts: Uint32Array; // DEPTH_BINS
  order: Uint32Array; // count
}

// Painter's-algorithm ordering, far-to-near. A comparison sort is O(n log n)
// and allocates a closure per element — at ~1M splats it stalls every camera
// move. This is a 16-bit counting sort: two O(n) passes over the depths plus
// one over the bins, no per-element allocation. `scratch` lets the caller reuse
// buffers across frames; omit it (e.g. in tests) to allocate fresh.
export function depthSortOrder(
  positions: Float32Array,
  count: number,
  view: Mat4,
  scratch?: SortScratch,
): Uint32Array {
  const depths = scratch?.depths ?? new Float32Array(count);
  const bins = scratch?.bins ?? new Uint16Array(count);
  const counts = scratch?.counts ?? new Uint32Array(DEPTH_BINS);
  const order = scratch?.order ?? new Uint32Array(count);
  counts.fill(0);

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    // View-space z; more negative = farther from the camera.
    const z =
      view[2] * positions[i * 3] +
      view[6] * positions[i * 3 + 1] +
      view[10] * positions[i * 3 + 2] +
      view[14];
    depths[i] = z;
    if (z < min) min = z;
    if (z > max) max = z;
  }

  const range = max - min;
  const scale = range > 0 ? (DEPTH_BINS - 1) / range : 0;
  for (let i = 0; i < count; i++) {
    const b = ((depths[i] - min) * scale) | 0; // far -> small bin, drawn first
    bins[i] = b;
    counts[b]++;
  }
  // Prefix sum turns counts into stable per-bin start offsets (ascending depth).
  let sum = 0;
  for (let b = 0; b < DEPTH_BINS; b++) {
    const c = counts[b];
    counts[b] = sum;
    sum += c;
  }
  for (let i = 0; i < count; i++) {
    order[counts[bins[i]]++] = i;
  }
  return order;
}
