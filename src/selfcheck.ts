// Headless verification (no WebGL): parses the demo scene through the real PLY
// path and checks the invariants the renderer relies on, plus unit tests for
// the covariance math and the 2D screen-space projection used in the shader.
// Run with `npm run check`. Exits non-zero on any failure.

import { parsePly, writePly, type RawSplat } from "./ply";
import { parseSplat, writeSplat, type RawSplatRecord } from "./splat";
import { depthSortOrder } from "./renderer";
import { buildDemoScene } from "./scene";
import { covarianceFromQuatScale, lookAt } from "./math";
import { OrbitCamera } from "./camera";

let failures = 0;
function check(name: string, cond: boolean, detail = ""): void {
  if (cond) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}${detail ? "  — " + detail : ""}`);
  }
}

function approx(a: number, b: number, eps = 1e-5): boolean {
  return Math.abs(a - b) <= eps;
}

// Symmetric 3x3 from [s00,s01,s02,s11,s12,s22]; PD via Sylvester's criterion.
function isPositiveDefinite(c: number[]): boolean {
  const [s00, s01, s02, s11, s12, s22] = c;
  const d1 = s00;
  const d2 = s00 * s11 - s01 * s01;
  const det = s00 * (s11 * s22 - s12 * s12)
    - s01 * (s01 * s22 - s12 * s02)
    + s02 * (s01 * s12 - s11 * s02);
  return d1 > 0 && d2 > 0 && det > 0;
}

// TS mirror of the shader's 3D->2D covariance projection (identity view).
function project2D(cov: number[], cam: number[], focal: number): [number, number, number] {
  const [s00, s01, s02, s11, s12, s22] = cov;
  const z = cam[2];
  // J (row-major here), W = identity, T = J.
  const j00 = focal / z, j02 = -(focal * cam[0]) / (z * z);
  const j11 = focal / z, j12 = -(focal * cam[1]) / (z * z);
  // cov2d = J * Vrk * J^T, take 2x2. Vrk symmetric.
  // Row 0 of J: [j00, 0, j02]; Row 1: [0, j11, j12].
  const m00 = j00 * s00 + j02 * s02; // (J*Vrk) row0 col0
  const m01 = j00 * s01 + j02 * s12; // row0 col1
  const m02 = j00 * s02 + j02 * s22; // row0 col2
  const n00 = j11 * s01 + j12 * s02; // row1 col0
  const n01 = j11 * s11 + j12 * s12; // row1 col1
  const n02 = j11 * s12 + j12 * s22; // row1 col2
  const a = m00 * j00 + m02 * j02; // (J*Vrk*J^T)[0][0]
  const b = m00 * 0 + m01 * j11 + m02 * j12; // [0][1]
  const c = n00 * 0 + n01 * j11 + n02 * j12; // [1][1]
  return [a, b, c];
}

console.log("covariance math");
{
  const id = covarianceFromQuatScale(1, 0, 0, 0, 1, 1, 1);
  check("identity quat + unit scale -> identity", id.every((v, i) => approx(v, [1, 0, 0, 1, 0, 1][i])));

  const aniso = covarianceFromQuatScale(1, 0, 0, 0, 2, 1, 1);
  check("scale (2,1,1) -> diag(4,1,1)", approx(aniso[0], 4) && approx(aniso[3], 1) && approx(aniso[5], 1) && approx(aniso[1], 0));

  // 90 deg about Z swaps x/y variance.
  const h = Math.SQRT1_2;
  const rot = covarianceFromQuatScale(h, 0, 0, h, 2, 1, 1);
  check("90deg-Z rot of (2,1,1) -> diag(1,4,1)", approx(rot[0], 1, 1e-4) && approx(rot[3], 4, 1e-4) && approx(rot[5], 1, 1e-4));
}

console.log("ply round-trip");
{
  const raw: RawSplat[] = [
    { x: 1.5, y: -2.25, z: 0.5, fdc: [0.1, 0.2, 0.3], opacity: 0.4, scale: [-3, -3, -3], rot: [1, 0, 0, 0] },
    { x: -7, y: 3, z: 9.5, fdc: [-0.5, 0, 0.5], opacity: -1, scale: [-2, -2.5, -3], rot: [0.5, 0.5, 0.5, 0.5] },
  ];
  const data = parsePly(writePly(raw));
  check("count preserved", data.count === 2, `got ${data.count}`);
  check("position preserved exactly", data.positions[0] === 1.5 && data.positions[1] === -2.25 && data.positions[5] === 9.5);
  check("alpha = sigmoid(0.4)", approx(data.colors[3], 1 / (1 + Math.exp(-0.4))));
}

console.log(".splat round-trip");
{
  const recs: RawSplatRecord[] = [
    { x: 1.5, y: 2.0, z: -0.5, scale: [0.1, 0.05, 0.05], rgba: [255, 128, 0, 200], rot: [1, 0, 0, 0] },
    { x: -3, y: -1, z: 4, scale: [0.2, 0.2, 0.02], rgba: [10, 20, 30, 40], rot: [0.5, 0.5, 0.5, 0.5] },
  ];
  const data = parseSplat(writeSplat(recs));
  check("count preserved", data.count === 2, `got ${data.count}`);
  check("Y is flipped (+Y down -> +Y up)", approx(data.positions[1], -2.0) && approx(data.positions[4], 1.0));
  check("X/Z preserved", approx(data.positions[0], 1.5) && approx(data.positions[2], -0.5));
  check("color = rgba/255", approx(data.colors[0], 1, 1e-2) && approx(data.colors[1], 128 / 255, 1e-2) && approx(data.colors[3], 200 / 255, 1e-2));
  check("scale is linear (no exp): isotropic-ish stays small", data.cov3d[0] < 0.1 && data.cov3d[0] > 0);

  let pd = true;
  for (let i = 0; i < data.count; i++) {
    if (!isPositiveDefinite(Array.from(data.cov3d.subarray(i * 6, i * 6 + 6)))) pd = false;
  }
  check("covariances positive-definite after Y reflection", pd);
}

console.log("depth sort");
{
  // Identity view -> view-space z equals world z; far-to-near is ascending z.
  const positions = new Float32Array([0, 0, 5, 0, 0, -3, 0, 0, 1, 0, 0, -10]);
  const view = lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]); // looks down -Z
  const order = depthSortOrder(positions, 4, view);
  // Camera at origin looking down -Z: a splat at z=-10 is farthest, z=5 is behind.
  const z = (i: number) => positions[i * 3 + 2];
  let sorted = true;
  for (let k = 1; k < order.length; k++) if (z(order[k]) < z(order[k - 1])) sorted = false;
  check("order is far-to-near (ascending world z under this view)", sorted, `order=${Array.from(order)}`);
  check("order is a permutation of 0..n-1", new Set(order).size === 4 && Math.max(...order) === 3);

  // Bigger random set: result must stay a valid far-to-near permutation.
  const N = 5000;
  const big = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    big[i * 3] = Math.sin(i) * 3;
    big[i * 3 + 1] = Math.cos(i * 1.7) * 3;
    big[i * 3 + 2] = ((i * 2654435761) % 1000) / 100 - 5;
  }
  const ord = depthSortOrder(big, N, view);
  let mono = true;
  for (let k = 1; k < N; k++) if (big[ord[k] * 3 + 2] < big[ord[k - 1] * 3 + 2] - 1e-3) mono = false;
  check("large set sorts monotonically by depth", mono);
  check("large set order is a permutation", new Set(ord).size === N);
}

console.log("demo scene");
{
  const demo = buildDemoScene();
  const data = parsePly(demo.ply);
  check("parsed count == generated count", data.count === demo.count, `${data.count} vs ${demo.count}`);
  check("has many splats", data.count > 10000, `${data.count}`);

  let finite = true, colorOk = true, alphaOk = true, pd = true;
  const stride = Math.max(1, Math.floor(data.count / 2000));
  for (let i = 0; i < data.count; i += stride) {
    for (let k = 0; k < 3; k++) if (!Number.isFinite(data.positions[i * 3 + k])) finite = false;
    for (let k = 0; k < 3; k++) { const v = data.colors[i * 4 + k]; if (v < 0 || v > 1) colorOk = false; }
    const al = data.colors[i * 4 + 3];
    if (!(al > 0 && al < 1)) alphaOk = false;
    const cov = Array.from(data.cov3d.subarray(i * 6, i * 6 + 6));
    if (!isPositiveDefinite(cov)) pd = false;
  }
  check("all positions finite", finite);
  check("all colors in [0,1]", colorOk);
  check("all alphas in (0,1)", alphaOk);
  check("all 3D covariances positive-definite", pd);
}

console.log("2D projection");
{
  // A unit-ish anisotropic splat 5 units in front of the camera.
  const cov = covarianceFromQuatScale(1, 0, 0, 0, 0.2, 0.05, 0.05);
  const [a, b, c] = project2D(cov, [0.3, -0.2, -5], 800);
  check("2D cov is PSD (det >= 0)", a * c - b * b >= 0, `det=${(a * c - b * b).toExponential(2)}`);
  check("2D cov has positive variance", a > 0 && c > 0, `a=${a}, c=${c}`);
  check("2D cov shrinks with distance", (() => {
    const near = project2D(cov, [0.3, -0.2, -3], 800);
    const far = project2D(cov, [0.3, -0.2, -9], 800);
    return near[0] > far[0];
  })());
}

console.log("orbit camera");
{
  const cam = new OrbitCamera([0, 0, 0]);
  const r0 = Math.hypot(...cam.eye() as [number, number, number]);
  check("eye sits at radius from target", approx(r0, cam.radius, 1e-4));

  const before = cam.eye();
  cam.rotate(120, 40);
  const after = cam.eye();
  const moved = Math.hypot(after[0] - before[0], after[1] - before[1], after[2] - before[2]);
  check("rotate moves the eye", moved > 0.1, `moved=${moved.toFixed(3)}`);
  check("rotate preserves radius", approx(Math.hypot(...after as [number, number, number]), cam.radius, 1e-4));

  cam.phi = 10; // beyond clamp
  cam.rotate(0, 0);
  check("elevation is clamped below 90deg", Math.abs(cam.phi) < Math.PI / 2);

  const rad = cam.radius;
  cam.zoom(-200);
  check("zoom in shrinks radius", cam.radius < rad);
  cam.zoom(1e6);
  check("zoom is clamped to a max", cam.radius <= 40);
}

console.log("");
if (failures === 0) {
  console.log("ALL CHECKS PASSED");
} else {
  console.log(`${failures} CHECK(S) FAILED`);
  process.exit(1);
}
