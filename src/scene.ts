// Procedural demo scene: two interlocked tori made of anisotropic Gaussians
// oriented along each ring's tangent. Fully synthetic (no captured data), it
// shows the thing that separates splats from point clouds, oriented, ellipsoidal
// kernels, and it stresses depth sorting since the rings overlap.

import { SH_C0 } from "./math";
import { writePly, type RawSplat } from "./ply";

const logit = (p: number) => Math.log(p / (1 - p));
const fdc = (c: number) => (c - 0.5) / SH_C0; // invert color activation

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

// Rotation matrix (orthonormal columns) -> quaternion (w, x, y, z).
function frameToQuat(
  c0: number[], c1: number[], c2: number[],
): [number, number, number, number] {
  const m00 = c0[0], m10 = c0[1], m20 = c0[2];
  const m01 = c1[0], m11 = c1[1], m21 = c1[2];
  const m02 = c2[0], m12 = c2[1], m22 = c2[2];
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    return [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    return [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    return [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s];
  }
}

const norm = (v: number[]): number[] => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
const cross = (a: number[], b: number[]): number[] => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

interface TorusOpts {
  R: number; r: number; nu: number; nv: number;
  hueShift: number;
  // map a local point/frame into world (for the second, rotated ring)
  xform: (p: number[]) => number[];
  xformDir: (d: number[]) => number[];
}

function torus(out: RawSplat[], o: TorusOpts): void {
  for (let iu = 0; iu < o.nu; iu++) {
    const u = (iu / o.nu) * Math.PI * 2;
    const cu = Math.cos(u), su = Math.sin(u);
    for (let iv = 0; iv < o.nv; iv++) {
      const v = (iv / o.nv) * Math.PI * 2;
      const cv = Math.cos(v), sv = Math.sin(v);

      const rad = o.R + o.r * cv;
      const local = [cu * rad, su * rad, o.r * sv];
      const ringCenter = [cu * o.R, su * o.R, 0];

      const tangent = norm([-su, cu, 0]); // d/du, along the ring
      const outward = norm([
        local[0] - ringCenter[0],
        local[1] - ringCenter[1],
        local[2] - ringCenter[2],
      ]);
      const bino = norm(cross(tangent, outward));

      // world-space frame
      const t = norm(o.xformDir(tangent));
      const n = norm(o.xformDir(outward));
      const b = norm(o.xformDir(bino));
      const p = o.xform(local);

      const q = frameToQuat(t, b, n); // x->tangent, y->binormal, z->outward
      const [cr, cg, cb] = hsvToRgb(((u / (Math.PI * 2)) + o.hueShift) % 1, 0.85, 0.95);

      out.push({
        x: p[0], y: p[1], z: p[2],
        fdc: [fdc(cr), fdc(cg), fdc(cb)],
        opacity: logit(0.9),
        scale: [Math.log(0.07), Math.log(0.022), Math.log(0.022)], // elongated along tangent
        rot: q,
      });
    }
  }
}

export function buildDemoScene(): { ply: ArrayBuffer; count: number } {
  const out: RawSplat[] = [];

  // Ring A: in the XY plane.
  torus(out, {
    R: 1.0, r: 0.32, nu: 360, nv: 44, hueShift: 0.0,
    xform: (p) => p,
    xformDir: (d) => d,
  });

  // Ring B: same shape rotated 90 deg about X and shifted, so the two link.
  const rotX = (p: number[]) => [p[0], -p[2], p[1]];
  torus(out, {
    R: 1.0, r: 0.32, nu: 360, nv: 44, hueShift: 0.5,
    xform: (p) => {
      const q = rotX(p);
      return [q[0] + 1.0, q[1], q[2]];
    },
    xformDir: rotX,
  });

  return { ply: writePly(out), count: out.length };
}
