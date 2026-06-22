// Minimal column-major 4x4 matrix helpers (WebGL convention) and the
// quaternion/scale -> 3D covariance used by Gaussian splatting. No deps.

export type Mat4 = Float32Array; // length 16, column-major

export function perspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovYRad / 2);
  const nf = 1.0 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

// Right-handed lookAt. eye/target/up are [x,y,z].
export function lookAt(eye: number[], target: number[], up: number[]): Mat4 {
  const z0 = eye[0] - target[0], z1 = eye[1] - target[1], z2 = eye[2] - target[2];
  let zl = Math.hypot(z0, z1, z2) || 1;
  const zx = z0 / zl, zy = z1 / zl, zz = z2 / zl;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  const xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl; xy /= xl; xz /= xl;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  const m = new Float32Array(16);
  m[0] = xx; m[1] = yx; m[2] = zx; m[3] = 0;
  m[4] = xy; m[5] = yy; m[6] = zy; m[7] = 0;
  m[8] = xz; m[9] = yz; m[10] = zz; m[11] = 0;
  m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  m[15] = 1;
  return m;
}

// Symmetric 3D covariance Sigma = R S S^T R^T, returned as
// [s00, s01, s02, s11, s12, s22]. q = [w, x, y, z] (need not be normalized).
export function covarianceFromQuatScale(
  qw: number, qx: number, qy: number, qz: number,
  sx: number, sy: number, sz: number,
): [number, number, number, number, number, number] {
  const n = Math.hypot(qw, qx, qy, qz) || 1;
  qw /= n; qx /= n; qy /= n; qz /= n;

  // Rotation matrix columns.
  const r00 = 1 - 2 * (qy * qy + qz * qz);
  const r01 = 2 * (qx * qy - qw * qz);
  const r02 = 2 * (qx * qz + qw * qy);
  const r10 = 2 * (qx * qy + qw * qz);
  const r11 = 1 - 2 * (qx * qx + qz * qz);
  const r12 = 2 * (qy * qz - qw * qx);
  const r20 = 2 * (qx * qz - qw * qy);
  const r21 = 2 * (qy * qz + qw * qx);
  const r22 = 1 - 2 * (qx * qx + qy * qy);

  // M = R * diag(s): scale each column.
  const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
  const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
  const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

  // Sigma = M M^T.
  const s00 = m00 * m00 + m01 * m01 + m02 * m02;
  const s01 = m00 * m10 + m01 * m11 + m02 * m12;
  const s02 = m00 * m20 + m01 * m21 + m02 * m22;
  const s11 = m10 * m10 + m11 * m11 + m12 * m12;
  const s12 = m10 * m20 + m11 * m21 + m12 * m22;
  const s22 = m20 * m20 + m21 * m21 + m22 * m22;
  return [s00, s01, s02, s11, s12, s22];
}

export const SH_C0 = 0.28209479177387814;

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
