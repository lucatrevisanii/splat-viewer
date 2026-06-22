// Parser for the compact ".splat" format (antimatter15/splat) that public
// WebGL Gaussian-splatting scenes ship in. Each splat is a fixed 32-byte
// record, already in *activated* form — unlike the 3DGS .ply (see ply.ts),
// which stores pre-activation SH / log-scale / logit-opacity:
//   pos    float32[3]  (12B)
//   scale  float32[3]  (12B)  linear, no exp()
//   rgba   uint8[4]     (4B)  color already SH/sigmoid-baked, just /255
//   rot    uint8[4]     (4B)  quaternion (w,x,y,z), dequantized (b-128)/128
//
// Coordinates follow the COLMAP/3DGS convention (+Y down). The viewer's orbit
// camera is +Y up, so we flip Y on the position and reflect the covariance
// across the Y plane (negate the xy and yz terms) — an exact reflection that
// preserves positive-definiteness.

import { covarianceFromQuatScale } from "./math";
import type { SplatData } from "./ply";

export const SPLAT_RECORD_SIZE = 32;

export function parseSplat(buffer: ArrayBuffer): SplatData {
  if (buffer.byteLength === 0 || buffer.byteLength % SPLAT_RECORD_SIZE !== 0) {
    throw new Error(
      `.splat: byte length ${buffer.byteLength} is not a positive multiple of ${SPLAT_RECORD_SIZE}`,
    );
  }
  const count = buffer.byteLength / SPLAT_RECORD_SIZE;
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 4);
  const cov3d = new Float32Array(count * 6);

  for (let i = 0; i < count; i++) {
    const base = i * SPLAT_RECORD_SIZE;

    positions[i * 3] = view.getFloat32(base, true);
    positions[i * 3 + 1] = -view.getFloat32(base + 4, true); // +Y down -> +Y up
    positions[i * 3 + 2] = view.getFloat32(base + 8, true);

    const sx = view.getFloat32(base + 12, true);
    const sy = view.getFloat32(base + 16, true);
    const sz = view.getFloat32(base + 20, true);

    colors[i * 4] = u8[base + 24] / 255;
    colors[i * 4 + 1] = u8[base + 25] / 255;
    colors[i * 4 + 2] = u8[base + 26] / 255;
    colors[i * 4 + 3] = u8[base + 27] / 255;

    const qw = (u8[base + 28] - 128) / 128;
    const qx = (u8[base + 29] - 128) / 128;
    const qy = (u8[base + 30] - 128) / 128;
    const qz = (u8[base + 31] - 128) / 128;

    const cov = covarianceFromQuatScale(qw, qx, qy, qz, sx, sy, sz);
    cov[1] = -cov[1]; // s01 (xy): reflect across the Y plane
    cov[4] = -cov[4]; // s12 (yz)
    cov3d.set(cov, i * 6);
  }

  return { count, positions, colors, cov3d };
}

// --- writer (used by the headless self-check to round-trip the format) -------

export interface RawSplatRecord {
  x: number; y: number; z: number;
  scale: [number, number, number]; // linear
  rgba: [number, number, number, number]; // 0..255
  rot: [number, number, number, number]; // w, x, y, z (any norm; quantized on write)
}

export function writeSplat(records: RawSplatRecord[]): ArrayBuffer {
  const buffer = new ArrayBuffer(records.length * SPLAT_RECORD_SIZE);
  const view = new DataView(buffer);
  const u8 = new Uint8Array(buffer);

  records.forEach((s, i) => {
    const base = i * SPLAT_RECORD_SIZE;
    view.setFloat32(base, s.x, true);
    view.setFloat32(base + 4, s.y, true);
    view.setFloat32(base + 8, s.z, true);
    view.setFloat32(base + 12, s.scale[0], true);
    view.setFloat32(base + 16, s.scale[1], true);
    view.setFloat32(base + 20, s.scale[2], true);
    u8[base + 24] = clampByte(s.rgba[0]);
    u8[base + 25] = clampByte(s.rgba[1]);
    u8[base + 26] = clampByte(s.rgba[2]);
    u8[base + 27] = clampByte(s.rgba[3]);

    const n = Math.hypot(s.rot[0], s.rot[1], s.rot[2], s.rot[3]) || 1;
    for (let k = 0; k < 4; k++) {
      u8[base + 28 + k] = clampByte((s.rot[k] / n) * 128 + 128);
    }
  });

  return buffer;
}

function clampByte(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}
