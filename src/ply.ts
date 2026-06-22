// Parser + writer for the binary PLY format used by 3D Gaussian Splatting
// (INRIA "gaussian-splatting"). Reads only the properties we render and is
// agnostic to property order / extra fields (normals, spherical harmonics
// rest terms, etc.). Activations match the reference implementation:
//   color = clamp(0.5 + SH_C0 * f_dc)   opacity = sigmoid(opacity)
//   scale = exp(scale)                  rotation = normalized (w, x, y, z)

import { covarianceFromQuatScale, SH_C0, sigmoid } from "./math";

export interface SplatData {
  count: number;
  positions: Float32Array; // 3 * count
  colors: Float32Array; // 4 * count, rgba in [0, 1]
  cov3d: Float32Array; // 6 * count: s00, s01, s02, s11, s12, s22
}

const TYPE_SIZE: Record<string, number> = {
  char: 1, uchar: 1, int8: 1, uint8: 1,
  short: 2, ushort: 2, int16: 2, uint16: 2,
  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
  double: 8, float64: 8,
};

interface Prop {
  name: string;
  type: string;
  offset: number;
}

export function parsePly(buffer: ArrayBuffer): SplatData {
  const bytes = new Uint8Array(buffer);
  const headerEnd = findHeaderEnd(bytes);
  if (headerEnd < 0) throw new Error("PLY: end_header not found");

  const header = new TextDecoder("ascii").decode(bytes.subarray(0, headerEnd));
  const lines = header.split("\n").map((l) => l.trim());

  if (!lines.some((l) => l === "format binary_little_endian 1.0")) {
    throw new Error("PLY: only binary_little_endian 1.0 is supported");
  }

  let count = 0;
  const props: Prop[] = [];
  let stride = 0;
  let inVertex = false;
  for (const line of lines) {
    if (line.startsWith("element ")) {
      const [, name, n] = line.split(/\s+/);
      inVertex = name === "vertex";
      if (inVertex) count = parseInt(n, 10);
    } else if (line.startsWith("property ") && inVertex) {
      const [, type, name] = line.split(/\s+/);
      const size = TYPE_SIZE[type];
      if (size === undefined) throw new Error(`PLY: unknown property type ${type}`);
      props.push({ name, type, offset: stride });
      stride += size;
    }
  }
  if (count === 0) throw new Error("PLY: no vertices");

  const get = (name: string) => props.find((p) => p.name === name);
  const need = (name: string) => {
    const p = get(name);
    if (!p) throw new Error(`PLY: missing property ${name}`);
    return p;
  };

  const px = need("x"), py = need("y"), pz = need("z");
  const fdc0 = need("f_dc_0"), fdc1 = need("f_dc_1"), fdc2 = need("f_dc_2");
  const op = need("opacity");
  const sc0 = need("scale_0"), sc1 = need("scale_1"), sc2 = need("scale_2");
  const r0 = need("rot_0"), r1 = need("rot_1"), r2 = need("rot_2"), r3 = need("rot_3");

  const view = new DataView(buffer, headerEnd);
  const read = (p: Prop, base: number): number => {
    const at = base + p.offset;
    switch (p.type) {
      case "float": case "float32": return view.getFloat32(at, true);
      case "double": case "float64": return view.getFloat64(at, true);
      case "uchar": case "uint8": return view.getUint8(at);
      case "char": case "int8": return view.getInt8(at);
      case "ushort": case "uint16": return view.getUint16(at, true);
      case "short": case "int16": return view.getInt16(at, true);
      case "uint": case "uint32": return view.getUint32(at, true);
      default: return view.getInt32(at, true);
    }
  };

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 4);
  const cov3d = new Float32Array(count * 6);

  for (let i = 0; i < count; i++) {
    const base = i * stride;
    positions[i * 3] = read(px, base);
    positions[i * 3 + 1] = read(py, base);
    positions[i * 3 + 2] = read(pz, base);

    colors[i * 4] = clamp01(0.5 + SH_C0 * read(fdc0, base));
    colors[i * 4 + 1] = clamp01(0.5 + SH_C0 * read(fdc1, base));
    colors[i * 4 + 2] = clamp01(0.5 + SH_C0 * read(fdc2, base));
    colors[i * 4 + 3] = sigmoid(read(op, base));

    const cov = covarianceFromQuatScale(
      read(r0, base), read(r1, base), read(r2, base), read(r3, base),
      Math.exp(read(sc0, base)), Math.exp(read(sc1, base)), Math.exp(read(sc2, base)),
    );
    cov3d.set(cov, i * 6);
  }

  return { count, positions, colors, cov3d };
}

function findHeaderEnd(bytes: Uint8Array): number {
  const marker = "end_header\n";
  for (let i = 0; i + marker.length <= bytes.length; i++) {
    let ok = true;
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker.charCodeAt(j)) { ok = false; break; }
    }
    if (ok) return i + marker.length;
  }
  return -1;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// --- writer -----------------------------------------------------------------

export interface RawSplat {
  x: number; y: number; z: number;
  fdc: [number, number, number]; // pre-activation SH dc
  opacity: number; // logit
  scale: [number, number, number]; // log
  rot: [number, number, number, number]; // w, x, y, z (any norm)
}

// Writes the standard 3DGS PLY (binary_little_endian) so the synthetic scene
// is exercised through the same parser real captures go through.
export function writePly(splats: RawSplat[]): ArrayBuffer {
  const names = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
  ];
  const header =
    "ply\n" +
    "format binary_little_endian 1.0\n" +
    `element vertex ${splats.length}\n` +
    names.map((n) => `property float ${n}`).join("\n") + "\n" +
    "end_header\n";

  const headerBytes = new TextEncoder().encode(header);
  const stride = names.length * 4;
  const buffer = new ArrayBuffer(headerBytes.length + splats.length * stride);
  new Uint8Array(buffer).set(headerBytes, 0);

  const view = new DataView(buffer, headerBytes.length);
  let o = 0;
  const put = (v: number) => { view.setFloat32(o, v, true); o += 4; };
  for (const s of splats) {
    put(s.x); put(s.y); put(s.z);
    put(s.fdc[0]); put(s.fdc[1]); put(s.fdc[2]);
    put(s.opacity);
    put(s.scale[0]); put(s.scale[1]); put(s.scale[2]);
    put(s.rot[0]); put(s.rot[1]); put(s.rot[2]); put(s.rot[3]);
  }
  return buffer;
}
