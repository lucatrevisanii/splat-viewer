import { SplatRenderer } from "./renderer";
import { OrbitCamera } from "./camera";
import { perspective } from "./math";
import { parsePly, type SplatData } from "./ply";
import { parseSplat } from "./splat";
import { buildDemoScene } from "./scene";

// A real, captured 3DGS scene loaded on boot so the default view is a
// photographic reconstruction, not a synthetic shape. Public, permissively
// licensed, served with CORS (see README). The synthetic torus is the offline
// fallback if this fetch fails.
const REMOTE_SCENE = {
  url: "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/bonsai-7k-mini.splat",
  label: "bonsai · Mip-NeRF 360 (CC BY)",
};

const canvas = document.getElementById("view") as HTMLCanvasElement;
const hud = document.getElementById("count") as HTMLElement;
const fpsEl = document.getElementById("fps") as HTMLElement;
const srcEl = document.getElementById("src") as HTMLElement;
const drop = document.getElementById("drop") as HTMLElement;
const fileInput = document.getElementById("file") as HTMLInputElement;
const spinner = document.getElementById("spinner") as HTMLElement;
const spinnerText = document.getElementById("spinner-text") as HTMLElement;

const renderer = new SplatRenderer(canvas);
const camera = new OrbitCamera();
const FOV_Y = (50 * Math.PI) / 180;

let viewW = 0;
let viewH = 0;

function resize(): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  // Fall back to a sensible size when the element has no layout yet.
  const w = Math.floor((canvas.clientWidth || 1280) * dpr);
  const h = Math.floor((canvas.clientHeight || 720) * dpr);
  if (w === viewW && h === viewH) return;
  viewW = canvas.width = w;
  viewH = canvas.height = h;
}

// On-demand rendering: draw only when something changed.
let scheduled = false;
let lastDraw = performance.now();

function draw(): void {
  scheduled = false;
  resize();
  const now = performance.now();
  const dt = now - lastDraw;
  lastDraw = now;
  if (dt > 0 && dt < 1000) fpsEl.textContent = (1000 / dt).toFixed(0) + " fps";

  const proj = perspective(FOV_Y, viewW / viewH, 0.1, 100);
  const view = camera.view();
  const f = (0.5 * viewH) / Math.tan(FOV_Y / 2);
  renderer.render(view, proj, [f, f], [viewW, viewH]);
}

function requestRender(): void {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(draw);
}

function load(data: SplatData, label: string): void {
  renderer.setSplats(data);
  hud.textContent = data.count.toLocaleString() + " splats";
  srcEl.textContent = label;
  requestRender();
}

// Center the camera on the scene and back off to fit its extent. A robust
// radius (90th-percentile distance from the centroid, on a sample) ignores the
// stray "floater" gaussians real captures scatter far from the subject.
function fitCamera(data: SplatData): void {
  const p = data.positions;
  const n = data.count;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += p[i * 3]; cy += p[i * 3 + 1]; cz += p[i * 3 + 2]; }
  cx /= n; cy /= n; cz /= n;

  const stride = Math.max(1, Math.floor(n / 50000));
  const dists: number[] = [];
  for (let i = 0; i < n; i += stride) {
    dists.push(Math.hypot(p[i * 3] - cx, p[i * 3 + 1] - cy, p[i * 3 + 2] - cz));
  }
  dists.sort((a, b) => a - b);
  const r = dists[Math.floor(dists.length * 0.9)] || 1;
  camera.fit([cx, cy, cz], r * 1.6);
  renderer.markDirty();
}

// Pick the parser by extension, falling back to a header sniff: 3DGS .ply
// starts with the ASCII bytes "ply", everything else 32-byte-aligned is .splat.
function parseScene(name: string, buf: ArrayBuffer): SplatData {
  if (name.toLowerCase().endsWith(".splat")) return parseSplat(buf);
  if (name.toLowerCase().endsWith(".ply")) return parsePly(buf);
  const head = new Uint8Array(buf, 0, Math.min(3, buf.byteLength));
  const isPly = head[0] === 0x70 && head[1] === 0x6c && head[2] === 0x79;
  return isPly ? parsePly(buf) : parseSplat(buf);
}

// --- input ------------------------------------------------------------------

let dragging = false;
let lastX = 0;
let lastY = 0;

canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  camera.rotate(e.clientX - lastX, e.clientY - lastY);
  lastX = e.clientX;
  lastY = e.clientY;
  renderer.markDirty();
  requestRender();
});
canvas.addEventListener("pointerup", (e) => {
  dragging = false;
  canvas.releasePointerCapture(e.pointerId);
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  camera.zoom(e.deltaY);
  renderer.markDirty();
  requestRender();
}, { passive: false });

window.addEventListener("resize", requestRender);

// --- .ply loading -----------------------------------------------------------

async function loadFile(file: File): Promise<void> {
  try {
    const buf = await file.arrayBuffer();
    const data = parseScene(file.name, buf);
    fitCamera(data);
    load(data, file.name);
  } catch (err) {
    alert("Failed to parse splat file: " + (err as Error).message);
  }
}

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) loadFile(fileInput.files[0]);
});

for (const ev of ["dragenter", "dragover"]) {
  window.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("active");
  });
}
for (const ev of ["dragleave", "drop"]) {
  window.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "dragleave" && e.target !== document.documentElement) return;
    drop.classList.remove("active");
  });
}
window.addEventListener("drop", (e) => {
  const dt = (e as DragEvent).dataTransfer;
  if (dt && dt.files[0]) loadFile(dt.files[0]);
});

// --- boot -------------------------------------------------------------------

function loadFallbackScene(): void {
  const demo = buildDemoScene();
  camera.fit([0.5, 0, 0], 2.8);
  renderer.markDirty();
  load(parsePly(demo.ply), "synthetic demo (offline)");
  draw();
}

async function boot(): Promise<void> {
  try {
    const res = await fetch(REMOTE_SCENE.url);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const buf = await res.arrayBuffer();
    const data = parseSplat(buf);
    fitCamera(data);
    load(data, REMOTE_SCENE.label);
    draw();
    spinner.classList.add("hidden");
  } catch (err) {
    console.warn("remote scene failed, falling back to synthetic demo:", err);
    spinnerText.textContent = "remote scene unavailable — showing offline demo";
    loadFallbackScene();
    setTimeout(() => spinner.classList.add("hidden"), 1000);
  }
}

void boot();
