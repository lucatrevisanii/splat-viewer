import { SplatRenderer } from "./renderer";
import { OrbitCamera } from "./camera";
import { perspective } from "./math";
import { parsePly, type SplatData } from "./ply";
import { buildDemoScene } from "./scene";

const canvas = document.getElementById("view") as HTMLCanvasElement;
const hud = document.getElementById("count") as HTMLElement;
const fpsEl = document.getElementById("fps") as HTMLElement;
const drop = document.getElementById("drop") as HTMLElement;
const fileInput = document.getElementById("file") as HTMLInputElement;

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

function load(data: SplatData): void {
  renderer.setSplats(data);
  hud.textContent = data.count.toLocaleString() + " splats";
  requestRender();
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
    const data = parsePly(buf);
    camera.target = [0, 0, 0];
    renderer.markDirty();
    load(data);
  } catch (err) {
    alert("Failed to parse PLY: " + (err as Error).message);
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

const demo = buildDemoScene();
load(parsePly(demo.ply));
draw(); // synchronous first frame so the canvas has content immediately
