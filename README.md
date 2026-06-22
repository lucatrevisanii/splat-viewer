# splat-viewer

A 3D Gaussian Splatting viewer that runs in the browser. WebGL2 from scratch, zero runtime dependencies.

**Live demo:** https://lucatrevisanii.github.io/splat-viewer/

Drag to orbit, scroll to zoom, drop a `.ply` or `.splat` to load your own scene. It boots by fetching a real captured reconstruction — the **bonsai** scene from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) — so the default view is a photographic 3D scene, not a synthetic shape. If that fetch fails (offline), it falls back to a procedural scene of interlocked Gaussian tori.

## What it does

Loads a 3D Gaussian Splatting scene and renders it with EWA splatting. Two input formats are supported:

- **`.ply`** — the format produced by INRIA's [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), storing *pre-activation* parameters. The parser applies the standard activations: `scale = exp(s)`, `opacity = sigmoid(o)`, color from the degree-0 spherical-harmonic DC term.
- **`.splat`** — the compact 32-byte-per-splat format public WebGL scenes ship in ([antimatter15/splat](https://github.com/antimatter15/splat)), already activated: linear scale, `uint8` color/opacity, and a quantized quaternion. Coordinates are flipped from the COLMAP `+Y`-down convention to the viewer's `+Y`-up.

From either, each splat's 3D covariance `Σ = R S Sᵀ Rᵀ` is built from its rotation quaternion and scale, and:

- **Project** — in the vertex shader, each 3D covariance is projected to a 2D screen-space conic via the Jacobian of the perspective projection (`Σ' = J W Σ Wᵀ Jᵀ`, Zwicker et al., *EWA Splatting*), then decomposed into major/minor axes to size an instanced billboard.
- **Composite** — the fragment shader evaluates the 2D Gaussian and blends back-to-front with premultiplied-alpha "over". Splats are depth-sorted on the CPU with a 16-bit counting sort (O(n)) each time the camera moves, which keeps scenes of ~1M splats interactive.

Data is packed into a float texture and drawn with a single instanced quad, so the only per-splat vertex attribute is an index.

## Run

```bash
npm install
npm run dev      # local dev server
npm run build    # type-check + production build to dist/
npm run check    # headless verification (no GPU needed)
```

`npm run check` parses the demo scene through the real PLY path and asserts the invariants the renderer depends on: covariance math, `.ply` and `.splat` round-trips, positive-definite covariances (including after the `.splat` Y-reflection), the counting-sort depth order, the 2D projection, and the orbit camera — all without a browser.

## Scope

This is a focused renderer, not a full splatting pipeline. It does **not**:

- train or optimize splats (it only views pre-computed `.ply` / `.splat` files)
- evaluate view-dependent color (uses the SH degree-0 DC term only, no higher-order SH)
- GPU-sort (depth sort runs on the CPU; the counting sort keeps ~1M splats interactive, but a huge scene would still want a GPU radix sort)
- handle the `.ksplat` / SOG compressed variants (binary `.ply` and 32-byte `.splat` only)

## Credits

The math follows the 3D Gaussian Splatting paper (Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023) and the EWA splatting formulation (Zwicker, Pfister, van Baar, Gross, 2001). Renderer written from scratch in WebGL2.

The default scene is **bonsai** from the [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) (Barron et al., CC BY 4.0), pre-trained to Gaussian splats and served via [Hugging Face](https://huggingface.co/datasets/dylanebert/3dgs); it is fetched at runtime and not redistributed in this repository.

## License

MIT
