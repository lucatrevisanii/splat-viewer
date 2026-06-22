# splat-viewer

A 3D Gaussian Splatting viewer that runs in the browser. WebGL2 from scratch, zero runtime dependencies.

**Live demo:** https://lucatrevisanii.github.io/splat-viewer/

Drag to orbit, scroll to zoom, drop a `.ply` to load your own scene. It boots with a synthetic scene (two interlocked tori of anisotropic Gaussians) so there's something to look at without a file.

## What it does

Loads a 3D Gaussian Splatting `.ply` (the format produced by INRIA's [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)) and renders it with EWA splatting:

- **Parse** — reads the binary PLY, applies the standard activations: `scale = exp(s)`, `opacity = sigmoid(o)`, color from the degree-0 spherical-harmonic DC term, and builds each splat's 3D covariance `Σ = R S Sᵀ Rᵀ` from its rotation quaternion and scale.
- **Project** — in the vertex shader, each 3D covariance is projected to a 2D screen-space conic via the Jacobian of the perspective projection (`Σ' = J W Σ Wᵀ Jᵀ`, Zwicker et al., *EWA Splatting*), then decomposed into major/minor axes to size an instanced billboard.
- **Composite** — the fragment shader evaluates the 2D Gaussian and blends back-to-front with premultiplied-alpha "over". Splats are depth-sorted on the CPU each time the camera moves.

Data is packed into a float texture and drawn with a single instanced quad, so the only per-splat vertex attribute is an index.

## Run

```bash
npm install
npm run dev      # local dev server
npm run build    # type-check + production build to dist/
npm run check    # headless verification (no GPU needed)
```

`npm run check` parses the demo scene through the real PLY path and asserts the invariants the renderer depends on, covariance math, PLY round-trip, positive-definite covariances, the 2D projection, and the orbit camera, all without a browser.

## Scope

This is a focused renderer, not a full splatting pipeline. It does **not**:

- train or optimize splats (it only views pre-computed `.ply` files)
- evaluate view-dependent color (uses the SH degree-0 DC term only, no higher-order SH)
- GPU-sort (depth sort runs on the CPU, fine for scenes up to a few hundred thousand splats)
- handle Token-2022-style compressed `.splat`/`.ksplat` variants (binary PLY only)

## Credits

The math follows the 3D Gaussian Splatting paper (Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023) and the EWA splatting formulation (Zwicker, Pfister, van Baar, Gross, 2001). Renderer written from scratch in WebGL2.

## License

MIT
