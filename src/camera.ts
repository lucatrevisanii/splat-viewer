// Orbit camera: azimuth/elevation around a target, drag to rotate, wheel to zoom.

import { lookAt, type Mat4 } from "./math";

export class OrbitCamera {
  theta = 0.6; // azimuth
  phi = 0.35; // elevation
  radius = 4.5;
  maxRadius = 40;
  target: number[];

  constructor(target: number[] = [0.5, 0, 0]) {
    this.target = target;
  }

  // Frame a loaded scene: look at its center and back off to fit its extent.
  fit(center: number[], radius: number): void {
    this.target = center;
    this.maxRadius = Math.max(40, radius * 4);
    this.radius = Math.min(this.maxRadius, Math.max(0.6, radius));
  }

  eye(): number[] {
    const cp = Math.cos(this.phi);
    return [
      this.target[0] + this.radius * cp * Math.sin(this.theta),
      this.target[1] + this.radius * Math.sin(this.phi),
      this.target[2] + this.radius * cp * Math.cos(this.theta),
    ];
  }

  view(): Mat4 {
    return lookAt(this.eye(), this.target, [0, 1, 0]);
  }

  rotate(dx: number, dy: number): void {
    this.theta -= dx * 0.005;
    this.phi += dy * 0.005;
    const lim = Math.PI / 2 - 0.05;
    this.phi = Math.max(-lim, Math.min(lim, this.phi));
  }

  zoom(delta: number): void {
    this.radius *= Math.exp(delta * 0.0015);
    this.radius = Math.max(0.6, Math.min(this.maxRadius, this.radius));
  }
}
