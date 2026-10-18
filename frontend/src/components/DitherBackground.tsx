import { useEffect, useRef } from "react";

/**
 * 8x8 Bayer Ordered Dithering Matrix
 * Values 0..63 normalized to [0, 1) by dividing by 64.
 */
const BAYER_8X8 = [
   0, 32,  8, 40,  2, 34, 10, 42,
  48, 16, 56, 24, 50, 18, 58, 26,
  12, 44,  4, 36, 14, 46,  6, 38,
  60, 28, 52, 20, 62, 30, 54, 22,
   3, 35, 11, 43,  1, 33,  9, 41,
  51, 19, 59, 27, 49, 17, 57, 25,
  15, 47,  7, 39, 13, 45,  5, 37,
  63, 31, 55, 23, 61, 29, 53, 21,
];

interface DitherBackgroundProps {
  className?: string;
  /** Size of each pixel block. Defaults to 3 for crisp, retro-computational dither. */
  pixelSize?: number;
}

interface AntigravityParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  baseVy: number;
  colorType: "starlight" | "specular" | "cobalt";
  size: number;
  twinklePhase: number;
}

/**
 * DitherBackground Component
 *
 * Harmonized cosmic obsidian & midnight royal sapphire Bayer dither aesthetic:
 * - Serene, steady ambient radiant light pools with gentle harmonic waves
 * - Constellation of 220 upward-floating zero-gravity quantum stipples drifting calmly
 * - Multi-tier smoothstep Bayer luminance quantization (deep obsidian, midnight navy, sapphire, cobalt, starlight, specular)
 * - 60fps native performance with direct Uint32Array memory blits
 * - Pure ambient background with no intrusive cursor shadow, halo, or dither warping
 */
export function DitherBackground({
  className = "",
  pixelSize = 3,
}: DitherBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const particlesRef = useRef<AntigravityParticle[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: false });
    if (!ctx) return;

    let animationFrameId: number;
    let width = 0;
    let height = 0;
    let cols = 0;
    let rows = 0;
    let imgData: ImageData | null = null;
    let buf32: Uint32Array | null = null;

    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    // Helper: pack 32-bit unsigned ABGR
    const packABGR = (r: number, g: number, b: number, a: number = 255) =>
      (((a & 0xff) << 24) | ((b & 0xff) << 16) | ((g & 0xff) << 8) | (r & 0xff)) >>> 0;

    // Helper: smoothstep for silky continuous luminance and dither interpolation
    const smoothstep = (min: number, max: number, value: number) => {
      const x = Math.max(0, Math.min(1, (value - min) / (max - min)));
      return x * x * (3 - 2 * x);
    };

    // Harmonized cosmic obsidian & midnight royal sapphire palette
    const colVoid = packABGR(4, 6, 20, 255);          // #040614 deep obsidian void ground
    const colMid = packABGR(9, 14, 40, 255);          // #090e28 midnight navy
    const colSapphire = packABGR(18, 26, 72, 255);    // #121a48 dark royal sapphire
    const colCobalt = packABGR(37, 53, 117, 255);     // #253575 luminous celestial cobalt
    const colStarlight = packABGR(74, 114, 212, 255); // #4a72d4 soft celestial starlight blue
    const colSpecular = packABGR(165, 200, 255, 255); // #a5c8ff soft icy specular stardust

    // Initialize 220 floating zero-gravity quantum stipple motes
    const initParticles = () => {
      const count = 220;
      const list: AntigravityParticle[] = [];
      for (let i = 0; i < count; i++) {
        const typeRoll = Math.random();
        list.push({
          x: Math.random(),
          y: Math.random(),
          vx: (Math.random() - 0.5) * 0.0004,
          vy: -0.0004 - Math.random() * 0.001, // Zero gravity: continuous upward buoyancy
          baseVy: -0.0004 - Math.random() * 0.001,
          colorType: typeRoll > 0.6 ? "specular" : typeRoll > 0.25 ? "starlight" : "cobalt",
          size: Math.random() > 0.5 ? 2 : 1,
          twinklePhase: Math.random() * Math.PI * 2,
        });
      }
      particlesRef.current = list;
    };

    initParticles();

    const updateDimensions = () => {
      const parent = canvas.parentElement;
      const displayW = parent ? parent.clientWidth : window.innerWidth;
      const displayH = parent ? parent.clientHeight : window.innerHeight;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const effectivePixelSize = Math.max(3, Math.round(pixelSize * (dpr > 1.5 ? 1 : 1)));

      cols = Math.max(32, Math.ceil(displayW / effectivePixelSize));
      rows = Math.max(32, Math.ceil(displayH / effectivePixelSize));

      width = cols;
      height = rows;

      if (!imgData || !buf32 || canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        imgData = ctx.createImageData(width, height);
        buf32 = new Uint32Array(imgData.data.buffer);
      }
    };

    updateDimensions();

    const handleResize = () => {
      updateDimensions();
      renderFrame(performance.now());
    };

    window.addEventListener("resize", handleResize);

    // Main frame rendering loop (calm, steady, and unwarped)
    const renderFrame = (timestamp: number) => {
      if (!ctx || !imgData || !buf32 || cols <= 0 || rows <= 0) return;

      const now = timestamp || performance.now();
      const t = prefersReducedMotion ? 0 : now * 0.0007;

      // Radiant light pool centers
      const orb1X = 0.22 + Math.sin(t * 0.9) * 0.08;
      const orb1Y = 0.16 + Math.cos(t * 0.7) * 0.06;
      const orb1R = 0.58;

      const orb2X = 0.82 + Math.cos(t * 0.8) * 0.08;
      const orb2Y = 0.78 + Math.sin(t * 1.1) * 0.06;
      const orb2R = 0.65;

      const orb4X = 0.5 + Math.sin(t * 0.5) * 0.05;
      const orb4Y = 0.45 + Math.cos(t * 0.6) * 0.05;
      const orb4R = 0.42;

      const invW = 1.0 / cols;
      const invH = 1.0 / rows;

      let idx = 0;
      for (let y = 0; y < rows; y++) {
        const ny = y * invH;
        const bayerRow = (y & 7) << 3;

        for (let x = 0; x < cols; x++) {
          const nx = x * invW;

          // Distance to background radiant orbs
          const dx1 = nx - orb1X;
          const dy1 = ny - orb1Y;
          const d1 = Math.sqrt(dx1 * dx1 + dy1 * dy1) / orb1R;

          const dx2 = nx - orb2X;
          const dy2 = ny - orb2Y;
          const d2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) / orb2R;

          const dx4 = nx - orb4X;
          const dy4 = ny - orb4Y;
          const d4 = Math.sqrt(dx4 * dx4 + dy4 * dy4) / orb4R;

          // Smooth cubic falloff for radiant ambient orbs (velvety deep background)
          const l1 = Math.pow(Math.max(0, 1 - d1), 1.6) * 0.4;
          const l2 = Math.pow(Math.max(0, 1 - d2), 1.6) * 0.35;
          const l4 = Math.pow(Math.max(0, 1 - d4), 1.6) * 0.2;

          // Gentle, silky harmonic wave billows
          const wave1 = Math.sin(nx * 4.2 + t * 0.5) * Math.cos(ny * 3.6 - t * 0.4) * 0.07;
          const wave2 = Math.sin((nx + ny) * 6.2 - t * 0.6) * 0.04;

          // Gentle perimeter margin stippling
          const edgeDist = Math.min(nx, 1 - nx, ny, 1 - ny);
          const edgeGlow = Math.pow(1 - Math.min(1, edgeDist * 2.8), 2.0) * 0.16;

          // Combined luminance: steady, calm ambient field
          const lum = Math.max(
            0,
            l1 + l2 + l4 + wave1 + wave2 + edgeGlow,
          );

          // 8x8 Bayer matrix thresholding
          const bayerThreshold = BAYER_8X8[bayerRow | (x & 7)] * (1.0 / 64.0);

          // Continuous smoothstep Bayer dither quantization across 5 continuous tone ranges:
          // colVoid -> colMid -> colSapphire -> colCobalt -> colStarlight -> colSpecular
          if (lum < 0.18) {
            const factor = smoothstep(0.01, 0.18, lum);
            buf32[idx] = factor > bayerThreshold ? colMid : colVoid;
          } else if (lum < 0.38) {
            const factor = smoothstep(0.18, 0.38, lum);
            buf32[idx] = factor > bayerThreshold ? colSapphire : colMid;
          } else if (lum < 0.6) {
            const factor = smoothstep(0.38, 0.6, lum);
            buf32[idx] = factor > bayerThreshold ? colCobalt : colSapphire;
          } else if (lum < 0.82) {
            const factor = smoothstep(0.6, 0.82, lum);
            buf32[idx] = factor > bayerThreshold ? colStarlight : colCobalt;
          } else {
            const factor = smoothstep(0.82, 1.05, lum);
            buf32[idx] = factor > bayerThreshold ? colSpecular : colStarlight;
          }

          idx++;
        }
      }

      // -----------------------------------------------------------------
      // Floating Zero-Gravity Quantum Particles (Defying Gravity!)
      // -----------------------------------------------------------------
      const particles = particlesRef.current;
      const numParticles = particles.length;

      for (let i = 0; i < numParticles; i++) {
        const p = particles[i];

        // Defy gravity: continuous upward buoyancy
        p.vy += (p.baseVy - p.vy) * 0.04;

        // Apply friction and integrate position calmly (no cursor repulsion)
        p.vx *= 0.94;
        p.vy *= 0.97;
        p.x += p.vx;
        p.y += p.vy;

        // Wrap boundaries: wrap upward particles to the bottom
        if (p.y < -0.04) {
          p.y = 1.04;
          p.x = Math.random();
          p.vx = (Math.random() - 0.5) * 0.0004;
        } else if (p.y > 1.05) {
          p.y = -0.03;
        }
        if (p.x < -0.04) p.x = 1.04;
        else if (p.x > 1.04) p.x = -0.04;

        // Draw particle directly into buffer if on screen
        const px = Math.floor(p.x * cols);
        const py = Math.floor(p.y * rows);

        if (px >= 1 && px < cols - 1 && py >= 1 && py < rows - 1) {
          const pCol = p.colorType === "specular" ? colSpecular : p.colorType === "starlight" ? colStarlight : colCobalt;
          const pIdx = py * cols + px;
          buf32[pIdx] = pCol;

          // Cross sparkle halo for visible quantum motes
          if (p.size === 2) {
            buf32[pIdx + 1] = colCobalt;
            buf32[pIdx - 1] = colCobalt;
            buf32[pIdx + cols] = colCobalt;
            buf32[pIdx - cols] = colCobalt;
          }
        }
      }

      ctx.putImageData(imgData, 0, 0);

      animationFrameId = requestAnimationFrame(renderFrame);
    };

    animationFrameId = requestAnimationFrame(renderFrame);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", handleResize);
    };
  }, [pixelSize]);

  return (
    <div
      className={`dither-background-container ${className}`}
      aria-hidden="true"
    >
      <canvas
        ref={canvasRef}
        className="dither-canvas"
      />
      <div className="dither-radial-vignette" />
      <div className="dither-grid-lines" />
    </div>
  );
}
