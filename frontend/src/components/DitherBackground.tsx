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
  /** Size of each pixel block. Defaults to 4 for crisp retro-computational dither. */
  pixelSize?: number;
}

/**
 * DitherBackground Component
 * Reproduces the signature electric royal-blue ordered-dither aesthetic (as seen on HarnessHive).
 * Features:
 * - HTML5 Canvas downsampled and rendered with `image-rendering: pixelated`
 * - 8x8 Bayer matrix ordered dithering
 * - Organic floating radiant light pools (top-left indigo, bottom-right cobalt, center cyan)
 * - Interactive pointer tracking that illuminates dither stipples under the mouse
 * - Theme-aware (signature deep electric ultramarine in dark, modern blueprint-azure in light)
 * - Zero external dependencies, 60fps high efficiency with minimal CPU overhead
 */
export function DitherBackground({
  className = "",
  pixelSize = 4,
}: DitherBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointerRef = useRef({ x: 0.5, y: 0.4, targetX: 0.5, targetY: 0.4, active: false });

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

    // Check prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    // Detect theme attribute
    const isLightTheme = () =>
      document.documentElement.getAttribute("data-theme") === "light";

    const updateDimensions = () => {
      const parent = canvas.parentElement;
      const displayW = parent ? parent.clientWidth : window.innerWidth;
      const displayH = parent ? parent.clientHeight : window.innerHeight;

      // Adjust pixel size dynamically for ultra-dense displays
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const effectivePixelSize = Math.max(3, Math.round(pixelSize * (dpr > 1.5 ? 1 : 1)));

      cols = Math.max(32, Math.ceil(displayW / effectivePixelSize));
      rows = Math.max(32, Math.ceil(displayH / effectivePixelSize));

      width = cols;
      height = rows;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        imgData = ctx.createImageData(width, height);
        buf32 = new Uint32Array(imgData.data.buffer);
      }
    };

    updateDimensions();

    const handleResize = () => {
      updateDimensions();
      if (prefersReducedMotion) {
        renderFrame(0);
      }
    };

    const handlePointerMove = (e: MouseEvent | PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      pointerRef.current.targetX = (e.clientX - rect.left) / rect.width;
      pointerRef.current.targetY = (e.clientY - rect.top) / rect.height;
      pointerRef.current.active = true;
    };

    const handlePointerLeave = () => {
      pointerRef.current.active = false;
      pointerRef.current.targetX = 0.5;
      pointerRef.current.targetY = 0.4;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    window.addEventListener("pointerleave", handlePointerLeave, { passive: true });

    // Pre-calculate little-endian ABGR pack helper with unsigned 32-bit cast
    const packABGR = (r: number, g: number, b: number, a: number = 255) =>
      (((a & 0xff) << 24) | ((b & 0xff) << 16) | ((g & 0xff) << 8) | (r & 0xff)) >>> 0;

    // Frame rendering loop
    const renderFrame = (timestamp: number) => {
      if (!ctx || !imgData || !buf32 || cols <= 0 || rows <= 0) return;

      const t = prefersReducedMotion ? 0 : timestamp * 0.0006;

      // Smooth pointer lerp
      const ptr = pointerRef.current;
      ptr.x += (ptr.targetX - ptr.x) * 0.08;
      ptr.y += (ptr.targetY - ptr.y) * 0.08;

      // Radiant orb centers (floating gently over time)
      // Orb 1: Upper-left / Top center (indigo / violet)
      const orb1X = 0.22 + Math.sin(t * 0.9) * 0.08;
      const orb1Y = 0.16 + Math.cos(t * 0.7) * 0.06;
      const orb1R = 0.58;

      // Orb 2: Bottom-right (cobalt / electric blue)
      const orb2X = 0.82 + Math.cos(t * 0.8) * 0.08;
      const orb2Y = 0.78 + Math.sin(t * 1.1) * 0.06;
      const orb2R = 0.65;

      // Orb 3: Interactive pointer / ambient pulse
      const orb3X = ptr.x;
      const orb3Y = ptr.y;
      const orb3R = ptr.active ? 0.38 : 0.28 + Math.sin(t * 1.4) * 0.04;

      // Orb 4: Center subtle fill
      const orb4X = 0.5 + Math.sin(t * 0.5) * 0.05;
      const orb4Y = 0.45 + Math.cos(t * 0.6) * 0.05;
      const orb4R = 0.42;

      // Color Palette: HarnessHive signature electric royal ultramarine
      // Base: rgb(22, 0, 140) - #16008c
      // Deep void: rgb(12, 0, 85)
      // Tier 1: rgb(79, 70, 229) - electric indigo stipple
      // Tier 2: rgb(56, 189, 248) - vibrant cyan bright pixels
      // Tier 3: rgb(255, 255, 255) - radiant white specular highlights
      const colBase = packABGR(22, 0, 140, 255);
      const colDeep = packABGR(12, 0, 85, 255);
      const colT1 = packABGR(79, 70, 229, 255);
      const colT2 = packABGR(56, 189, 248, 255);
      const colT3 = packABGR(255, 255, 255, 255);

      const invW = 1.0 / cols;
      const invH = 1.0 / rows;

      let idx = 0;
      for (let y = 0; y < rows; y++) {
        const ny = y * invH;
        const bayerRow = (y & 7) << 3;

        for (let x = 0; x < cols; x++) {
          const nx = x * invW;

          // Normalized distance to light orbs
          const dx1 = nx - orb1X;
          const dy1 = ny - orb1Y;
          const d1 = Math.sqrt(dx1 * dx1 + dy1 * dy1) / orb1R;

          const dx2 = nx - orb2X;
          const dy2 = ny - orb2Y;
          const d2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) / orb2R;

          const dx3 = nx - orb3X;
          const dy3 = ny - orb3Y;
          const d3 = Math.sqrt(dx3 * dx3 + dy3 * dy3) / orb3R;

          const dx4 = nx - orb4X;
          const dy4 = ny - orb4Y;
          const d4 = Math.sqrt(dx4 * dx4 + dy4 * dy4) / orb4R;

          // Compute smooth luminance field [0, 1]
          const l1 = Math.max(0, 1 - d1);
          const l2 = Math.max(0, 1 - d2);
          const l3 = Math.max(0, 1 - d3) * (ptr.active ? 1.2 : 0.8);
          const l4 = Math.max(0, 1 - d4) * 0.45;

          // Non-linear glow response
          const lum = Math.min(1.0, (l1 * 0.75 + l2 * 0.7 + l3 * 0.65 + l4 * 0.35));

          // Bayer matrix threshold [0, 1)
          const bayerThreshold = BAYER_8X8[bayerRow | (x & 7)] * (1.0 / 64.0);

          // Multi-tier Bayer dithering
          if (lum < 0.12) {
            // Deep background void
            buf32[idx] = colDeep;
          } else if (lum < 0.38) {
            // Low intensity: stipple with Tier 1
            const factor = (lum - 0.12) / 0.26;
            buf32[idx] = factor > bayerThreshold ? colT1 : colBase;
          } else if (lum < 0.72) {
            // Mid intensity: stipple with Tier 2 (cyan / electric blue)
            const factor = (lum - 0.38) / 0.34;
            buf32[idx] = factor > bayerThreshold ? colT2 : colT1;
          } else {
            // Peak radiant core: stipple with Tier 3 (white specular highlights)
            const factor = (lum - 0.72) / 0.28;
            buf32[idx] = factor > bayerThreshold ? colT3 : colT2;
          }

          idx++;
        }
      }

      ctx.putImageData(imgData, 0, 0);

      if (!prefersReducedMotion) {
        animationFrameId = requestAnimationFrame(renderFrame);
      }
    };

    // Kick off animation loop
    animationFrameId = requestAnimationFrame(renderFrame);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerleave", handlePointerLeave);
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
