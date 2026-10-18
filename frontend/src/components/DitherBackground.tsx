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
  /** Size of each pixel block. Defaults to 4 for crisp, bold, clearly discernible retro-computational dither. */
  pixelSize?: number;
}

interface AntigravityRipple {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  strength: number;
  speed: number;
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
 * DitherBackground Component with Antigravity Interactive Disturbance
 *
 * Harmonized cosmic obsidian & midnight royal sapphire Bayer dither aesthetic
 * enriched with real-time Antigravity interactive physics:
 * - Anti-gravity coordinate space repulsion bubble warping the dither mesh
 * - Aerodynamic comet wake trailing the cursor velocity
 * - Expanding high-contrast shockwave ripples on clicks and rapid strokes
 * - Constellation of 220 upward-floating zero-gravity quantum stipples (scattering on approach)
 * - Multi-tier smoothstep Bayer luminance quantization (deep obsidian, midnight navy, sapphire, cobalt, starlight, specular)
 * - 60fps native performance with direct Uint32Array memory blits
 */
export function DitherBackground({
  className = "",
  pixelSize = 3,
}: DitherBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointerRef = useRef({
    x: 0.5,
    y: 0.4,
    targetX: 0.5,
    targetY: 0.4,
    prevX: 0.5,
    prevY: 0.4,
    vx: 0,
    vy: 0,
    speed: 0,
    active: false,
    down: false,
    lastWakeTime: 0,
  });

  const ripplesRef = useRef<AntigravityRipple[]>([]);
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
    let lastTimestamp = performance.now();

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
          vy: -0.0004 - Math.random() * 0.001, // Negative gravity: continuous upward buoyancy!
          baseVy: -0.0004 - Math.random() * 0.001,
          colorType: typeRoll > 0.6 ? "specular" : typeRoll > 0.25 ? "starlight" : "cobalt",
          size: Math.random() > 0.5 ? 2 : 1,
          twinklePhase: Math.random() * Math.PI * 2,
        });
      }
      particlesRef.current = list;
    };

    initParticles();

    // Spawn an Antigravity disturbance shockwave ripple
    const triggerRipple = (x: number, y: number, strength = 1.4, maxRadius = 0.75) => {
      if (ripplesRef.current.length > 8) {
        ripplesRef.current.shift();
      }
      ripplesRef.current.push({
        x,
        y,
        radius: 0.02,
        maxRadius,
        strength,
        speed: 0.72,
      });
    };

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

    const handlePointerMove = (e: MouseEvent | PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;

      const ptr = pointerRef.current;
      const vx = nx - ptr.prevX;
      const vy = ny - ptr.prevY;
      const speed = Math.sqrt(vx * vx + vy * vy);

      ptr.targetX = nx;
      ptr.targetY = ny;
      ptr.vx = vx;
      ptr.vy = vy;
      ptr.speed = speed;
      ptr.prevX = nx;
      ptr.prevY = ny;
      ptr.active = true;

      // Antigravity wake: rapid movements emit shockwave pulses
      const now = performance.now();
      if (speed > 0.035 && now - ptr.lastWakeTime > 120) {
        ptr.lastWakeTime = now;
        triggerRipple(nx, ny, 1.1, 0.48);
      }
    };

    const handlePointerDown = (e: MouseEvent | PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width;
      const ny = (e.clientY - rect.top) / rect.height;

      const ptr = pointerRef.current;
      ptr.down = true;
      ptr.active = true;
      ptr.targetX = nx;
      ptr.targetY = ny;

      // Explosive Antigravity shockwave burst
      triggerRipple(nx, ny, 2.2, 0.95);

      // Violent particle scattering on click
      const particles = particlesRef.current;
      const aspect = (width || 1) / (height || 1);
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        const dx = p.x - nx;
        const dy = (p.y - ny) * aspect;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 0.45) {
          const impulse = Math.pow(1 - d / 0.45, 1.8) * 0.07;
          p.vx += (dx / (d + 0.01)) * impulse;
          p.vy += (dy / (d + 0.01)) * impulse;
        }
      }
    };

    const handlePointerUp = () => {
      pointerRef.current.down = false;
    };

    const handlePointerLeave = () => {
      const ptr = pointerRef.current;
      ptr.active = false;
      ptr.down = false;
      ptr.targetX = 0.5;
      ptr.targetY = 0.4;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("pointermove", handlePointerMove, { passive: true });
    window.addEventListener("pointerdown", handlePointerDown, { passive: true });
    window.addEventListener("pointerup", handlePointerUp, { passive: true });
    window.addEventListener("pointerleave", handlePointerLeave, { passive: true });

    // Main frame rendering loop
    const renderFrame = (timestamp: number) => {
      if (!ctx || !imgData || !buf32 || cols <= 0 || rows <= 0) return;

      const now = timestamp || performance.now();
      const dt = Math.min(0.05, (now - lastTimestamp) * 0.001);
      lastTimestamp = now;

      const t = prefersReducedMotion ? 0 : now * 0.0007;
      const aspect = width / height;

      // Smooth pointer lerp
      const ptr = pointerRef.current;
      ptr.x += (ptr.targetX - ptr.x) * 0.16;
      ptr.y += (ptr.targetY - ptr.y) * 0.16;

      // Decay velocity speed gently
      ptr.speed *= 0.92;

      // Update Antigravity shockwave ripples
      const ripples = ripplesRef.current;
      for (let r = ripples.length - 1; r >= 0; r--) {
        const rip = ripples[r];
        rip.radius += rip.speed * dt;
        if (rip.radius >= rip.maxRadius) {
          ripples.splice(r, 1);
        }
      }

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
      const numRipples = ripples.length;

      let idx = 0;
      for (let y = 0; y < rows; y++) {
        const ny = y * invH;
        const bayerRow = (y & 7) << 3;

        for (let x = 0; x < cols; x++) {
          const nx = x * invW;

          // -----------------------------------------------------------------
          // Antigravity Field Disturbance: Space-Time / Dither Mesh Warping
          // -----------------------------------------------------------------
          const dpx = nx - ptr.x;
          const dpy = (ny - ptr.y) * aspect;
          const pDist = Math.sqrt(dpx * dpx + dpy * dpy);

          let warpX = 0;
          let warpY = 0;
          let disturbanceGlow = 0;

          // Inverse-square Antigravity repulsion (pushes dither mesh outward)
          const repelRadius = ptr.down ? 0.38 : ptr.active ? 0.28 : 0.18;
          if (pDist < repelRadius && pDist > 0.001) {
            const rNorm = pDist / repelRadius;
            const force = Math.pow(1 - rNorm, 1.8) * (ptr.down ? 0.32 : 0.18) * (ptr.active ? 1.0 : 0.4);
            warpX += (dpx / pDist) * force;
            warpY += (dpy / pDist) * force;

            // Radiant energy ring at the perimeter of the anti-gravity bubble
            const crest = Math.sin(rNorm * Math.PI) * (ptr.down ? 0.85 : 0.52);
            disturbanceGlow += crest;
          }

          // Aerodynamic wake behind cursor
          if (ptr.speed > 0.012 && pDist < 0.32 && pDist > 0.01) {
            const wakeAlignment = -(dpx * ptr.vx + dpy * ptr.vy) / (pDist * (ptr.speed + 0.001));
            if (wakeAlignment > 0.15) {
              const wakeStrength = wakeAlignment * (1 - pDist / 0.32) * ptr.speed * 14.0;
              disturbanceGlow += Math.min(0.75, wakeStrength);
            }
          }

          // Shockwave ripple displacement & concentric high-contrast wavefronts
          for (let r = 0; r < numRipples; r++) {
            const rip = ripples[r];
            const rdx = nx - rip.x;
            const rdy = (ny - rip.y) * aspect;
            const rd = Math.sqrt(rdx * rdx + rdy * rdy);
            const delta = Math.abs(rd - rip.radius);
            const ringWidth = 0.085;
            if (delta < ringWidth && rd > 0.001) {
              const progress = rip.radius / rip.maxRadius;
              const fade = (1 - progress) * rip.strength;
              const env = (1 - delta / ringWidth) * fade;
              const wave = Math.sin((rd - rip.radius) * (Math.PI / ringWidth) * 2);
              warpX += (rdx / rd) * wave * env * 0.095;
              warpY += (rdy / rd) * wave * env * 0.095;
              disturbanceGlow += Math.max(0, env * 0.7);
            }
          }

          // Sample coordinates warped by the Antigravity disturbance field
          const snx = nx + warpX;
          const sny = ny + warpY;

          // Distance to background radiant orbs using warped coordinates
          const dx1 = snx - orb1X;
          const dy1 = sny - orb1Y;
          const d1 = Math.sqrt(dx1 * dx1 + dy1 * dy1) / orb1R;

          const dx2 = snx - orb2X;
          const dy2 = sny - orb2Y;
          const d2 = Math.sqrt(dx2 * dx2 + dy2 * dy2) / orb2R;

          const dx4 = snx - orb4X;
          const dy4 = sny - orb4Y;
          const d4 = Math.sqrt(dx4 * dx4 + dy4 * dy4) / orb4R;

          // Smooth cubic falloff for radiant ambient orbs (velvety deep background)
          const l1 = Math.pow(Math.max(0, 1 - d1), 1.6) * 0.4;
          const l2 = Math.pow(Math.max(0, 1 - d2), 1.6) * 0.35;
          const l4 = Math.pow(Math.max(0, 1 - d4), 1.6) * 0.2;

          // Gentle, silky harmonic wave billows
          const wave1 = Math.sin(snx * 4.2 + t * 0.5) * Math.cos(sny * 3.6 - t * 0.4) * 0.07;
          const wave2 = Math.sin((snx + sny) * 6.2 - t * 0.6) * 0.04;

          // Gentle perimeter margin stippling
          const edgeDist = Math.min(snx, 1 - snx, sny, 1 - sny);
          const edgeGlow = Math.pow(1 - Math.min(1, edgeDist * 2.8), 2.0) * 0.16;

          // Combined luminance with Antigravity interactive disturbance boost
          const lum = Math.max(
            0,
            l1 + l2 + l4 + wave1 + wave2 + edgeGlow + disturbanceGlow,
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

        // Repel from cursor when within range (Antigravity repulsion forcefield)
        const dx = p.x - ptr.x;
        const dy = (p.y - ptr.y) * aspect;
        const d = Math.sqrt(dx * dx + dy * dy);
        const pRepelRadius = ptr.down ? 0.34 : 0.22;

        if (d < pRepelRadius && d > 0.001) {
          const force = (1 - d / pRepelRadius) * (ptr.down ? 0.025 : 0.011);
          p.vx += (dx / d) * force;
          p.vy += (dy / d) * force;
        }

        // Apply friction and integrate position
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
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointerup", handlePointerUp);
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
