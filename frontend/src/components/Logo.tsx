import type { SVGProps } from "react";

type MarkProps = SVGProps<SVGSVGElement> & {
  size?: number;
};

/**
 * Model Anatomy Emblem Mark
 * An isometric axonometric tensor cube with dissected neural facets:
 * - Top facet: Architecture graph topology (indigo/violet)
 * - Left facet: Tensor dimensions & weights (cyan/sky)
 * - Right facet: Artifact provenance & deterministic evidence (royal blue)
 * - Center: Neural junction core node
 */
export function ModelAnatomyMark({
  size = 24,
  className,
  ...props
}: MarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
      {...props}
    >
      <defs>
        {/* Top facet gradient - Architecture topology */}
        <linearGradient
          id="ma-grad-top"
          x1="3"
          y1="3"
          x2="29"
          y2="18"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="#818cf8" />
          <stop offset="100%" stopColor="#6366f1" />
        </linearGradient>

        {/* Left facet gradient - Tensor dimensions */}
        <linearGradient
          id="ma-grad-left"
          x1="3"
          y1="10.5"
          x2="16"
          y2="33"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="100%" stopColor="#0284c7" />
        </linearGradient>

        {/* Right facet gradient - Provenance & evidence */}
        <linearGradient
          id="ma-grad-right"
          x1="16"
          y1="18"
          x2="29"
          y2="33"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="#4f46e5" />
          <stop offset="100%" stopColor="#312e81" />
        </linearGradient>

        {/* Specular line gradient */}
        <linearGradient
          id="ma-specular"
          x1="16"
          y1="3"
          x2="16"
          y2="33"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
          <stop offset="50%" stopColor="#a5b4fc" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#6366f1" stopOpacity="0.2" />
        </linearGradient>

        {/* Center core glow */}
        <radialGradient
          id="ma-glow"
          cx="18"
          cy="18"
          r="8"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.9" />
          <stop offset="50%" stopColor="#818cf8" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Top Facet */}
      <path
        d="M 18 3.5 L 31.5 11.25 L 18 19 L 4.5 11.25 Z"
        fill="url(#ma-grad-top)"
        fillOpacity="0.95"
      />

      {/* Left Facet */}
      <path
        d="M 4.5 11.25 L 18 19 L 18 32.5 L 4.5 24.75 Z"
        fill="url(#ma-grad-left)"
        fillOpacity="0.9"
      />

      {/* Right Facet */}
      <path
        d="M 18 19 L 31.5 11.25 L 31.5 24.75 L 18 32.5 Z"
        fill="url(#ma-grad-right)"
        fillOpacity="0.95"
      />

      {/* Inner Tensor Lattice Lines */}
      <path
        d="M 11.25 7.375 L 24.75 15.125"
        stroke="#ffffff"
        strokeOpacity="0.35"
        strokeWidth="1"
        strokeLinecap="round"
      />
      <path
        d="M 24.75 7.375 L 11.25 15.125"
        stroke="#ffffff"
        strokeOpacity="0.35"
        strokeWidth="1"
        strokeLinecap="round"
      />

      {/* Facet Edge Seams */}
      <path
        d="M 18 3.5 L 18 19"
        stroke="url(#ma-specular)"
        strokeWidth="1.25"
        strokeLinecap="round"
      />
      <path
        d="M 4.5 11.25 L 18 19"
        stroke="url(#ma-specular)"
        strokeWidth="1.25"
        strokeLinecap="round"
      />
      <path
        d="M 31.5 11.25 L 18 19"
        stroke="url(#ma-specular)"
        strokeWidth="1.25"
        strokeLinecap="round"
      />
      <path
        d="M 18 19 L 18 32.5"
        stroke="url(#ma-specular)"
        strokeWidth="1.25"
        strokeLinecap="round"
      />

      {/* Outer Contour */}
      <path
        d="M 18 3.5 L 31.5 11.25 L 31.5 24.75 L 18 32.5 L 4.5 24.75 L 4.5 11.25 Z"
        stroke="#ffffff"
        strokeOpacity="0.25"
        strokeWidth="1"
        strokeLinejoin="round"
      />

      {/* Central Junction Core */}
      <circle cx="18" cy="19" r="3.5" fill="url(#ma-glow)" />
      <circle cx="18" cy="19" r="1.5" fill="#ffffff" />

      {/* Neural Node Points */}
      <circle cx="18" cy="3.5" r="1.2" fill="#ffffff" fillOpacity="0.9" />
      <circle cx="31.5" cy="11.25" r="1.2" fill="#ffffff" fillOpacity="0.9" />
      <circle cx="4.5" cy="11.25" r="1.2" fill="#ffffff" fillOpacity="0.9" />
      <circle cx="18" cy="32.5" r="1.2" fill="#38bdf8" fillOpacity="0.9" />
    </svg>
  );
}

type LogoProps = {
  markSize?: number;
  showTag?: boolean;
  tagText?: string;
  className?: string;
};

/**
 * Model Anatomy Complete Brand Lockup
 */
export function ModelAnatomyLogo({
  markSize = 22,
  showTag = false,
  tagText = "STUDIO",
  className = "",
}: LogoProps) {
  return (
    <span className={`model-anatomy-lockup ${className}`}>
      <span className="brand-gem" aria-hidden="true">
        <ModelAnatomyMark size={markSize} />
      </span>
      <span className="brand-wordmark">
        <span className="brand-word-model">Model</span>
        <span className="brand-word-anatomy">Anatomy</span>
      </span>
      {showTag && <span className="brand-tag">{tagText}</span>}
    </span>
  );
}
