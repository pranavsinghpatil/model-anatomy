/**
 * Adapted from Magic UI's DotPattern registry component.
 * Source: https://github.com/magicuidesign/magicui/blob/efe76fc5755eca1142dd4f74c1d6f0ace2887e3c/apps/www/registry/magicui/dot-pattern.tsx
 * Copyright (c) Magic UI. MIT License; see frontend/THIRD_PARTY_NOTICES.md.
 * Adaptations: static SVG pattern, currentColor, and no client sizing or glow animation.
 */
import { useId, type SVGProps } from "react";

export interface DotPatternProps extends SVGProps<SVGSVGElement> {
  width?: number;
  height?: number;
  x?: number;
  y?: number;
  cx?: number;
  cy?: number;
  cr?: number;
}

/** A lightweight, decorative SVG dot pattern that fills its container. */
export function DotPattern({
  width = 16,
  height = 16,
  x = 0,
  y = 0,
  cx = 1,
  cy = 1,
  cr = 1,
  className,
  ...props
}: DotPatternProps) {
  const patternId = useId();

  return (
    <svg
      className={className}
      focusable="false"
      preserveAspectRatio="none"
      width="100%"
      height="100%"
      {...props}
      aria-hidden="true"
    >
      <defs>
        <pattern
          id={patternId}
          width={width}
          height={height}
          patternUnits="userSpaceOnUse"
          x={x}
          y={y}
        >
          <circle cx={cx} cy={cy} r={cr} fill="currentColor" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${patternId})`} />
    </svg>
  );
}
