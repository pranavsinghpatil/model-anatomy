/**
 * Adapted from SmoothUI FluidHoverHighlight, MIT © 2024 Eduardo Calvo.
 * Pinned source and full license: frontend/THIRD_PARTY_NOTICES.md.
 * Keeps the upstream rect API; omits the unused hook adapter.
 */
import { AnimatePresence, useReducedMotion } from "motion/react";
import * as m from "motion/react-m";

export interface HighlightRect {
  height: number;
  width: number;
  left: number;
  top: number;
}

const toTarget = (rect: HighlightRect) => ({
  height: rect.height,
  width: rect.width,
  x: rect.left,
  y: rect.top,
});

export function FluidHoverHighlight({ rect }: { rect: HighlightRect | null }) {
  const reduceMotion = useReducedMotion();
  return (
    <AnimatePresence>
      {rect && (
        <m.div
          aria-hidden="true"
          className="workspace-nav-highlight"
          data-slot="fluid-hover-highlight"
          initial={{ opacity: 0, ...toTarget(rect) }}
          animate={{ opacity: 1, ...toTarget(rect) }}
          exit={{
            opacity: 0,
            transition: { duration: reduceMotion ? 0 : 0.08 },
          }}
          transition={
            reduceMotion
              ? { duration: 0 }
              : {
                  type: "spring",
                  bounce: 0,
                  duration: 0.16,
                  opacity: { duration: 0.08 },
                }
          }
        />
      )}
    </AnimatePresence>
  );
}
