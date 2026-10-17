/**
 * Adapted from SmoothUI's Button Copy registry component:
 * https://github.com/educlopez/smoothui/blob/1534f0d39ab56e64b0ddd207bcc065b386b04d8e/packages/smoothui/components/button-copy/index.tsx
 * Copyright (c) 2024 Eduardo Calvo. MIT License; see frontend/THIRD_PARTY_NOTICES.md.
 * Adaptations: accepts a value and label, uses the app's copy-control styling,
 * reports clipboard errors visibly, and removes the artificial loading delay.
 */
import { AnimatePresence, useReducedMotion } from "motion/react";
import * as m from "motion/react-m";
import { Check, Copy, LoaderCircle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

export interface ButtonCopyProps {
  value: string;
  label?: string;
  duration?: number;
  disabled?: boolean;
}

type ButtonState = "idle" | "loading" | "success" | "error";

/** Copy a value with animated, accessible idle/loading/success feedback. */
export function ButtonCopy({
  value,
  label = "Copy",
  duration = 1600,
  disabled = false,
}: ButtonCopyProps) {
  const [state, setState] = useState<ButtonState>("idle");
  const reduceMotion = useReducedMotion();
  const requestId = useRef(0);
  const successTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );

  useEffect(
    () => () => {
      requestId.current += 1;
      if (successTimer.current) clearTimeout(successTimer.current);
    },
    [],
  );

  const copyValue = useCallback(async () => {
    const request = ++requestId.current;
    if (successTimer.current) clearTimeout(successTimer.current);
    setState("loading");

    try {
      await navigator.clipboard.writeText(value);
      if (request !== requestId.current) return;
      setState("success");
      successTimer.current = setTimeout(() => {
        if (request === requestId.current) setState("idle");
      }, duration);
    } catch {
      if (request === requestId.current) setState("error");
    }
  }, [duration, value]);

  const icon = {
    idle: <Copy size={14} />,
    loading: (
      <LoaderCircle
        className="animate-spin motion-reduce:animate-none"
        size={14}
      />
    ),
    success: <Check size={14} />,
    error: <Copy size={14} />,
  }[state];

  const ariaLabel = {
    idle: label,
    loading: "Copying...",
    success: "Copied",
    error: "Copy failed. Select and copy the text.",
  }[state];

  return (
    <span
      className="copy-control"
      data-state={
        state === "success" ? "copied" : state === "error" ? "failed" : state
      }
    >
      <button
        type="button"
        className="button button-ghost"
        aria-label={ariaLabel}
        aria-live="polite"
        disabled={disabled || state === "loading" || state === "success"}
        onClick={copyValue}
      >
        <AnimatePresence initial={false} mode="popLayout">
          <m.span
            aria-hidden="true"
            className="copy-glyph"
            key={state}
            initial={
              reduceMotion
                ? { opacity: 1 }
                : { filter: "blur(10px)", opacity: 0, y: -12 }
            }
            animate={
              reduceMotion
                ? { opacity: 1 }
                : { filter: "blur(0px)", opacity: 1, y: 0 }
            }
            exit={
              reduceMotion
                ? { opacity: 0, transition: { duration: 0 } }
                : { filter: "blur(8px)", opacity: 0, y: 12 }
            }
            transition={
              reduceMotion
                ? { duration: 0 }
                : { bounce: 0, duration: 0.2, type: "spring" }
            }
          >
            {icon}
          </m.span>
        </AnimatePresence>
      </button>
      <span role="status">
        {state === "success"
          ? "Copied"
          : state === "error"
            ? "Copy failed. Select and copy the text."
            : ""}
      </span>
    </span>
  );
}
