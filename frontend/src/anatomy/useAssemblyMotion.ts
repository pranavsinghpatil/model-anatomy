import { useEffect, useRef, useState } from "react";

export function useAssemblyMotion(
  target: number,
  container: React.RefObject<HTMLDivElement | null>,
) {
  const [pose, setPose] = useState(target);
  const current = useRef(target);
  const [visible, setVisible] = useState(true);
  const [still, setStill] = useState(
    () => matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    const change = () => setStill(media.matches);
    media.addEventListener("change", change);
    const observer = new IntersectionObserver(([entry]) =>
      setVisible(entry.isIntersecting && !document.hidden),
    );
    if (container.current) observer.observe(container.current);
    const visibility = () =>
      setVisible(
        !document.hidden && Boolean(container.current?.getClientRects().length),
      );
    document.addEventListener("visibilitychange", visibility);
    return () => {
      media.removeEventListener("change", change);
      observer.disconnect();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [container]);
  useEffect(() => {
    if (still || !visible) {
      current.current = target;
      setPose(target);
      return;
    }
    let frame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const dt = Math.min((now - previous) / 1000, 0.04);
      previous = now;
      current.current += (target - current.current) * (1 - Math.exp(-14 * dt));
      if (Math.abs(current.current - target) < 0.001) {
        current.current = target;
        setPose(target);
        return;
      }
      setPose(current.current);
      frame = requestAnimationFrame(tick);
    };
    if (Math.abs(current.current - target) > 0.001)
      frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, still, visible]);
  return { pose, still, visible };
}
