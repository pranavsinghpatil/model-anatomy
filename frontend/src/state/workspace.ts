import { useEffect, useState } from "react";
import type { Viewport } from "@xyflow/react";
export type Crumb = { id: string; label: string };
export type WorkspaceSession = {
  path: Crumb[];
  selected: Record<string, string | null>;
  viewports: Record<string, Viewport>;
  inspectorTab: "module" | "tensors" | "evidence";
  inspectorOpen: boolean;
  questions: boolean;
  navigationOpen: boolean;
  layout: Record<string, number> | undefined;
};
const sessions = new Map<string, WorkspaceSession>();
export function useWorkspaceSession(id: string) {
  const [state, setState] = useState<WorkspaceSession>(
    () =>
      sessions.get(id) ?? {
        path: [],
        selected: {},
        viewports: {},
        inspectorTab: "module",
        inspectorOpen: false,
        questions: false,
        navigationOpen: true,
        layout: undefined,
      },
  );
  useEffect(() => {
    sessions.set(id, state);
  }, [id, state]);
  return [state, setState] as const;
}
export function useMedia(query: string) {
  const [matches, setMatches] = useState(() => matchMedia(query).matches);
  useEffect(() => {
    const media = matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);
  return matches;
}
