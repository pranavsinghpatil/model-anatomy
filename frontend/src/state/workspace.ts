import { useEffect, useState } from "react";
import type { Viewport } from "@xyflow/react";
import type { AnatomyView } from "../anatomy/model";
export type Crumb = { id: string; label: string };
export type GraphLayoutPreference = {
  direction: "DOWN" | "RIGHT";
  density: "compact" | "detailed";
  connectionLabels?: boolean;
};
export type WorkspaceSession = {
  path: Crumb[];
  selected: Record<string, string | null>;
  viewports: Record<string, Viewport>;
  graphLayouts: Record<string, GraphLayoutPreference>;
  scene: "graph" | "anatomy";
  anatomyViews: Record<string, AnatomyView>;
  inspectorTab: "module" | "tensors" | "evidence";
  inspectorOpen: boolean;
  questions: boolean;
  navigationOpen: boolean;
  layout: Record<string, number> | undefined;
};
const sessions = new Map<string, WorkspaceSession>();
function savedPreferences(id: string): Partial<WorkspaceSession> {
  try {
    const value = JSON.parse(
      sessionStorage.getItem(`anatomy-workspace:${id}`) ?? "null",
    );
    if (!value || value.version !== 1) return {};
    const graphLayouts = Object.fromEntries(
      Object.entries(value.graphLayouts ?? {}).filter(([, entry]) => {
        if (!entry || typeof entry !== "object") return false;
        const layout = entry as GraphLayoutPreference;
        return (
          ["DOWN", "RIGHT"].includes(layout.direction) &&
          ["compact", "detailed"].includes(layout.density)
        );
      }),
    ) as WorkspaceSession["graphLayouts"];
    return {
      graphLayouts,
      scene: value.scene === "anatomy" ? "anatomy" : "graph",
    };
  } catch {
    return {};
  }
}
export function useWorkspaceSession(id: string) {
  const [state, setState] = useState<WorkspaceSession>(() => {
    return (
      sessions.get(id) ?? {
        path: [],
        selected: {},
        viewports: {},
        graphLayouts: {},
        scene: "graph",
        anatomyViews: {},
        inspectorTab: "module",
        inspectorOpen: false,
        questions: false,
        navigationOpen: false,
        layout: undefined,
        ...savedPreferences(id),
      }
    );
  });
  useEffect(() => {
    sessions.delete(id);
    sessions.set(id, state);
    if (sessions.size > 20) sessions.delete(sessions.keys().next().value!);
  }, [id, state]);
  useEffect(() => {
    try {
      sessionStorage.setItem(
        `anatomy-workspace:${id}`,
        JSON.stringify({
          version: 1,
          scene: state.scene,
          graphLayouts: Object.fromEntries(
            Object.entries(state.graphLayouts).slice(-100),
          ),
        }),
      );
    } catch {
      /* Browsing remains available when storage is disabled. */
    }
  }, [id, state.scene, state.graphLayouts]);
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
