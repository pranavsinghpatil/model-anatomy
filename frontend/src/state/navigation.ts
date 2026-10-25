import {
  useCallback,
  useEffect,
  useRef,
  type Dispatch,
  type SetStateAction,
} from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { GraphIndex, Citation } from "../api/contract";
import {
  useWorkspaceSession,
  type Crumb,
  type WorkspaceSession,
} from "./workspace";

const lookups = new WeakMap<
  GraphIndex,
  Map<string, GraphIndex["graphs"][number]>
>();
export function graphPath(index: GraphIndex, id: string): Crumb[] | null {
  let records = lookups.get(index);
  if (!records) {
    records = new Map(index.graphs.map((item) => [item.id, item]));
    lookups.set(index, records);
  }
  const path: Crumb[] = [];
  const visited = new Set<string>();
  let current: string | null = id;
  while (current !== null) {
    if (visited.has(current)) return null;
    visited.add(current);
    const record = records.get(current);
    if (!record) return null;
    path.push({ id: record.id, label: record.label });
    current = record.parent_graph_id;
  }
  path.reverse();
  return path[0]?.id === index.root_graph_id ? path : null;
}
export function citationTarget(index: GraphIndex, citation: Citation) {
  if (citation.source === "stored_ir") {
    const node = index.nodes.find(
      (item) => citation.locator === `node:${item.graph_id}/${item.id}`,
    );
    if (node) return { graph_id: node.graph_id, node_id: node.id };
  }
  return (
    index.citations.find(
      (item) =>
        item.source === citation.source &&
        item.kind === citation.kind &&
        item.locator === citation.locator,
    ) ?? null
  );
}
export function viewHref(search: URLSearchParams, view: string) {
  const next = new URLSearchParams(search);
  next.set("view", view);
  next.delete("awaiting");
  return `?${next}`;
}
export const graphIndexQuery = (id: string) => ({
  queryKey: ["graph-index", id],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.graphIndex(id, signal),
  staleTime: Infinity,
});

export function useArchitectureNavigation(
  id: string,
  root: Crumb | null,
  ready: boolean,
) {
  const [base, setBase] = useWorkspaceSession(id);
  const [search, setSearch] = useSearchParams();
  const graphId = search.get("graph") ?? root?.id;
  const nodeId = search.get("node");
  const nested = Boolean(root && graphId !== root.id);
  const index = useQuery({ ...graphIndexQuery(id), enabled: ready && nested });
  const known = base.path.at(-1)?.id === graphId ? base.path : null;
  const path = nested
    ? index.data && graphId
      ? graphPath(index.data, graphId)
      : known
    : [];
  const state: WorkspaceSession = {
    ...base,
    path: path ?? [],
    selected: { ...base.selected, ...(graphId ? { [graphId]: nodeId } : {}) },
  };
  const current = useRef(state);
  current.current = state;
  const setState: Dispatch<SetStateAction<WorkspaceSession>> = useCallback(
    (update) => {
      const previous = current.current;
      const next = typeof update === "function" ? update(previous) : update;
      if (next === previous) return;
      current.current = next;
      setBase(next);
      const before = previous.path.at(-1)?.id ?? root?.id;
      const after = next.path.at(-1)?.id ?? root?.id;
      const selected = after ? (next.selected[after] ?? null) : null;
      if (
        after &&
        (before !== after || (before && previous.selected[before] !== selected))
      ) {
        // Diagram callbacks can finish after another view has navigated.
        // Preserve the current URL instead of the callback's captured query.
        if (
          window.location.pathname !== `/app/analyses/${encodeURIComponent(id)}`
        )
          return;
        const result = new URLSearchParams(window.location.search);
        if (after === root?.id) result.delete("graph");
        else result.set("graph", after);
        if (selected) result.set("node", selected);
        else result.delete("node");
        setSearch(result, { replace: before === after });
      }
    },
    [id, setBase, setSearch, root?.id],
  );
  const initialFocus = useRef(true);
  useEffect(() => {
    if (!ready || !root) return;
    if (initialFocus.current) {
      initialFocus.current = false;
      if (nodeId) setBase((previous) => ({ ...previous, inspectorOpen: true }));
    }
  }, [ready, root?.id, nodeId, setBase]);
  const error =
    ready && nested
      ? index.data && !path
        ? new Error("This graph is not part of the stored model hierarchy.")
        : !known
          ? index.error
          : null
      : null;
  return {
    state,
    setState,
    graphId,
    restoring: ready && nested && !path && !error,
    error,
    retry: () => index.refetch(),
    reset: () => {
      setBase((previous) => ({ ...previous, path: [], inspectorOpen: false }));
      setSearch(
        (params) => {
          const next = new URLSearchParams(params);
          next.delete("graph");
          next.delete("node");
          return next;
        },
        { replace: true },
      );
    },
  };
}
