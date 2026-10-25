import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronRight } from "lucide-react";
import { api } from "../api/client";
import type { GraphNode } from "../api/contract";
import type { Crumb } from "../state/workspace";
import { graphIndexQuery, graphPath } from "../state/navigation";
import { nodeTitle } from "../graph/presentation";
import { ErrorNotice, Loading } from "./Common";

type Props = {
  analysisId: string;
  root: Crumb;
  currentPath: Crumb[];
  selectedId?: string;
  navigate: (path: Crumb[], node: GraphNode) => void;
};

export function ComponentTree(props: Props) {
  const active = useRef(true);
  useEffect(() => {
    active.current = true;
    return () => {
      active.current = false;
    };
  }, []);
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(40);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const client = useQueryClient();
  const search = query.trim().toLowerCase();
  const index = useQuery({
    ...graphIndexQuery(props.analysisId),
    enabled: search.length >= 2,
  });
  const matches =
    index.data?.nodes.filter((node) =>
      `${node.label} ${node.module_path ?? ""} ${node.kind}`
        .toLowerCase()
        .includes(search),
    ) ?? [];
  return (
    <div className="component-outline model-component-tree">
      <label className="model-tree-search">
        <span>Find across model</span>
        <input
          type="search"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setLimit(40);
          }}
          placeholder="Module name, type, or path"
        />
      </label>
      {error && <ErrorNotice error={error} />}
      {search.length >= 2 ? (
        <>
          {index.isPending && <Loading text="Loading model locations" />}
          {index.error && (
            <ErrorNotice
              error={index.error}
              retry={() => void index.refetch()}
            />
          )}
          {index.data && (
            <p className="outline-scope">
              {matches.length} matching components
            </p>
          )}
          <ul>
            {matches.slice(0, limit).map((node) => (
              <li key={`${node.graph_id}:${node.id}`}>
                <button
                  type="button"
                  className="component-select"
                  disabled={opening}
                  onClick={async () => {
                    setOpening(true);
                    setError(null);
                    try {
                      const graph = await client.fetchQuery({
                        queryKey: ["graph", props.analysisId, node.graph_id],
                        queryFn: ({ signal }) =>
                          api.graph(props.analysisId, node.graph_id, signal),
                        staleTime: Infinity,
                      });
                      const actual = graph.nodes.find(
                        (item) => item.id === node.id,
                      );
                      const path = graphPath(index.data!, node.graph_id);
                      if (!actual || !path)
                        throw new Error(
                          "The component location could not be verified.",
                        );
                      if (active.current) props.navigate(path, actual);
                    } catch (cause) {
                      setError(
                        cause instanceof Error
                          ? cause
                          : new Error("Could not open component"),
                      );
                    } finally {
                      if (active.current) setOpening(false);
                    }
                  }}
                >
                  <span>
                    <strong>{node.label}</strong>
                    <small>{node.module_path ?? node.kind}</small>
                  </span>
                </button>
              </li>
            ))}
          </ul>
          {matches.length > limit && (
            <button
              type="button"
              className="button button-ghost"
              onClick={() => setLimit(limit + 40)}
            >
              Show more matches
            </button>
          )}
        </>
      ) : (
        <>
          <p className="outline-scope">
            Registered module hierarchy · expand to load children
          </p>
          <TreeBranch {...props} path={[props.root]} />
        </>
      )}
    </div>
  );
}

function TreeBranch({ path, ...props }: Props & { path: Crumb[] }) {
  const graphId = path.at(-1)!.id;
  const graph = useQuery({
    queryKey: ["graph", props.analysisId, graphId],
    queryFn: ({ signal }) => api.graph(props.analysisId, graphId, signal),
    staleTime: Infinity,
  });
  const [visible, setVisible] = useState(80);
  const [expanded, setExpanded] = useState<Set<string>>(
    () => new Set(props.currentPath.map((item) => item.id)),
  );
  useEffect(() => {
    setExpanded(
      (previous) =>
        new Set([...previous, ...props.currentPath.map((item) => item.id)]),
    );
  }, [props.currentPath]);
  if (graph.error)
    return (
      <ErrorNotice error={graph.error} retry={() => void graph.refetch()} />
    );
  if (!graph.data) return <Loading text="Loading module children" />;
  const activeGraph = props.currentPath.at(-1)?.id ?? props.root.id;
  const ancestorIndex = graph.data.nodes.findIndex(
    (node) =>
      node.child_graph_id &&
      props.currentPath.some((crumb) => crumb.id === node.child_graph_id),
  );
  const limit = Math.max(visible, ancestorIndex + 1);
  return (
    <ul aria-label={`Modules in ${path.at(-1)!.label}`}>
      {graph.data.nodes.slice(0, limit).map((node) => {
        const child = node.child_graph_id;
        const cyclic = child && path.some((crumb) => crumb.id === child);
        return (
          <li key={node.id}>
            <div className="component-tree-row">
              {child && !cyclic ? (
                <button
                  type="button"
                  className="tree-disclosure"
                  aria-label={`${expanded.has(child) ? "Collapse" : "Expand"} ${node.label}`}
                  aria-expanded={expanded.has(child)}
                  onClick={() =>
                    setExpanded((previous) => {
                      const next = new Set(previous);
                      if (next.has(child)) next.delete(child);
                      else next.add(child);
                      return next;
                    })
                  }
                >
                  <ChevronRight size={14} />
                </button>
              ) : (
                <span className="tree-leaf" aria-hidden="true" />
              )}
              <button
                type="button"
                className={`component-select ${activeGraph === graphId && props.selectedId === node.id ? "selected" : ""}`}
                data-component-id={node.id}
                onClick={() => props.navigate(path, node)}
                title={node.module_path ?? node.label}
              >
                <span>
                  <strong>{nodeTitle(node)}</strong>
                  <small>
                    {node.kind.replaceAll("_", " ")}
                    {!child ? " · leaf" : ""}
                  </small>
                </span>
              </button>
            </div>
            {child && expanded.has(child) && !cyclic && (
              <TreeBranch
                {...props}
                path={[...path, { id: child, label: node.label }]}
              />
            )}
          </li>
        );
      })}
      {graph.data.nodes.length > limit && (
        <li>
          <button
            type="button"
            className="button button-ghost"
            onClick={() => setVisible(limit + 80)}
          >
            Show more modules ({graph.data.nodes.length - limit} remaining)
          </button>
        </li>
      )}
    </ul>
  );
}
