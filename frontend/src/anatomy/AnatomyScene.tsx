import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Expand,
  Layers3,
  RotateCcw,
  RotateCw,
  ScanEye,
  ZoomIn,
  ZoomOut,
  FileCheck2,
  Boxes,
} from "lucide-react";
import type {
  AnalysisDetail,
  ArchitectureGraph,
  GraphNode,
} from "../api/contract";
import { count, shape } from "../format";
import { nodeTitle } from "../graph/presentation";
import { componentMeaning } from "../components/component-purpose";
import { IconButton } from "../components/Common";
import { Button } from "../components/ui";
import { AssemblyDrawing } from "./AssemblyDrawing";
import { buildAssembly, defaultAnatomyView, type AnatomyView } from "./model";
import "./anatomy.css";

export function AnatomyScene({
  analysis,
  graph,
  selected,
  view,
  onView,
  select,
  focus,
  open,
  evidence,
  technical,
}: {
  analysis: AnalysisDetail;
  graph: ArchitectureGraph;
  selected: GraphNode | null;
  view: AnatomyView;
  onView: (view: AnatomyView) => void;
  select: (node: GraphNode) => void;
  focus: (node: GraphNode) => void;
  open: (node: GraphNode) => void;
  evidence: (node: GraphNode) => void;
  technical: () => void;
}) {
  const [page, setPage] = useState(0);
  const pages = Math.max(1, Math.ceil(graph.nodes.length / 24));
  const currentPage = Math.min(page, pages - 1);
  useEffect(() => {
    setPage(0);
  }, [graph.id]);
  useEffect(() => {
    const index = graph.nodes.findIndex((node) => node.id === selected?.id);
    if (index >= 0) setPage(Math.floor(index / 24));
  }, [graph.id, selected?.id]);
  function turnPage(next: number) {
    setPage(next);
    const node = graph.nodes[next * 24];
    if (node) focus(node);
  }
  const assembly = useMemo(
    () => buildAssembly(graph, selected?.id, currentPage),
    [graph, selected?.id, currentPage],
  );
  const patch = (update: Partial<AnatomyView>) =>
    onView({ ...view, ...update });
  const current = selected ?? assembly.stations[0]?.node;
  const index = assembly.stations.findIndex(
    (station) => station.node.id === current?.id,
  );
  const move = (offset: number) => {
    const station =
      assembly.stations[
        (index + offset + assembly.stations.length) % assembly.stations.length
      ];
    if (station) focus(station.node);
  };
  if (!assembly.stations.length)
    return (
      <div className="canvas-state">
        <strong>No components to illustrate</strong>
        <p>The current graph contains no reported components.</p>
        <Button onClick={technical}>Use technical graph</Button>
      </div>
    );
  return (
    <section
      className="anatomy-scene"
      data-testid="anatomy-scene"
      aria-label="Model anatomy illustration"
    >
      <div className="anatomy-toolbar" aria-label="Illustration controls">
        <Button
          aria-pressed={view.separation > 0.5}
          onClick={() => patch({ separation: view.separation > 0.5 ? 0 : 1 })}
        >
          <Layers3 size={16} />
          {view.separation > 0.5 ? "Reassemble" : "Separate parts"}
        </Button>
        <label className="anatomy-separation">
          <span>Separation</span>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={Math.round(view.separation * 100)}
            onChange={(event) =>
              patch({ separation: Number(event.target.value) / 100 })
            }
          />
          <output>{Math.round(view.separation * 100)}%</output>
        </label>
        <div className="anatomy-camera-controls">
          <IconButton
            label="Rotate assembly left"
            disabled={view.angle <= 30}
            onClick={() => patch({ angle: Math.max(30, view.angle - 6) })}
          >
            <RotateCcw size={16} />
          </IconButton>
          <IconButton
            label="Rotate assembly right"
            disabled={view.angle >= 60}
            onClick={() => patch({ angle: Math.min(60, view.angle + 6) })}
          >
            <RotateCw size={16} />
          </IconButton>
          <IconButton
            label="Zoom illustration out"
            disabled={view.zoom <= 0.6}
            onClick={() => patch({ zoom: Math.max(0.6, view.zoom - 0.2) })}
          >
            <ZoomOut size={16} />
          </IconButton>
          <IconButton
            label="Zoom illustration in"
            disabled={view.zoom >= 2.4}
            onClick={() => patch({ zoom: Math.min(2.4, view.zoom + 0.2) })}
          >
            <ZoomIn size={16} />
          </IconButton>
          <IconButton label="Fit assembly" onClick={() => patch({ zoom: 1 })}>
            <Expand size={16} />
          </IconButton>
          <IconButton
            label="Reset illustration view"
            onClick={() => onView(defaultAnatomyView)}
          >
            <ScanEye size={16} />
          </IconButton>
        </div>
      </div>
      <div className="anatomy-scene-caption">
        <span>
          <Boxes size={14} />
          {assembly.hasConnections ? "Model assembly" : "Component inventory"}
        </span>
        <span>
          {assembly.stations.length} of {graph.nodes.length} components ·{" "}
          {assembly.connections.length} illustrated links · {graph.edges.length}{" "}
          reported relationships
        </span>
      </div>
      {pages > 1 && (
        <nav
          className="anatomy-pagination"
          aria-label="Illustration component pages"
        >
          <Button
            disabled={currentPage === 0}
            onClick={() => turnPage(currentPage - 1)}
          >
            Previous components
          </Button>
          <span aria-live="polite">
            Page {currentPage + 1} of {pages} · components{" "}
            {currentPage * 24 + 1}–
            {Math.min((currentPage + 1) * 24, graph.nodes.length)}
          </span>
          <Button
            disabled={currentPage + 1 >= pages}
            onClick={() => turnPage(currentPage + 1)}
          >
            Next components
          </Button>
        </nav>
      )}
      <AssemblyDrawing
        assembly={assembly}
        view={view}
        selected={current ?? null}
        select={select}
        focus={focus}
        reassemble={() => patch({ separation: 0 })}
      />
      {current && (
        <div className="anatomy-readout" aria-live="polite" aria-atomic="true">
          <div className="anatomy-part-navigation">
            <IconButton
              label="Previous illustrated component"
              onClick={() => move(-1)}
            >
              <ArrowLeft size={16} />
            </IconButton>
            <span>
              {index + 1} / {assembly.stations.length}
            </span>
            <IconButton
              label="Next illustrated component"
              onClick={() => move(1)}
            >
              <ArrowRight size={16} />
            </IconButton>
          </div>
          <div className="anatomy-part-summary">
            <div>
              <strong>{nodeTitle(current)}</strong>
              <span>{current.kind.replaceAll("_", " ")}</span>
              {current.repeat > 1 && (
                <span className="anatomy-repeat">
                  ×{count(current.repeat)} instances
                </span>
              )}
            </div>
            <p>{componentMeaning(current.kind)}</p>
            <dl>
              <div>
                <dt>Parameters</dt>
                <dd
                  title={
                    analysis.quality?.mode === "publisher_reference_topology"
                      ? "Unverified"
                      : count(current.parameter_count)
                  }
                >
                  {analysis.quality?.mode === "publisher_reference_topology"
                    ? "Unknown"
                    : count(current.parameter_count, true)}
                </dd>
              </div>
              {current.outputs[0]?.shape != null && (
                <div>
                  <dt>First output shape</dt>
                  <dd>{shape(current.outputs[0]?.shape ?? null)}</dd>
                </div>
              )}
            </dl>
          </div>
          <div className="anatomy-part-actions">
            <Button onClick={() => select(current)}>
              Inspect component <ArrowRight size={14} />
            </Button>
            {current.child_graph_id && (
              <Button onClick={() => open(current)}>
                Explore inside <Layers3 size={14} />
              </Button>
            )}
            <Button variant="ghost" onClick={() => evidence(current)}>
              <FileCheck2 size={15} /> Evidence
            </Button>
          </div>
        </div>
      )}
      <div className="anatomy-scope">
        <span>
          Illustrative geometry, not model hardware or a runtime simulation.
          Links retain the reported dependency type, not runtime tensor flow.
          Each page displays up to 24 actual components.
        </span>
        <Button variant="ghost" onClick={technical}>
          Technical graph <ArrowRight size={13} />
        </Button>
      </div>
      {assembly.stations.some((station) => station.disconnected) && (
        <p className="anatomy-boundary">
          Components without an acyclic dependency layout sit separately in the
          assembly. Their position does not establish execution order.
        </p>
      )}
      {assembly.omitted > 0 && (
        <p className="anatomy-boundary">
          {assembly.omitted} components are outside this illustration page. Use
          component pages, the Components tree, or the technical graph to
          inspect them.
        </p>
      )}
      {analysis.status === "partial" && (
        <p className="anatomy-boundary">
          Some architecture checks remain incomplete. Open the Report view for
          the recorded reasons. Activation shapes are outside these checks.
        </p>
      )}
    </section>
  );
}
