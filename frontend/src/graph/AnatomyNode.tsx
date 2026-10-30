import { Handle, Position, type NodeProps } from "@xyflow/react";
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  ChevronRight,
  Layers3,
  ScanLine,
  SlidersHorizontal,
  Table2,
  Waypoints,
  Braces,
  MoveRight,
  CircleDot,
} from "lucide-react";
import type { AnatomyNode as NodeType, GraphPort } from "./layout";
import { count, shape } from "../format";
import "./anatomy-node.css";
import { nodeCategory, nodeSize, nodeTitle } from "./presentation";

function portStyle(port: GraphPort, direction: "DOWN" | "RIGHT") {
  return direction === "RIGHT"
    ? { top: `${port.position}%` }
    : { left: `${port.position}%` };
}

const categoryIcons = {
  boundary: CircleDot,
  attention: Waypoints,
  position: ScanLine,
  normalization: SlidersHorizontal,
  projection: MoveRight,
  structure: Layers3,
  embedding: Table2,
  other: Braces,
};

export function ModelNode({ data, selected }: NodeProps<NodeType>) {
  const node = data.original;
  const category = nodeCategory(node.kind);
  const boundary = category === "boundary";
  const detailed = data.density === "detailed";
  const Icon =
    node.kind === "input"
      ? ArrowDownToLine
      : node.kind === "output"
        ? ArrowUpFromLine
        : categoryIcons[category];
  const input = node.inputs[0];
  const output = node.outputs[0];
  const size = nodeSize(node, data.density);
  const kind = node.kind.replaceAll("_", " ");
  return (
    <div
      className={`model-node category-${category} density-${data.density} ${node.child_graph_id ? "has-child" : ""} ${node.repeat > 1 ? "is-repeated" : ""} ${selected ? "is-selected" : ""} ${data.dimmed ? "is-dimmed" : ""}`}
      data-node-id={node.id}
      data-kind={node.kind}
      style={{ width: size.width, height: size.height }}
    >
      {data.targetPorts.map((port) => (
        <Handle
          key={port.id}
          id={port.id}
          type="target"
          position={data.direction === "RIGHT" ? Position.Left : Position.Top}
          className="graph-port"
          style={portStyle(port, data.direction)}
          title={`Reported input port: ${port.label}`}
        />
      ))}
      <div className="anatomy-node-body">
        <div className="anatomy-node-icon" aria-hidden="true">
          <Icon size={boundary ? 19 : 21} strokeWidth={1.7} />
        </div>
        <div className="anatomy-node-heading">
          <div className="node-header-row">
            <div className="node-kind">
              <span>{kind}</span>
              {node.repeat > 1 && (
                <span
                  className="repeat"
                  title={`${count(node.repeat)} instances · ${node.repetition?.mode ?? "repetition mode unknown"}`}
                >
                  ×{count(node.repeat)}
                </span>
              )}
            </div>
            {node.parameter_count > 0 && (
              <span
                className="node-mini-param"
                title={`${count(node.parameter_count)} parameters`}
              >
                {count(node.parameter_count, true)}
              </span>
            )}
            {boundary &&
              (node.kind === "input" ? output : input)?.shape != null && (
                <span className="node-mini-shape">
                  {shape(
                    (node.kind === "input" ? output : input)?.shape ?? null,
                  )}
                </span>
              )}
          </div>
          <div
            className="node-title"
            title={`${node.label}${node.module_path ? ` · ${node.module_path}` : ""}`}
          >
            {nodeTitle(node)}
          </div>
        </div>
      </div>
      {detailed && (
        <div className="anatomy-node-facts">
          {!boundary && (input?.shape != null || output?.shape != null) && (
            <div
              className="node-ports"
              aria-label="First reported input and output shapes"
            >
              <div
                title={`${node.inputs.length} input specification(s); first shown`}
              >
                <span>
                  IN{node.inputs.length > 1 ? ` · ${node.inputs.length}` : ""}
                </span>
                <code>{input ? shape(input.shape) : "Unknown"}</code>
              </div>
              <MoveRight size={14} aria-hidden="true" />
              <div
                title={`${node.outputs.length} output specification(s); first shown`}
              >
                <span>
                  OUT
                  {node.outputs.length > 1 ? ` · ${node.outputs.length}` : ""}
                </span>
                <code>{output ? shape(output.shape) : "Unknown"}</code>
              </div>
            </div>
          )}
          {boundary ? (
            (node.kind === "input" ? output : input)?.shape != null && (
              <code className="boundary-shape">
                {shape((node.kind === "input" ? output : input)?.shape ?? null)}
              </code>
            )
          ) : (
            <div className="node-footer">
              <span>
                <strong title={count(node.parameter_count)}>
                  {count(node.parameter_count, true)}
                </strong>{" "}
                parameters
              </span>
              <code title={node.module_type ?? "Module type unknown"}>
                {node.module_type ?? "Unknown type"}
              </code>
            </div>
          )}
        </div>
      )}
      {node.child_graph_id && (
        <button
          className="node-open nodrag nopan"
          aria-label={`Open ${node.label}`}
          title={`Explore the nested graph of ${node.label}`}
          onClick={(event) => {
            event.stopPropagation();
            data.openGraph(node);
          }}
        >
          <span>
            {node.repeat > 1
              ? "Explore representative block"
              : "Explore inside"}
          </span>
          <ChevronRight size={17} />
        </button>
      )}
      {data.sourcePorts.map((port) => (
        <Handle
          key={port.id}
          id={port.id}
          type="source"
          position={
            data.direction === "RIGHT" ? Position.Right : Position.Bottom
          }
          className="graph-port"
          style={portStyle(port, data.direction)}
          title={`Reported output port: ${port.label}`}
        />
      ))}
    </div>
  );
}
