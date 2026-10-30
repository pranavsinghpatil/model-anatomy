import {
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
  type EdgeProps,
} from "@xyflow/react";
import type { AnatomyEdge } from "./layout";
import { edgeLabel } from "./presentation";

function pathFromPoints(points: { x: number; y: number }[]) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${point.x},${point.y}`)
    .join(" ");
}

export function RoutedEdge(props: EdgeProps<AnatomyEdge>) {
  const [fallback] = getSmoothStepPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
  });
  const route = props.data?.route;
  const labelPosition = props.data?.labelPosition;
  const title = props.data?.original
    ? edgeLabel(props.data.original, true)
    : String(props.label ?? "");
  return (
    <>
      <BaseEdge
        id={props.id}
        path={route && route.length > 1 ? pathFromPoints(route) : fallback}
        markerEnd={props.markerEnd}
        style={props.style}
        interactionWidth={14}
      />
      {props.label && (
        <EdgeLabelRenderer>
          <div
            className={`graph-edge-label nodrag nopan ${props.selected ? "is-selected" : ""}`}
            title={title}
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelPosition?.x ?? (props.sourceX + props.targetX) / 2}px,${labelPosition?.y ?? (props.sourceY + props.targetY) / 2}px)`,
              opacity: props.style?.opacity,
            }}
          >
            {props.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
