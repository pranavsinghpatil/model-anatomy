import { useMemo, useRef, useState } from "react";
import type { GraphNode } from "../api/contract";
import {
  assemblyFaces,
  assemblyFrame,
  faceDepth,
  facePath,
  faceVisible,
  partLabel,
  project,
  type Assembly,
  type AnatomyView,
} from "./model";
import { useAssemblyMotion } from "./useAssemblyMotion";

export function AssemblyDrawing({
  assembly,
  view,
  selected,
  select,
  focus,
  reassemble,
}: {
  assembly: Assembly;
  view: AnatomyView;
  selected: GraphNode | null;
  select: (node: GraphNode) => void;
  focus: (node: GraphNode) => void;
  reassemble: () => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<string | null>(null);
  const { pose, still, visible } = useAssemblyMotion(
    view.separation,
    container,
  );
  const frame = useMemo(() => assemblyFrame(assembly), [assembly]);
  const faces = useMemo(
    () =>
      assemblyFaces(assembly, pose)
        .filter((face) => faceVisible(face, view.angle))
        .sort((a, b) => {
          // A large mounting plane's centroid can sit in front of distant
          // components even though the entire plane is below them. Paint the
          // supporting bench first, then depth-sort the component surfaces.
          if (Boolean(a.nodeId) !== Boolean(b.nodeId)) return a.nodeId ? 1 : -1;
          return faceDepth(a, view.angle) - faceDepth(b, view.angle);
        }),
    [assembly, pose, view.angle],
  );
  const selectedStation = assembly.stations.find(
    (station) => station.node.id === selected?.id,
  );
  const center =
    view.zoom > 1.05 && selectedStation
      ? project(
          [selectedStation.origin[0], selectedStation.origin[1], 30],
          view.angle,
        )
      : [frame.x + frame.width / 2, frame.y + frame.height / 2];
  const width = frame.width / view.zoom;
  const height = frame.height / view.zoom;
  const hovered = assembly.stations.find(
    (station) => station.node.id === hover,
  )?.node;
  const benchFaces = faces.filter((face) => !face.nodeId);
  const partFaces = faces.filter((face) => face.nodeId);
  const renderFace = (face: (typeof faces)[number]) => (
    <path
      key={face.id}
      d={facePath(face, view.angle)}
      className={`assembly-face face-${face.tone} ${face.nodeId === selected?.id ? "is-selected" : ""} ${face.nodeId === hover ? "is-hovered" : ""}`}
      data-part-id={face.nodeId}
      data-solid-id={face.id}
      aria-hidden="true"
      onPointerEnter={() => face.nodeId && setHover(face.nodeId)}
      onClick={() => {
        const station = assembly.stations.find(
          (item) => item.node.id === face.nodeId,
        );
        if (station) select(station.node);
      }}
    >
      <title>
        {face.nodeId
          ? partLabel(
              assembly.stations.find(
                (station) => station.node.id === face.nodeId,
              )!.node,
            )
          : "Illustrative mounting bench"}
      </title>
    </path>
  );
  return (
    <div
      className="assembly-drawing"
      ref={container}
      data-testid="anatomy-drawing"
      data-pose={pose.toFixed(3)}
      data-motion={still ? "reduced" : visible ? "visible" : "paused"}
    >
      <svg
        viewBox={`${center[0] - width / 2} ${center[1] - height / 2} ${width} ${height}`}
        role="group"
        aria-label="Interactive model assembly"
        aria-describedby="assembly-keyboard-help"
        tabIndex={0}
        onPointerLeave={() => setHover(null)}
        onKeyDown={(event) => {
          const index = assembly.stations.findIndex(
            (station) => station.node.id === selected?.id,
          );
          if (
            [
              "ArrowLeft",
              "ArrowRight",
              "ArrowUp",
              "ArrowDown",
              "Home",
              "End",
            ].includes(event.key)
          ) {
            event.preventDefault();
            const next =
              event.key === "Home"
                ? 0
                : event.key === "End"
                  ? assembly.stations.length - 1
                  : (index +
                      (["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1) +
                      assembly.stations.length) %
                    assembly.stations.length;
            if (assembly.stations[next]) focus(assembly.stations[next].node);
          } else if ((event.key === "Enter" || event.key === " ") && selected) {
            event.preventDefault();
            select(selected);
          } else if (event.key === "Escape") reassemble();
        }}
      >
        <title>Inspected model components as an illustrative assembly</title>
        {benchFaces.map(renderFace)}
        <g aria-hidden="true" className="assembly-connections">
          {assembly.connections.map((edge) => {
            const source = assembly.stations.find(
              (station) => station.node.id === edge.source,
            )!;
            const target = assembly.stations.find(
              (station) => station.node.id === edge.target,
            )!;
            const start: [number, number, number] = [
              source.origin[0] + 32,
              source.origin[1],
              3,
            ];
            const end: [number, number, number] = [
              target.origin[0] - 32,
              target.origin[1],
              3,
            ];
            const middle = (start[0] + end[0]) / 2;
            const points = [
              start,
              [middle, start[1], 3],
              [middle, end[1], 3],
              end,
            ] as [number, number, number][];
            return (
              <path
                key={edge.id}
                data-reported-edge={edge.id}
                data-relation={edge.relation}
                strokeDasharray={
                  edge.relation === "conditional_dependency"
                    ? "3 5"
                    : edge.relation === "dependency"
                      ? "7 4"
                      : undefined
                }
                className={
                  edge.source === selected?.id || edge.target === selected?.id
                    ? "is-selected"
                    : ""
                }
                d={points
                  .map(
                    (point, i) =>
                      `${i ? "L" : "M"}${project(point, view.angle).join(",")}`,
                  )
                  .join(" ")}
              >
                <title>{edge.relation.replaceAll("_", " ")}</title>
              </path>
            );
          })}
        </g>
        {partFaces.map(renderFace)}
      </svg>
      <p id="assembly-keyboard-help" className="sr-only">
        Arrow keys choose a reported component. Enter opens its inspector.
        Escape reassembles the illustration. The Components list is an
        alternative to this scene.
      </p>
      <div className="assembly-hover-label" aria-hidden="true">
        {hovered
          ? partLabel(hovered)
          : "Select a part to inspect its recorded facts"}
      </div>
    </div>
  );
}
