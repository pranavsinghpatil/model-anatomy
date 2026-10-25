import type { GraphNode } from "../api/contract";
import { valueText } from "../format";

export function recordedAttributes(node: GraphNode): [string, unknown][] {
  const attributes = node.metadata.attributes;
  if (
    !attributes ||
    typeof attributes !== "object" ||
    Array.isArray(attributes)
  )
    return [];
  return Object.entries(attributes)
    .filter(
      ([key, value]) =>
        key !== "training" &&
        (typeof value === "number" ||
          typeof value === "boolean" ||
          typeof value === "string" ||
          (Array.isArray(value) &&
            value.every(
              (item) => typeof item === "number" || typeof item === "string",
            ))),
    )
    .sort(([left], [right]) => left.localeCompare(right));
}

export function ModuleProperties({ node }: { node: GraphNode }) {
  const attributes = recordedAttributes(node);
  return (
    <section
      className="inspector-section"
      aria-label="Recorded module properties"
    >
      <h3>Module properties</h3>
      <dl className="properties">
        <dt>Native type</dt>
        <dd>
          <code>{node.module_type ?? "Not reported"}</code>
        </dd>
        {attributes.map(([key, value]) => (
          <div className="property-row" key={key}>
            <dt>{key.replaceAll("_", " ")}</dt>
            <dd>
              {Array.isArray(value)
                ? `[${value.join(", ")}]`
                : valueText(value)}
            </dd>
          </div>
        ))}
        <dt>Structure boundary</dt>
        <dd>
          {node.child_graph_id ? "Contains child modules" : "Leaf component"}
        </dd>
      </dl>
      {!attributes.length && (
        <p className="empty-copy">
          No additional module attributes were recorded.
        </p>
      )}
    </section>
  );
}
