import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { Page } from "@playwright/test";
import type {
  AnalysisDetail,
  ArchitectureGraph,
  Evidence,
  GraphNode,
} from "../src/api/contract";

type RawNode = Omit<GraphNode, "child_graph_id"> & {
  child_graph: RawGraph | null;
};
type RawGraph = Omit<ArchitectureGraph, "nodes"> & { nodes: RawNode[] };
type Artifact = {
  model_id: string;
  architecture: string;
  root_graph: RawGraph;
  metadata: {
    revision: string;
    unique_parameter_count: number;
    dimensions: AnalysisDetail["dimensions"];
    analysis_report: { warnings: string[] };
  };
  tensors: Record<string, Record<string, unknown>>;
  raw_structure: { modules: Record<string, Record<string, unknown>> };
};
// Tests project real engine artifacts into the agreed shallow HTTP contract.
// No fixture is included in the production app or used as an API fallback.
const load = (directory: string) =>
  JSON.parse(
    readFileSync(
      fileURLToPath(
        new URL(`../../output/${directory}/model.json`, import.meta.url),
      ),
      "utf8",
    ),
  ) as Artifact;
const artifacts = {
  "example-gpt2": load("gpt2"),
  "example-smollm": load("huggingfacetb--smollm2-135m"),
};
const graphMap = (root: RawGraph): Map<string, ArchitectureGraph> => {
  const result = new Map<string, ArchitectureGraph>();
  function visit(graph: RawGraph) {
    result.set(graph.id, {
      ...graph,
      nodes: graph.nodes.map(({ child_graph, ...node }) => ({
        ...node,
        child_graph_id: child_graph?.id ?? null,
      })),
    });
    for (const node of graph.nodes)
      if (node.child_graph) visit(node.child_graph);
  }
  visit(root);
  return result;
};
export const graphs = graphMap(artifacts["example-gpt2"].root_graph);
export const rootGraph = graphs.get(artifacts["example-gpt2"].root_graph.id)!;
export const block = rootGraph.nodes.find((node) => node.repeat > 1)!;
export const embedding = rootGraph.nodes.find(
  (node) => node.kind === "embedding",
)!;
export function analysisFor(id = "example-gpt2"): AnalysisDetail {
  const artifact =
    artifacts[id as keyof typeof artifacts] ?? artifacts["example-gpt2"];
  return {
    id,
    model_id: artifact.model_id,
    architecture: artifact.architecture,
    revision: artifact.metadata.revision,
    status: "completed",
    stage: "completed",
    error: null,
    warnings: artifact.metadata.analysis_report.warnings,
    created_at: "2026-10-08T09:00:00Z",
    expires_at: null,
    parameter_count: artifact.metadata.unique_parameter_count,
    root_graph_id: artifact.root_graph.id,
    cached: false,
    prepared: id.startsWith("example-"),
    dimensions: artifact.metadata.dimensions,
  };
}
export type Scenario = {
  job?: AnalysisDetail;
  ai?: boolean;
  reconstruct?: boolean;
  postStatus?: number;
  postError?: string;
  questionError?: boolean;
  metadataCitation?: boolean;
  getStatus?: number;
  graphError?: boolean;
  pollComplete?: boolean;
  empty?: boolean;
  symbolic?: boolean;
  paginated?: boolean;
};
export async function mockService(page: Page, scenario: Scenario = {}) {
  const requests: { method: string; path: string; body: unknown }[] = [];
  let job = scenario.job ?? analysisFor("job-test");
  let polls = 0;
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const parts = url.pathname.split("/").slice(3).map(decodeURIComponent);
    requests.push({
      method: request.method(),
      path: url.pathname + url.search,
      body: request.postDataJSON(),
    });
    const json = (body: unknown, status = 200) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    const error = (status: number, code: string, message: string) =>
      json({ error: { code, message, request_id: "test-request" } }, status);
    if (parts[0] === "capabilities")
      return json({
        api_version: "v1",
        reconstruction_available: scenario.reconstruct ?? true,
        ai_available: scenario.ai ?? true,
        storage_backend: "local",
        supported_models: ["gpt2", "llama"],
      });
    if (parts[0] === "examples")
      return json({
        examples: scenario.empty
          ? []
          : Object.keys(artifacts).map((id) => ({
              ...analysisFor(id),
              title: id === "example-gpt2" ? "GPT-2" : "SmolLM2 135M",
            })),
      });
    if (parts[0] !== "analyses") return error(404, "not_found", "Not found");
    if (parts.length === 1) {
      if (request.method() === "POST")
        return scenario.postStatus
          ? error(
              scenario.postStatus,
              scenario.postError ?? "unavailable",
              "Analysis could not be started",
            )
          : json(job, 202);
      return json({ analyses: scenario.empty ? [] : [job] });
    }
    const id = parts[1];
    if (parts.length === 2) {
      if (request.method() === "DELETE") {
        job = { ...job, status: "cancelled", stage: "cancelled" };
        return json(job);
      }
      if (scenario.getStatus && id === "job-test")
        return error(
          scenario.getStatus,
          scenario.getStatus === 410 ? "expired" : "not_found",
          scenario.getStatus === 410
            ? "Analysis has expired"
            : "Analysis not found",
        );
      if (id === "job-test") {
        if (scenario.pollComplete && ++polls > 1)
          job = {
            ...job,
            status: "completed",
            stage: "completed",
            root_graph_id: rootGraph.id,
          };
        return json(job);
      }
      return json(analysisFor(id));
    }
    const artifact =
      artifacts[id as keyof typeof artifacts] ?? artifacts["example-gpt2"];
    const collection = graphMap(artifact.root_graph);
    if (parts[2] === "graphs") {
      if (scenario.graphError)
        return error(503, "graph_unavailable", "Stored graph unavailable");
      const graph = collection.get(parts[3]);
      if (!graph) return error(404, "not_found", "Graph not found");
      if (scenario.symbolic)
        return json({
          ...graph,
          nodes: graph.nodes.map((node) => ({
            ...node,
            inputs: node.inputs.map((tensor) => ({
              ...tensor,
              shape: ["batch", "sequence", "hidden"],
              dtype: null,
            })),
          })),
        });
      return json(graph);
    }
    if (parts[2] === "nodes") {
      const graph = collection.get(parts[3]);
      const node = graph?.nodes.find((node) => node.id === parts[4]);
      if (!node) return error(404, "not_found", "Node not found");
      const prefix = node.module_path;
      let tensors = Object.values(artifact.tensors).filter(
        (tensor) =>
          tensor.name === node.metadata.tensor_name ||
          ((tensor.parameter_refs as string[] | undefined) ?? []).some(
            (path) =>
              prefix != null &&
              (path === prefix || path.startsWith(prefix + ".")),
          ),
      );
      if (scenario.paginated)
        tensors = Array.from({ length: 51 }, (_, index) => ({
          name: `test.pagination.${index}`,
          dtype: "F32",
          shape: ["batch", 768],
          parameter_count: 768,
          artifact: "test.safetensors",
        }));
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const responseNode = scenario.symbolic
        ? {
            ...node,
            inputs: node.inputs.map((tensor) => ({
              ...tensor,
              shape: ["batch", "sequence", "hidden"],
              dtype: null,
            })),
          }
        : node;
      return json({
        graph_id: graph!.id,
        node: responseNode,
        tensors: tensors.slice(offset, offset + 50),
        tensor_count: tensors.length,
        evidence: node.evidence as Evidence[],
        module: artifact.raw_structure.modules[prefix ?? ""] ?? null,
        warnings: [],
      });
    }
    if (parts[2] === "questions") {
      if (scenario.questionError)
        return error(503, "ai_unavailable", "Explanation provider unavailable");
      const body = request.postDataJSON() as {
        graph_id?: string;
        node_id?: string;
      };
      return json({
        answer: "The component is identified by the supplied module evidence.",
        citations: [
          {
            id: "e1",
            source: "stored_ir",
            kind: "direct",
            locator: scenario.metadataCitation
              ? "metadata"
              : `node:${body.graph_id ?? rootGraph.id}/${body.node_id ?? embedding.id}`,
          },
        ],
        model: "test-evidence-provider",
        ai_generated: true,
        warnings: [],
      });
    }
    if (parts[2] === "exports")
      return route.fulfill({
        contentType:
          parts[3] === "zip" ? "application/zip" : "application/json",
        body:
          parts[3] === "zip"
            ? Buffer.from("PK\u0003\u0004test")
            : JSON.stringify(artifact),
        headers: {
          "Content-Disposition": `attachment; filename="model.${parts[3]}"`,
        },
      });
    return error(404, "not_found", "Not found");
  });
  return requests;
}
