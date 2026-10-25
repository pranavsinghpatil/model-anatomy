import type { Page } from "@playwright/test";
import type {
  AnalysisDetail,
  AnalysisSummary,
  ArchitectureGraph,
  ArtifactInspectionResponse,
  Evidence,
  GraphNode,
  Health,
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
    inputs: { input_specs: Record<string, unknown> };
    analysis_report: { warnings: string[] };
  };
  tensors: Record<string, Record<string, unknown>>;
  raw_structure: { modules: Record<string, Record<string, unknown>> };
};
// Self-contained HTTP contract fixtures exercise renderer behavior, including
// legacy repetition and dataflow. They are not evidence that a parser model is
// complete. Parser acceptance uses actual metadata through the real API/worker.
const fixtureEvidence: Evidence[] = [
  {
    source: "contract_fixture",
    kind: "direct",
    locator: "test-only",
    details: {},
  },
];
const port = {
  shape: ["B", "S", 768],
  dtype: "F32",
  semantic_name: "hidden_states",
};
const moduleNode = (
  id: string,
  label: string,
  kind: string,
  path: string | null = null,
  extra: Partial<RawNode> = {},
): RawNode => ({
  id,
  label,
  kind,
  module_path: path,
  module_type: kind,
  repeat: 1,
  inputs: [port],
  outputs: [port],
  parameter_count: 0,
  metadata: {},
  evidence: fixtureEvidence,
  confidence: 1,
  source_modules: path ? [path] : [],
  repetition: null,
  child_graph: null,
  ...extra,
});
const chain = (id: string, nodes: RawNode[]): RawGraph => ({
  id,
  nodes,
  metadata: {},
  edges: nodes.slice(1).map((node, index) => ({
    source: nodes[index].id,
    target: node.id,
    source_port: "output",
    target_port: "input",
    tensor: port,
    label: null,
    relation: "dataflow",
    evidence: fixtureEvidence,
  })),
});
const attention = chain("attention", [
  moduleNode(
    "qkv",
    "QKV Projection",
    "projection",
    "transformer.h.0.attn.c_attn",
  ),
  moduleNode(
    "attention_out",
    "Output Projection",
    "projection",
    "transformer.h.0.attn.c_proj",
  ),
]);
const decoder = chain("decoder", [
  moduleNode(
    "norm1",
    "Attention LayerNorm",
    "layer_norm",
    "transformer.h.0.ln_1",
  ),
  moduleNode(
    "attn",
    "Self Attention",
    "self_attention",
    "transformer.h.0.attn",
    { child_graph: attention },
  ),
  moduleNode("norm2", "MLP LayerNorm", "layer_norm", "transformer.h.0.ln_2"),
  moduleNode("mlp", "MLP", "mlp", "transformer.h.0.mlp"),
]);
const fixtureRoot = chain("root", [
  moduleNode("input", "Token IDs", "input"),
  moduleNode("embedding", "Token Embedding", "embedding", "transformer.wte", {
    parameter_count: 38597376,
    metadata: { attributes: { num_embeddings: 50257, embedding_dim: 768 } },
  }),
  moduleNode("block", "Decoder Block", "transformer_block", "transformer.h", {
    repeat: 12,
    child_graph: decoder,
    repetition: {
      count: 12,
      mode: "sequential",
      instance_paths: Array.from(
        { length: 12 },
        (_, index) => `transformer.h.${index}`,
      ),
    },
    parameter_count: 85000000,
  }),
  moduleNode("norm", "Final LayerNorm", "layer_norm", "transformer.ln_f"),
  moduleNode("head", "LM Head", "lm_head", "lm_head"),
  moduleNode("output", "Output", "output"),
]);
const tensorName = "transformer.wte.weight";
const artifact: Artifact = {
  model_id: "openai-community/gpt2",
  architecture: "GPT2LMHeadModel",
  root_graph: fixtureRoot,
  metadata: {
    revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    unique_parameter_count: 124439808,
    dimensions: { hidden: 768, heads: 12, layers: 12, vocabulary: 50257 },
    inputs: { input_specs: { input_ids: {} } },
    analysis_report: { warnings: [] },
  },
  tensors: {
    [tensorName]: {
      name: tensorName,
      shape: [50257, 768],
      dtype: "F32",
      parameter_count: 38597376,
      artifact:
        "https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors",
      parameter_refs: [tensorName],
    },
  },
  raw_structure: {
    modules: {
      "transformer.wte": {
        type: "Embedding",
        attributes: { num_embeddings: 50257, embedding_dim: 768 },
      },
    },
  },
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
export const graphs = graphMap(artifact.root_graph);
export const rootGraph = graphs.get(artifact.root_graph.id)!;
export const block = rootGraph.nodes.find((node) => node.repeat > 1)!;
export const embedding = rootGraph.nodes.find(
  (node) => node.kind === "embedding",
)!;
export function analysisFor(id = "job-test"): AnalysisDetail {
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
    quality: {
      mode: "static_module_architecture",
      architecture_scope: "Native registered module architecture",
      architecture_checks: {
        module_inventory: true,
        parameter_mapping: true,
        native_architecture: true,
        module_topology: true,
      },
      checkpoint_parameter_coverage: 1,
      unknown_nodes: 0,
      unmapped_checkpoint_tensors: 0,
      forward_executed: false,
      runtime_trace_available: false,
    },
    dimensions: artifact.metadata.dimensions,
    input_names: Object.keys(artifact.metadata.inputs.input_specs),
  };
}
export type Scenario = {
  job?: AnalysisDetail;
  ai?: boolean;
  health?: Health;
  healthError?: "network" | "invalid";
  noRecent?: boolean;
  recent?: AnalysisSummary[];
  postStatus?: number;
  postError?: string;
  questionError?: boolean;
  metadataCitation?: boolean;
  getStatus?: number;
  graphError?: boolean;
  pollComplete?: boolean;
  pollResult?: "completed" | "partial" | "inspected";
  symbolic?: boolean;
  paginated?: boolean;
  artifactPages?: ArtifactInspectionResponse[];
};
export async function mockService(page: Page, scenario: Scenario = {}) {
  const requests: { method: string; path: string; body: unknown }[] = [];
  let job = scenario.job ?? analysisFor();
  let polls = 0;
  const apiRoute = (url: URL) =>
    url.pathname === "/health" || url.pathname.startsWith("/api/v1/");
  await page.route(apiRoute, async (route) => {
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
    if (url.pathname === "/health") {
      if (scenario.healthError === "network") return route.abort("failed");
      if (scenario.healthError === "invalid")
        return error(503, "unavailable", "Invalid health response");
      const health: Health = scenario.health ?? {
        status: "ok",
        api_version: "v1",
        checks: { database: true, storage: true, worker: true },
        analysis_available: true,
        ai_available: scenario.ai ?? true,
        storage_backend: "local",
        reason: null,
      };
      return json(health, health.status === "ok" ? 200 : 503);
    }
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
      return json({
        analyses: scenario.noRecent ? [] : (scenario.recent ?? [job]),
      });
    }
    const id = parts[1];
    if (parts.length === 2) {
      if (id !== "job-test")
        return error(404, "not_found", "Analysis not found");
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
        if (scenario.pollComplete && ++polls > 1) {
          const status = scenario.pollResult ?? "completed";
          job = {
            ...job,
            status,
            stage: status,
            root_graph_id: status === "inspected" ? null : rootGraph.id,
            ...(status === "partial"
              ? {
                  partial_reasons: [
                    "Checkpoint parameter mapping is incomplete.",
                  ],
                }
              : {}),
            ...(status === "inspected"
              ? {
                  architecture: null,
                  parameter_count: null,
                  inspection: {
                    kind: "artifact_inspection" as const,
                    reason_code: "architecture_not_supported",
                    message:
                      "Artifact metadata is available; no architecture graph was produced.",
                    file_count: 1,
                    tensor_count: 0,
                    checkpoint_tensor_element_count: null,
                    checkpoint_available: false,
                  },
                }
              : {}),
          };
        }
        return json(job);
      }
      return error(404, "not_found", "Analysis not found");
    }
    if (id !== "job-test") return error(404, "not_found", "Analysis not found");
    const collection = graphMap(artifact.root_graph);
    if (parts[2] === "graph-index") {
      const locations: {
        id: string;
        label: string;
        parent_graph_id: string | null;
        parent_node_id: string | null;
      }[] = [];
      function visit(
        graph: RawGraph,
        label: string,
        parentGraph: string | null,
        parentNode: string | null,
      ) {
        locations.push({
          id: graph.id,
          label,
          parent_graph_id: parentGraph,
          parent_node_id: parentNode,
        });
        for (const node of graph.nodes)
          if (node.child_graph)
            visit(node.child_graph, node.label, graph.id, node.id);
      }
      visit(artifact.root_graph, "gpt2", null, null);
      return json({
        root_graph_id: rootGraph.id,
        graphs: locations,
        nodes: [...collection.values()].flatMap((graph) =>
          graph.nodes.map((node) => ({
            id: node.id,
            graph_id: graph.id,
            label: node.label,
            kind: node.kind,
            module_path: node.module_path,
            child_graph_id: node.child_graph_id,
          })),
        ),
        citations: [...collection.values()].flatMap((graph) =>
          graph.nodes.flatMap((node) =>
            node.evidence.map((item) => ({
              ...item,
              graph_id: graph.id,
              node_id: node.id,
            })),
          ),
        ),
      });
    }
    if (parts[2] === "artifacts") {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const page = scenario.artifactPages?.find(
        (item) => item.offset === offset,
      );
      if (!page)
        return error(404, "not_found", "Artifact page fixture not found");
      return json(page);
    }
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
        warnings: [
          "AI interpretation is not independently verified. Citations and numeric values are checked; confirm other claims against the cited evidence.",
        ],
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
