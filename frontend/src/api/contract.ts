import type { components } from "./generated";

export type InspectionSummary = components["schemas"]["InspectionSummary"];
export type ArtifactInspectionResponse =
  components["schemas"]["ArtifactInspectionResponse"];
export type AnalysisDetail = components["schemas"]["AnalysisDetail"];
export type AnalysisSummary = components["schemas"]["AnalysisSummary"];
export type AnalysisStatus = components["schemas"]["AnalysisDetail"]["status"];
export type Evidence = components["schemas"]["Evidence"];
export type TensorSpec = components["schemas"]["TensorSpec"];
export type Dimension = NonNullable<TensorSpec["shape"]>[number];
export type GraphNode = components["schemas"]["GraphNode"];
export type GraphEdge = components["schemas"]["GraphEdge"];
export type ArchitectureGraph = components["schemas"]["GraphResponse"];
export type TensorMetadata = components["schemas"]["TensorRecord"];
export type NodeDetail = components["schemas"]["NodeInspection"];
export type Citation = components["schemas"]["Citation"];
export type QuestionAnswer = components["schemas"]["QuestionAnswer"];
export type Health = components["schemas"]["Health"];
export type AnalysesResponse = components["schemas"]["AnalysesResponse"];
export type CreateAnalysisRequest = components["schemas"]["CreateAnalysis"];
export type CatalogItem = components["schemas"]["CatalogItem"];
export type CatalogResponse = components["schemas"]["CatalogResponse"];
export type QuestionRequest = components["schemas"]["Question"];
export type ExportFormat =
  "json" | "mermaid" | "markdown" | "diagnostics" | "audit" | "zip";
