"""Public HTTP schemas; reconstruction IR remains owned by the engine."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Readiness(BaseModel):
    status: Literal["ready", "unavailable"] = "unavailable"
    reason: str | None = None


class Capabilities(BaseModel):
    api_version: Literal["v1"] = "v1"
    reconstruction_available: bool = False
    ai_available: bool = False
    storage_backend: Literal["local", "supabase"] = "local"
    supported_models: list[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class CreateAnalysis(BaseModel):
    model: str = Field(min_length=1, max_length=512)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    graph_id: str | None = Field(default=None, max_length=128)
    node_id: str | None = Field(default=None, max_length=128)


AnalysisStatus = Literal[
    "queued", "running", "completed", "partial", "unsupported", "failed", "timed_out", "cancelled"
]
EvidenceKind = Literal["direct", "derived", "reconstructed", "unknown"]


class AnalysisError(BaseModel):
    code: str
    message: str


class Dimensions(BaseModel):
    hidden: int | None = None
    heads: int | None = None
    layers: int | None = None
    vocabulary: int | None = None


class AnalysisDetail(BaseModel):
    id: str
    model_id: str
    architecture: str | None
    revision: str | None
    status: AnalysisStatus
    stage: str
    error: AnalysisError | None
    warnings: list[str]
    created_at: str
    expires_at: str | None
    parameter_count: int | None
    root_graph_id: str | None
    cached: bool
    prepared: bool
    dimensions: Dimensions


class AnalysisSummary(AnalysisDetail):
    pass


class AnalysesResponse(BaseModel):
    analyses: list[AnalysisSummary]


class ExampleSummary(BaseModel):
    id: str
    title: str
    model_id: str
    revision: str | None
    architecture: str | None
    parameter_count: int | None
    status: AnalysisStatus


class ExamplesResponse(BaseModel):
    examples: list[ExampleSummary]


class Evidence(BaseModel):
    source: str
    kind: EvidenceKind
    locator: str
    details: dict[str, Any]


class TensorSpec(BaseModel):
    shape: list[int | str] | None
    dtype: str | None
    semantic_name: str


class Repetition(BaseModel):
    count: int
    mode: Literal["sequential", "parallel", "unknown"]
    instance_paths: list[str]


class GraphNode(BaseModel):
    id: str
    label: str
    kind: str
    module_path: str | None
    module_type: str | None
    repeat: int
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    parameter_count: int
    metadata: dict[str, Any]
    evidence: list[Evidence]
    confidence: float
    source_modules: list[str]
    repetition: Repetition | None
    child_graph_id: str | None


class GraphEdge(BaseModel):
    source: str
    target: str
    tensor: TensorSpec | None
    source_port: str
    target_port: str
    label: str | None
    evidence: list[Evidence]
    relation: str


class GraphResponse(BaseModel):
    id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any]


class TensorRecord(BaseModel):
    name: str
    dtype: str
    shape: list[int]
    parameter_count: int
    artifact: str
    parameter_refs: list[str] = Field(default_factory=list)
    role: str | None = None
    evidence_kind: EvidenceKind | None = None
    role_evidence_kind: EvidenceKind | None = None


class NodeInspection(BaseModel):
    graph_id: str
    node: GraphNode
    tensors: list[TensorRecord]
    tensor_count: int
    evidence: list[Evidence]
    module: dict[str, Any] | None
    warnings: list[str]


class Citation(BaseModel):
    id: str
    source: str
    kind: EvidenceKind
    locator: str


class QuestionAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    ai_generated: Literal[True]
    warnings: list[str]
