"""Public HTTP schemas; reconstruction IR remains owned by the engine."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthChecks(BaseModel):
    database: bool
    storage: bool
    worker: bool


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: Literal["v1"] = "v1"
    checks: HealthChecks
    analysis_available: bool
    ai_available: bool
    storage_backend: Literal["local", "supabase"]
    reason: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class CreateAnalysis(BaseModel):
    model: str = Field(
        min_length=1,
        max_length=512,
        description="Hugging Face repository ID or HTTPS URL to inspect.",
    )
    refresh: bool = False


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    graph_id: str | None = Field(default=None, max_length=128)
    node_id: str | None = Field(default=None, max_length=128)


AnalysisStatus = Literal[
    "queued",
    "running",
    "completed",
    "partial",
    "inspected",
    "unsupported",
    "failed",
    "timed_out",
    "cancelled",
]
EvidenceKind = Literal["direct", "derived", "reconstructed", "unknown"]


class AnalysisError(BaseModel):
    code: str
    message: str


class Dimensions(BaseModel):
    hidden: int | None = None
    heads: int | list[int] | None = None
    layers: int | None = None
    vocabulary: int | None = None


class CatalogItem(BaseModel):
    id: str
    model_id: str
    revision: str
    architecture: str | None
    status: Literal["completed", "partial", "inspected"]
    parameter_count: int | None
    dimensions: Dimensions
    engine_version: str
    schema_version: str
    artifact: str
    explicit_artifact: bool
    inspected_at: str


class CatalogResponse(BaseModel):
    models: list[CatalogItem]
    total: int
    offset: int
    limit: int


class BenchmarkCatalogResponse(BaseModel):
    """Measured publication; case dictionaries are allowlisted by the catalog service."""

    schema_version: Literal[1]
    generated_at: str
    engine_version: str
    manifest_sha256: str
    manifest_hash_encoding: Literal["canonical-json-v1"]
    method: str
    status_counts: dict[str, int]
    results: list[dict[str, Any]]
    runs: list[dict[str, Any]]
    baseline_cases: int | None = None
    recheck_attempts: int | None = None
    engine_versions: list[str] | None = None
    runner_sha256: str | None = None
    engine_source_sha256: str | None = None
    baseline_runner_sha256: str | None = None
    baseline_engine_source_sha256: str | None = None


class DiscoveryCatalogResponse(BaseModel):
    results: list[dict[str, Any]]
    total: int
    status_counts: dict[str, int]
    generated_at: str
    benchmark_publication_id: str | None
    external_publications: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisQuality(BaseModel):
    semantic_coverage: float | None = Field(default=None, ge=0, le=1)
    module_semantic_coverage: float | None = Field(default=None, ge=0, le=1)
    checkpoint_parameter_coverage: float | None = Field(default=None, ge=0, le=1)
    tensor_shape_coverage: float | None = Field(default=None, ge=0, le=1)
    unknown_nodes: int | None = Field(default=None, ge=0)
    unmapped_checkpoint_tensors: int | None = Field(default=None, ge=0)
    forward_executed: bool | None = None
    runtime_trace_available: bool | None = None
    mode: str | None = None


class DeclaredCheckpointIndex(BaseModel):
    url: str
    canonical_sha256: str
    declared_tensor_names: int = Field(ge=0)
    declared_shards: int = Field(ge=0)
    tensor_record_limit: int = Field(ge=0)


class InspectionSummary(BaseModel):
    kind: Literal["artifact_inspection"]
    reason_code: str
    message: str
    file_count: int
    tensor_count: int
    checkpoint_tensor_element_count: int | None
    checkpoint_available: bool
    declared_checkpoint_index: DeclaredCheckpointIndex | None = None
    architecture_blockers: list[str] = Field(default_factory=list)


class AnalysisDetail(BaseModel):
    id: str
    model_id: str
    architecture: str | None
    revision: str | None
    status: AnalysisStatus
    stage: str
    error: AnalysisError | None
    warnings: list[str]
    limitations: list[str] = Field(default_factory=list)
    partial_reasons: list[str] = Field(default_factory=list)
    quality: AnalysisQuality = Field(default_factory=AnalysisQuality)
    engine_version: str | None = None
    created_at: str
    expires_at: str | None
    parameter_count: int | None
    root_graph_id: str | None
    cached: bool
    dimensions: Dimensions
    input_names: list[str] = Field(default_factory=list)
    inspection: InspectionSummary | None = None


class AnalysisSummary(AnalysisDetail):
    pass


class AnalysesResponse(BaseModel):
    analyses: list[AnalysisSummary]


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
    shape_convention: Literal["tensor_axes", "ggml_dimensions"] = "tensor_axes"
    tensor_type_id: int | None = None
    parameter_count: int
    artifact: str
    parameter_refs: list[str] = Field(default_factory=list)
    role: str | None = None
    evidence_kind: EvidenceKind | None = None
    role_evidence_kind: EvidenceKind | None = None


class ModelComponent(BaseModel):
    path: str
    library: str | None
    declared_class: str | None
    checkpoint_files: list[str]


class ModelProfile(BaseModel):
    repository_kind: str
    declared_task: str | None
    declared_library: str | None
    config_model_type: str | None
    declared_architectures: list[str]
    formats: list[str]
    selected_file: str | None
    selected_config: str | None
    components: list[ModelComponent]
    architecture_blockers: list[str]
    metadata_basis: str


class ArtifactInspectionResponse(BaseModel):
    kind: Literal["artifact_inspection", "architecture"]
    reason_code: str | None
    message: str | None
    revision: str
    files: list[str]
    file_count: int
    tensors: list[TensorRecord]
    tensor_count: int
    checkpoint_tensor_element_count: int | None
    checkpoint_available: bool
    declared_checkpoint_index: DeclaredCheckpointIndex | None = None
    checkpoint_format: str | None = None
    selected_file_tensor_element_count: int | None = None
    selected_file_split_metadata: dict[str, int] | None = None
    model_profile: ModelProfile | None = None
    source_config: dict[str, Any] | None
    evidence: list[Evidence]
    offset: int
    limit: int


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
