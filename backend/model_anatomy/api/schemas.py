"""Public HTTP schemas; reconstruction IR remains owned by the engine."""

from typing import Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Readiness(BaseModel):
    status: Literal["unavailable"] = "unavailable"
    reason: Literal["reconstruction_not_implemented"] = "reconstruction_not_implemented"


class Capabilities(BaseModel):
    api_version: Literal["v1"] = "v1"
    reconstruction_available: Literal[False] = False
    supported_models: list[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
