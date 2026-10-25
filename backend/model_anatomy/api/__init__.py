"""HTTP boundary for Model Anatomy, independent of model-library imports."""

from model_anatomy.dependencies import require_dependencies

require_dependencies("api", component="API")
