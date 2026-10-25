"""Storage dtype names used by checkpoint compatibility checks."""

import torch

DTYPES = {
    torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.float64: "F64", torch.int64: "I64", torch.int32: "I32", torch.bool: "BOOL",
}
