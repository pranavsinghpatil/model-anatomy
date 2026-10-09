"""Shape transformations must retain unknowns rather than invent dimensions."""

import pytest

from model_anatomy.src.ir import TensorSpec
from model_anatomy.src.shape_ops import operation_shape


@pytest.mark.parametrize(
    ("kind", "metadata", "shapes", "expected"),
    [
        ("reshape", {"method": "flatten", "dimensions": [2]}, [["B", 24, 2, 3]], ["B", 24, 6]),
        ("reshape", {"method": "flatten", "dimensions": [2]}, [["B", 24, "H", 3]], None),
        ("reshape", {"method": "transpose", "dimensions": [1, -1]}, [["B", 24, 6]], ["B", 6, 24]),
        ("reshape", {"method": "transpose", "dimensions": [1, 9]}, [["B", 24, 6]], None),
        ("reshape", {"method": "expand", "dimensions": ["B", -1, -1]}, [[1, 1, 24]], ["B", 1, 24]),
        ("reshape", {"method": "expand", "dimensions": ["B", 3, 12]}, [[1, 1, 24]], None),
        ("reshape", {"method": "view", "dimensions": [-1, 24]}, [["B", 6, 24]], None),
        (
            "reshape",
            {"method": "flatten", "dimensions": [2], "keyword_arguments": True},
            [["B", 24, 2, 3]],
            None,
        ),
        ("selection", {"indices": [None, 0, None]}, [["B", 5, 24]], ["B", 24]),
        ("selection", {"indices": [None, 5, None]}, [["B", 5, 24]], None),
        ("selection", {"indices": [None, "unknown", None]}, [["B", 5, 24]], None),
        ("concatenation", {"method": "cat", "dimension": 1}, [["B", 1, 24], ["B", 6, 24]], ["B", 7, 24]),
        ("concatenation", {"method": "cat", "dimension": 1}, [["B", 1, 24], ["B", 6, 12]], None),
        ("concatenation", {"method": "stack", "dimension": 1}, [["B", 1, 24], ["B", 6, 24]], None),
        ("concatenation", {"method": "cat", "dimension": 1}, [["B", 1, 24], None], None),
    ],
)
def test_bounded_shape_algebra(kind, metadata, shapes, expected):
    specs = [TensorSpec(shape, "F32", "test") for shape in shapes]
    assert operation_shape(kind, metadata, specs).shape == expected
