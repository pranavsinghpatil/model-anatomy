"""Symbolic native checkpoint conversions using names and header shapes only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import math
import re
from typing import Any

from transformers import core_model_loading as native

from .errors import ArtifactError


@dataclass(frozen=True)
class Layout:
    shape: tuple[int, ...]
    dtype: str
    sources: tuple[str, ...]


_OPERATION_METHODS = {
    cls: cls.convert for cls in (
        native.MergeModulelist, native.Concatenate, native.Transpose,
        native.Chunk, native.Split,
    )
}


def source_evidence(value: Any) -> dict[str, str]:
    return {
        'implementation': f'{value.__module__}.{value.__qualname__}',
        'source_sha256': hashlib.sha256(inspect.getsource(value).encode()).hexdigest(),
    }


def _axis(value: Any, rank: int) -> int:
    if type(value) is not int or not -rank <= value < rank:
        raise ValueError('Conversion axis is outside the tensor rank')
    return value % rank


def _combine(values: list[Layout], axis: int, *, stack: bool) -> Layout:
    if not values or len({v.dtype for v in values}) != 1:
        raise ValueError('Conversion requires complete inputs with one storage dtype')
    shape = list(values[0].shape)
    dim = _axis(axis, len(shape) + int(stack))
    if stack:
        if any(v.shape != tuple(shape) for v in values):
            raise ValueError('Stack input shapes disagree')
        shape.insert(dim, len(values))
    else:
        if any(len(v.shape) != len(shape) or any(a != b for i, (a, b) in enumerate(zip(v.shape, shape)) if i != dim) for v in values):
            raise ValueError('Concatenate input shapes disagree')
        shape[dim] = sum(v.shape[dim] for v in values)
    return Layout(tuple(shape), values[0].dtype, tuple(s for v in values for s in v.sources))


def _natural(name: str) -> tuple:
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in name.split('.'))


def _symbolic_operations(rule: Any, groups: dict[str, list[Layout]], expected: dict[str, tuple[int, ...]]) -> tuple[dict[str, Layout], list[dict]]:
    if set(groups) != set(rule.source_patterns):
        raise ValueError('A declared conversion input pattern is missing')
    state: dict[str, Layout | list[Layout]] = dict(groups)
    steps = []
    for operation in rule.operations:
        cls = type(operation)
        if cls not in _OPERATION_METHODS or cls.convert is not _OPERATION_METHODS[cls]:
            raise ValueError('Unsupported or overridden native conversion operation')
        if getattr(operation, 'num_shards_attribute', None) is not None:
            raise ValueError('Configuration-dependent split counts are not established')
        targets = rule.target_patterns
        if cls is native.MergeModulelist:
            merged = {}
            for pattern, inputs in state.items():
                if not isinstance(inputs, list):
                    raise ValueError('Stack requires a source tensor list')
                merged[targets[0] if len(state) == 1 and len(targets) == 1 else pattern] = _combine(inputs, operation.dim, stack=True)
            state = merged
        elif cls is native.Concatenate:
            if len(targets) != 1:
                raise ValueError('Concatenate requires one target')
            values = []
            for pattern in rule.source_patterns:
                if pattern in state:
                    item = state[pattern]
                    values.extend(item if isinstance(item, list) else [item])
            state = {targets[0]: _combine(values, operation.dim, stack=False)}
        elif cls is native.Transpose:
            if len(state) != 1 or len(targets) != 1:
                raise ValueError('Transpose requires one input and one target')
            item = next(iter(state.values()))
            if isinstance(item, list):
                if len(item) != 1:
                    raise ValueError('Transpose input is ambiguous')
                item = item[0]
            shape = list(item.shape)
            if not operation.check_dims or tuple(shape) != expected[targets[0]]:
                a, b = _axis(operation.dim0, len(shape)), _axis(operation.dim1, len(shape))
                shape[a], shape[b] = shape[b], shape[a]
            state = {targets[0]: Layout(tuple(shape), item.dtype, item.sources)}
        else:
            if len(state) != 1:
                raise ValueError('Split requires one input')
            item = next(iter(state.values()))
            if isinstance(item, list):
                if len(item) != 1:
                    raise ValueError('Split input is ambiguous')
                item = item[0]
            dim = _axis(operation.dim, len(item.shape))
            if cls is native.Chunk:
                if item.shape[dim] % len(targets):
                    raise ValueError('Uneven chunk boundaries are not established')
                sizes = [item.shape[dim] // len(targets)] * len(targets)
            else:
                sizes = [expected[target][dim] for target in targets]
                if sum(sizes) != item.shape[dim]:
                    raise ValueError('Split sizes do not cover the source')
            state = {}
            for target, size in zip(targets, sizes):
                shape = list(item.shape)
                shape[dim] = size
                state[target] = Layout(tuple(shape), item.dtype, item.sources)
        steps.append({**source_evidence(cls.convert), 'operation': cls.__name__,
                      'attributes': vars(operation).copy()})
    if set(state) != set(expected) or any(not isinstance(item, Layout) or item.shape != expected[key] for key, item in state.items()):
        raise ValueError('Converted header shapes do not match all native targets')
    inputs = {name: value for group in groups.values() for value in group for name in value.sources}
    if sum(math.prod(v.shape) for v in inputs.values()) != sum(math.prod(v.shape) for v in state.values()):
        raise ValueError('Conversion does not conserve tensor elements')
    return state, steps


def converted_bindings(model: Any, checkpoint: Any, renamed: dict[str, str], rules: list[Any]) -> tuple[dict[str, dict], list[dict]]:
    """Account for every input and target before accepting a packed parameter binding."""
    parameters = dict(model.named_parameters(remove_duplicate=False))
    prefix = model.base_model_prefix + '.' if model.base_model_prefix else ''
    converters = [r for r in rules if type(r) is native.WeightConverter]
    batches: dict[tuple[int, str], dict[str, list[tuple[str, str]]]] = {}
    failures: list[dict] = []
    for source in sorted(checkpoint.tensors, key=_natural):
        name = renamed.get(source, source)
        for index, rule in enumerate(converters):
            target, pattern = rule.rename_source_key(name)
            if pattern is None:
                continue
            batches.setdefault((index, target), {}).setdefault(pattern, []).append((source, name))
            break
    bindings: dict[str, dict] = {}
    used: set[str] = set()
    catalog = None
    if batches:
        from transformers import conversion_mapping

        catalog = {
            'selection': source_evidence(conversion_mapping.get_model_conversion_mapping),
            'catalog_source_sha256': hashlib.sha256(inspect.getsource(conversion_mapping).encode()).hexdigest(),
            'model_class': f'{type(model).__module__}.{type(model).__qualname__}',
            'config_model_type': getattr(getattr(model, 'config', None), 'model_type', None),
        }
    for (index, first_target), buckets in batches.items():
        rule = converters[index]
        sources = [source for bucket in buckets.values() for source, _ in bucket]
        try:
            if rule.quantization_operation is not None or rule.distributed_operation is not None:
                raise ValueError('Quantized or distributed conversion is outside metadata layout proof')
            first_pattern = rule.target_patterns[0]
            if first_target.count(first_pattern) != 1:
                raise ValueError('Converter target expansion is ambiguous')
            start, _, end = first_target.partition(first_pattern)
            targets = {}
            for pattern in rule.target_patterns:
                name = start + pattern + end
                candidates = {name, prefix + name, name.removeprefix(prefix)} & parameters.keys()
                if len(candidates) != 1:
                    raise ValueError('Converter target has no unique native owner')
                targets[pattern] = next(iter(candidates))
            groups = {}
            for pattern, values in buckets.items():
                if '*.' in pattern:
                    # Only a single numeric module-list selector has a bounded stacking proof.
                    if pattern.count('*.') != 1:
                        raise ValueError('Multiple module-list selectors are unsupported')
                    expression = pattern.replace('*.', r'(?P<index>\d+)\.')
                    indices = []
                    for _, name in values:
                        match = re.search(expression, name)
                        if match is None:
                            raise ValueError('Source selector is not a numeric module index')
                        indices.append(int(match['index']))
                    if indices != list(range(len(indices))):
                        raise ValueError('Module-list indices are missing or duplicated')
                groups[pattern] = [Layout(checkpoint.tensors[s].shape, checkpoint.tensors[s].dtype, (s,)) for s, _ in values]
            expected = {pattern: tuple(parameters[name].shape) for pattern, name in targets.items()}
            layouts, steps = _symbolic_operations(rule, groups, expected)
            if set(sources) & used or set(targets.values()) & bindings.keys():
                raise ArtifactError('Conflicting native checkpoint conversions')
            proof = {'rule': 'native_header_layout_conversion', 'source_patterns': rule.source_patterns,
                     'target_patterns': rule.target_patterns, 'scope_prefix': rule.scope_prefix,
                     'native_transform': source_evidence(type(rule).rename_source_key),
                     'native_catalog': catalog,
                     'operations': steps, 'input_shapes': {s: list(checkpoint.tensors[s].shape) for s in sources},
                     'tensor_values_verified': False, 'weight_payload_bytes_read': 0}
            for pattern, layout in layouts.items():
                bindings[targets[pattern]] = {'sources': list(layout.sources), 'shape': list(layout.shape),
                                               'dtype': layout.dtype, 'evidence': proof}
            used.update(sources)
        except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
            failures.append({'rule': 'native_header_layout_conversion', 'target': first_target,
                             'source_patterns': rule.source_patterns, 'sources': sources,
                             'reason': str(exc), 'blocking': True})
    return bindings, failures
