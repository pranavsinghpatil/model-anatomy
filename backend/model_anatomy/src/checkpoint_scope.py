"""Evidence for checkpoint state outside the selected native module graph."""

from __future__ import annotations

from typing import Any

from .checkpoint_transforms import source_evidence


def classify_inactive_components(model: Any, raw: dict, alternate: Any, alternate_raw: dict) -> None:
    """Bind optional state to an exact native counterpart whose slot is disabled here.

    An alternate task is only a source of component identity. Its execution graph
    is never substituted for the selected model, and shapes alone cannot bind a
    checkpoint component.
    """
    for source, item in raw['checkpoint_extras'].items():
        if item['kind'] != 'unmapped':
            continue
        converted = raw['checkpoint_name_conversions'].get(source, {}).get('target', source)
        parts = converted.split('.')
        owner, owner_path, disabled = model, '', None
        for index, name in enumerate(parts[:-1]):
            children = vars(owner).get('_modules', {})
            if name in children and children[name] is not None:
                owner = children[name]
                owner_path = '.'.join(parts[:index + 1])
            elif (name in children and children[name] is None) or vars(owner).get(name, object()) is None:
                disabled = '.'.join(parts[index:])
                break
            else:
                break
        if disabled is None:
            continue
        alternatives = []
        for path, record in alternate_raw['parameters'].items():
            if source not in record['tensor_names'] or record.get('checkpoint_layout'):
                continue
            for prefix, counterpart in alternate.named_modules():
                target = f'{prefix}.{disabled}' if prefix else disabled
                if type(counterpart) is type(owner) and target == path:
                    parent_config = getattr(owner, 'config', None)
                    other_config = getattr(counterpart, 'config', None)
                    if parent_config is None or other_config is None:
                        continue
                    first = {k: v for k, v in parent_config.to_dict().items() if k != 'architectures'}
                    second = {k: v for k, v in other_config.to_dict().items() if k != 'architectures'}
                    if first != second:
                        continue
                    from .architecture_selection import compatible_dtype
                    if not compatible_dtype(record['module_dtype'], record['dtype']):
                        continue
                    parameter = alternate.get_parameter(path)
                    alternatives.append((path, counterpart, list(parameter.shape), str(parameter.dtype)))
        if len(alternatives) != 1:
            continue
        path, counterpart, shape, dtype = alternatives[0]
        item.update(kind='inactive_component_state', evidence_kind='reconstructed',
                    reason='An exact native counterpart owns this optional component; its slot is None in the selected model',
                    selected_parent_path=owner_path, disabled_component=disabled.split('.')[0],
                    alternate_parameter_path=path, alternate_model_class=type(alternate).__name__,
                    native_shape=shape, native_dtype=dtype,
                    constructor=source_evidence(type(counterpart).__init__),
                    primary_graph_presence=False, tensor_values_verified=False,
                    blocking_for_architecture=False)


def buffer_storage_relation(native_dtype: str, stored_dtype: str) -> dict | None:
    """Describe byte storage for an already owned boolean buffer, without claiming values."""
    if native_dtype == 'torch.bool' and stored_dtype == 'U8':
        return {'rule': 'owned_boolean_buffer_byte_storage',
                'native_dtype': native_dtype, 'checkpoint_dtype': stored_dtype,
                'scope': 'Same registered owner and shape; byte values and boolean conversion remain unverified',
                'tensor_values_verified': False, 'blocking_for_architecture': False}
    return None


def checkpoint_differences(raw: dict, compatibility: dict) -> dict:
    """Separate binding gaps, identified inactive state and numerical-only notes."""
    missing = {
        name: {'kind': 'missing_native_parameter', 'shape': item['shape'],
               'blocking_for_architecture': True,
               'reason': 'A registered parameter has no proven checkpoint binding; inactivity has not been established'}
        for name, item in raw['parameters'].items()
        if name == item['canonical_parameter'] and not item['checkpoint_matched']
    }
    return {
        'missing_native_parameters': missing,
        'checkpoint_state': raw['checkpoint_extras'],
        'missing_persistent_buffers': compatibility['missing_persistent_buffers'],
        'buffer_storage_notes': compatibility['numerical_integrity']['buffer_storage_notes'],
        'dtype_conflicts': compatibility['dtype_conflicts'],
        'layout_conversions': raw.get('checkpoint_layout_conversions', {}),
        'unsupported_conversions': raw.get('checkpoint_conversion_failures', []),
        'scope': 'Names, native components and symbolic layouts; checkpoint payload values remain unverified',
    }
