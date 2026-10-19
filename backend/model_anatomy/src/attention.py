"""Attention views derived from selected source operations, never keyword templates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .ir import Evidence, Graph
from .shape_rules import known

if TYPE_CHECKING:
    from .hierarchy import HierarchyBuilder


def build_attention(builder: HierarchyBuilder, path: str) -> Graph:
    """Retain source dataflow instead of asserting an unproved attention template."""
    graph = builder.source_graph(path)
    report = builder.flow_reports.get(path, {})
    failures = [item['reason'] for item in report.get('unexplained_operations', [])]
    nodes = [node for child in graph.walk() for node in child.nodes]
    contracts = [node.metadata['shape_check'] for node in nodes if node.metadata.get('shape_check')]
    for child in graph.walk():
        failures.extend(child.metadata.get('warnings', []))
    failures.extend(check.get('reason', 'Shape contract is unproven') for check in contracts
                    if check['status'] in {'unknown', 'invalid'})
    failures.extend(node.metadata['unsupported']['reason'] for node in nodes if node.metadata.get('unsupported'))
    conditions = report.get('branch_conditions', [])
    layouts = [{'node_id': node.id, 'rule': node.metadata['shape_check']['rule'],
                'shape': node.metadata['shape_check'].get('output_shape'),
                'status': node.metadata['shape_check']['status'],
                'source': node.metadata.get('source_location')}
               for node in graph.nodes if node.kind == 'reshape' and node.metadata.get('shape_check')]
    inputs = [{'argument': node.metadata.get('argument_name'), 'shape': spec.shape, 'dtype': spec.dtype}
              for node in graph.nodes if node.kind == 'input' for spec in node.outputs]
    multiplications = [{'node_id': node.id, 'inputs': [spec.shape for spec in node.inputs],
                        'outputs': [spec.shape for spec in node.outputs],
                        'source': node.metadata.get('source_location')}
                       for node in graph.nodes if node.metadata.get('shape_check', {}).get('rule') in {'matmul_contract', 'bmm_contract'}]
    softmax = [{'node_id': node.id, 'axis': node.metadata.get('normalization_axis'),
                'source': node.metadata.get('source_location')}
               for node in graph.nodes if node.metadata.get('native_callable') == 'torch.nn.functional.softmax']
    scales = [{'node_id': node.id, 'operator': node.metadata.get('operator'),
               'scalar': node.metadata.get('scalar'), 'source': node.metadata.get('source_location')}
              for node in graph.nodes if node.metadata.get('shape_check', {}).get('rule') in
              {'tensor_scalar_Mult', 'tensor_scalar_Div', 'elementwise_Mult', 'elementwise_Div'}]
    helpers = [item for item in report.get('resolved_operations', []) if item['rule'] == 'selected_python_helper']
    incoming = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.relation == 'dataflow':
            incoming[edge.target].append(edge.source)

    def ancestors(node_id: str) -> set[str]:
        pending, seen = [node_id], set()
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(incoming[current])
        return seen

    matrices = {item['node_id'] for item in multiplications}
    normalizations = {item['node_id'] for item in softmax if item['axis'] == -1}
    output_dependencies = set().union(*(ancestors(node.id) for node in graph.nodes if node.kind == 'output'))
    lineage = []
    for aggregate in matrices & output_dependencies:
        operands = incoming[aggregate]
        if len(operands) != 2:
            continue
        for normalization in normalizations & ancestors(operands[0]):
            for scores in (matrices - {aggregate}) & ancestors(normalization):
                lineage.append({'scores': scores, 'normalization': normalization, 'aggregation': aggregate})
    proof: dict[str, Any] = {
        'rule': 'selected_source_operations', 'status': 'source_partial' if failures else 'source_established',
        'module_path': path, 'conditions': conditions, 'profile': report.get('input_profile', []),
        'source': builder.raw['source_fingerprints'].get(path),
        'failure_reasons': list(dict.fromkeys(failures)),
        'obligations': {
            'input_layout': {'status': 'established' if inputs and all(known(item['shape']) for item in inputs) else 'unknown', 'inputs': inputs},
            'projection_heads_layout': {'status': 'source_only', 'layouts': layouts,
                'reason': 'Actual reshape/transpose dimensions are retained; no config-derived head layout is inserted'},
            'scale': {'status': 'source_only', 'operations': scales,
                'reason': 'Only selected scalar or tensor arithmetic is represented; no default inverse-square-root scale'},
            'mask': {'status': 'source_only',
                'reason': 'Actual predicate selection/addition and buffer dependencies are retained; causal/window buffer values are not equated'},
            'rotary': {'status': 'source_only',
                'reason': 'Only interpreted helper/module operations are retained; no class-name-based rotary stage'},
            'kernel_branch': {'status': 'established' if not failures else 'unknown', 'selected_helpers': helpers},
        },
        'matrix_operations': multiplications, 'softmax_operations': softmax,
        'attention_pattern': 'source_observed' if lineage and not failures else 'unknown',
        'attention_lineage': lineage,
        'attention_pattern_reason': '' if lineage and not failures else 'Score/normalization/aggregation output lineage is unproven; retain source dependencies',
        'synthetic_nodes_added': False, 'semantic_template_applied': False,
        'tensor_execution': False, 'forward_executed': False,
        'scope': 'Recorded profile and source operations; semantic template remains a source fallback',
    }
    if proof['attention_pattern'] == 'unknown':
        graph.metadata.setdefault('warnings', []).append(
            'Attention semantics are unproven; selected source graph retained without a semantic template')
    graph.metadata.update(topology='attention source dependencies', attention_proof=proof)
    for node in graph.nodes:
        check = node.metadata.get('shape_check', {})
        reason = node.metadata.get('unsupported', {}).get('reason') or check.get('reason')
        details = {'rule': check.get('rule', node.metadata.get('source_rule', 'source_dependency')),
                   'status': check.get('status', 'unknown' if node.metadata.get('unsupported') else 'source_dependency'),
                   'graph_proof': graph.id, 'source': node.metadata.get('source_location') or proof['source'],
                   'conditions': node.metadata.get('branch_evidence', []),
                   'failure_reasons': [reason] if reason else [], 'runtime_verified': False}
        node.metadata['refinement'] = details
        node.evidence.append(Evidence('python_source', 'unknown' if node.metadata.get('unsupported') else 'derived',
                                     'selected_attention_source', details))
        if node.kind == 'residual' and 'skip_argument' not in node.metadata:
            node.kind, node.label = 'add', 'Elementwise add'
    for edge in graph.edges:
        edge.evidence.append(Evidence('python_source', 'derived', 'selected_attention_dependency',
            {'rule': 'source_dataflow', 'graph_proof': graph.id, 'source_node': edge.source,
             'target_node': edge.target, 'failure_reasons': proof['failure_reasons'], 'runtime_verified': False}))
    return graph
