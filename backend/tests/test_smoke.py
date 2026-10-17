from model_anatomy.ir.schema import Graph, ModelIR


def test_ir_smoke():
    ir = ModelIR(model_id="example/model", architecture="Unknown", root_graph=Graph("overview", [], []))
    assert ir.model_id == "example/model"
    assert ir.root_graph.nodes == []
    assert ModelIR.from_json(ir.to_json()).model_id == "example/model"
