from model_anatomy.ir.schema import ModelIR


def test_ir_smoke():
    ir = ModelIR(model_id="example/model")
    assert ir.model_id == "example/model"
    assert ir.components == []
