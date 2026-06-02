import builtins

import pytest

from deployment import export_bm_tracker_onnx as exporter


def test_reads_onnx_names_without_onnxruntime(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Identity",
                inputs=["actual_input"],
                outputs=["actual_output"],
            )
        ],
        name="test_graph",
        inputs=[
            helper.make_tensor_value_info(
                "actual_input", TensorProto.FLOAT, [None, 3]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "actual_output", TensorProto.FLOAT, [None, 3]
            )
        ],
    )
    model = helper.make_model(graph)
    onnx_path = tmp_path / "model.onnx"
    onnx.save(model, onnx_path)

    real_import = builtins.__import__

    def import_without_onnxruntime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_onnxruntime)

    input_names, output_names = exporter._read_onnx_io_names(
        onnx_path,
        fallback_input_names=["fallback_input"],
        fallback_output_names=["fallback_output"],
    )

    assert input_names == ["actual_input"]
    assert output_names == ["actual_output"]
