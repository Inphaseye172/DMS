#!/usr/bin/env python3
"""Writes tiny ONNX models with the same I/O signatures as the real depth models, for unit tests.

  tests/fixtures/onnx/synthetic_depth.onnx     image [B,3,H,W] f32 -> depth [B,H,W] f32
        depth = mean over channels of the (ImageNet-normalized) input  => brighter input = "nearer"
  tests/fixtures/onnx/synthetic_temporal.onnx  frames [B,T,3,H,W] -> depth [B,T,H,W]
        depth[t] = 0.5 * mean_c(frame t) + 0.5 * mean_{t,c}(all frames)   (visible temporal coupling)
  tests/fixtures/onnx/bad_rank.onnx            input rank 2 (must be rejected at session creation)

Requires the `onnx` package. Run: python3 -I tools/make_synthetic_model.py
"""
import os

import onnx
from onnx import TensorProto, helper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "onnx")
OPSET = [helper.make_opsetid("", 17)]


def save(model, name):
    model.ir_version = 8  # readable by ONNX Runtime >= 1.10 (the onnx package defaults to newer IR versions)
    onnx.checker.check_model(model)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    onnx.save(model, path)
    print("wrote", os.path.abspath(path), os.path.getsize(path), "bytes")


def single():
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch_size", 3, "height", "width"])
    y = helper.make_tensor_value_info("depth", TensorProto.FLOAT, ["batch_size", "height", "width"])
    node = helper.make_node("ReduceMean", ["image"], ["depth"], axes=[1], keepdims=0)
    g = helper.make_graph([node], "synthetic_depth", [x], [y])
    save(helper.make_model(g, opset_imports=OPSET, producer_name="dms-tests"), "synthetic_depth.onnx")


def temporal():
    x = helper.make_tensor_value_info("frames", TensorProto.FLOAT, ["batch_size", "t", 3, "height", "width"])
    y = helper.make_tensor_value_info("depth", TensorProto.FLOAT, ["batch_size", "t", "height", "width"])
    half = helper.make_tensor("half", TensorProto.FLOAT, [], [0.5])
    nodes = [
        helper.make_node("ReduceMean", ["frames"], ["per_frame"], axes=[2], keepdims=0),          # [B,T,H,W]
        helper.make_node("ReduceMean", ["per_frame"], ["all_frames"], axes=[1], keepdims=1),       # [B,1,H,W]
        helper.make_node("Add", ["per_frame", "all_frames"], ["sum"]),
        helper.make_node("Mul", ["sum", "half"], ["depth"]),
    ]
    g = helper.make_graph(nodes, "synthetic_temporal", [x], [y], initializer=[half])
    save(helper.make_model(g, opset_imports=OPSET, producer_name="dms-tests"), "synthetic_temporal.onnx")


def bad_rank():
    x = helper.make_tensor_value_info("v", TensorProto.FLOAT, ["n", 3])
    y = helper.make_tensor_value_info("o", TensorProto.FLOAT, ["n", 3])
    g = helper.make_graph([helper.make_node("Identity", ["v"], ["o"])], "bad_rank", [x], [y])
    save(helper.make_model(g, opset_imports=OPSET, producer_name="dms-tests"), "bad_rank.onnx")


if __name__ == "__main__":
    single()
    temporal()
    bad_rank()
