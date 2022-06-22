import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs

parser = argparse.ArgumentParser(description="Convert ONNX model to WebGL format")
parser.add_argument("input", help="Input ONNX model")
parser.add_argument("output", help="Output WebGL model")
args = parser.parse_args()

input_path = args.input
output_path = args.output


@gs.Graph.register()
def replace_ConstantOfShape(self, node):
    assert node.op == "ConstantOfShape"
    in_tensor = node.inputs[0]
    in_tensor.outputs.clear()
    out_tensor = node.outputs[0]
    out_tensor.inputs.clear()

    zeros_like_in_tensor = self.layer(
        op="Mul", inputs=[in_tensor, np.zeros(1, np.int64)], outputs=["zeros_like"]
    )[0]
    ones_like_in_tensor = self.layer(
        op="Add",
        inputs=[zeros_like_in_tensor, np.ones(1, np.int64)],
        outputs=["ones_like"],
    )[0]
    value = self.layer(
        op="Reshape",
        inputs=[node.attrs["value"].values, ones_like_in_tensor],
        outputs=["value"],
    )[0]
    return self.layer(op="Tile", inputs=[value, in_tensor], outputs=[out_tensor])


graph = gs.import_onnx(onnx.load(input_path))
for node in graph.nodes:
    if node.op == "ConstantOfShape":
        graph.replace_ConstantOfShape(node)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), output_path)

print("done!")
