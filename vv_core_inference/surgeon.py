import numpy as np
import onnx
import onnx_graphsurgeon as gs

@gs.Graph.register()
def replace_ConvTranspose(self, node):
    assert node.op == "ConvTranspose"
    in_tensor, weight, bias = node.inputs
    in_tensor.outputs.clear()
    weight.outputs.clear()
    bias.outputs.clear()
    out_tensor = node.outputs[0]
    out_tensor.inputs.clear()

    kernel_size = node.attrs["kernel_shape"]

    assert len(kernel_size) == 1, "only supports conv_transpose1d"
    kernel_size = kernel_size[0]
    groups = node.attrs["group"]
    dilation = node.attrs["dilations"][0]
    padding = node.attrs["pads"]
    stride = node.attrs["strides"][0]

    assert groups == 1
    assert dilation == 1
    assert padding[0] == padding[1]
    padding = padding[0]

    weight_numpy = weight.values
    weight_numpy_conv = np.ascontiguousarray(weight_numpy.transpose(1,0,2)[:,:,::-1])

    print("replace", node.name)
    h1 = self.layer(op="Unsqueeze", inputs=[in_tensor], outputs=["expanded"], attrs={"axes": [3]})[0]
    h2 = self.layer(op="Pad", inputs=[h1, [0, 0, 0, 0, 0, 0, 0, stride-1]], outputs=["pad_inner"])[0]
    h3 = self.layer(op="Reshape", inputs=[h2, [0, 0, -1]], outputs=["unpooled"])[0]
    h4 = self.layer(op="Pad", inputs=[h3, np.array([0, 0, kernel_size - padding - 1, 0, 0, kernel_size - padding - stride], np.int64)], outputs=["pad_outer"])[0]
    return self.layer(op="Conv", inputs=[h4, weight_numpy_conv, bias.values], outputs=[out_tensor], attrs={
        "dilations": [1],
        "group": 1,
        "kernel_shape": [kernel_size],
        "pads": [0, 0],
        "strides": [1]
    })

@gs.Graph.register()
def replace_Conv(self, node):
    # 1d -> 2d (webgl only supports conv2d)
    assert node.op == "Conv"
    in_tensor, weight, bias = node.inputs
    in_tensor.outputs.clear()
    weight.outputs.clear()
    bias.outputs.clear()
    out_tensor = node.outputs[0]
    out_tensor.inputs.clear()

    kernel_size = node.attrs["kernel_shape"]
    assert len(kernel_size) == 1, "only supports conv1d"
    kernel_size = kernel_size[0]
    groups = node.attrs["group"]
    dilation = node.attrs["dilations"][0]
    padding = node.attrs["pads"]
    stride = node.attrs["strides"][0]

    print("replace", node.name)
    h1 = self.layer(op="Unsqueeze", inputs=[in_tensor], outputs=["in_2d"], attrs={"axes": [3]})[0]
    h2 = self.layer(op="Conv", inputs=[h1, weight.values[:, :, :, None], bias], outputs=["out_2d"], attrs={
        "dilations": [dilation, dilation],
        "group": groups,
        "kernel_shape": [kernel_size, 1],
        "pads": [padding[0], 0, padding[1], 0],
        "strides": [stride, stride],
    })[0]
    return self.layer(op="Squeeze", inputs=[h2], outputs=[out_tensor], attrs={"axes": [3]})

def fold_unsqueeze(node):
    if node.op != "Squeeze":
        return
    squeeze = node
    axes = node.attrs["axes"]
    if not (len(node.outputs[0].outputs) == 1 and node.o().op == "LeakyRelu"):
        return
    relu = node.o()
    if not (len(relu.outputs[0].outputs) == 1 and relu.o().op == "Unsqueeze" and relu.o().attrs["axes"] == axes):
        return
    unsqueeze = relu.o()

    in_node = squeeze.i()
    in_node.outputs = squeeze.outputs
    squeeze.outputs.clear()

    relu.outputs = unsqueeze.outputs
    unsqueeze.outputs.clear()
    print("eliminate", node.name)


def surgeon(filename, outname):
    graph = gs.import_onnx(onnx.load(filename))
    # ConvTranspose -> Conv
    targets = [node for node in graph.nodes if node.op == "ConvTranspose"]
    for node in targets:
        graph.replace_ConvTranspose(node)
    graph.cleanup().toposort()
    # Conv1d -> Conv2d
    targets = [node for node in graph.nodes if node.op == "Conv"]
    for node in targets:
        graph.replace_Conv(node)
    graph.cleanup().toposort()
    # fold --Squeeze--LeakyRelu--Unsqueeze-- into --LeakyRelu--
    targets = [node for node in graph.nodes if node.op == "Squeeze"]
    for node in targets:
        fold_unsqueeze(node)
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), outname)

# surgeon("model/hifigan/hifigan.onnx", "model/hifigan/hifigan_modified.onnx")
# surgeon("model/hifigan/hifigan.onnx", "../vv_check_web/public/hifigan.onnx")