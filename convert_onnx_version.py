# https://github.com/onnx/onnx/blob/18c23d92ec80a3f679e883a460fa6ae8a9457c0a/docs/PythonAPIOverview.md#converting-version-of-an-onnx-model-within-default-domain-aionnx

import argparse

import onnx
from onnx import helper, version_converter

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("output_model_path")
parser.add_argument("target_version", type=int, help="Target ONNX version")
args = parser.parse_args()

# Preprocessing: load the model to be converted.
model_path = args.model_path
original_model = onnx.load(model_path)

# print("The model before conversion:\n{}".format(original_model))

# A full list of supported adapters can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
# Apply the version conversion on the original model
converted_model = version_converter.convert_version(original_model, args.target_version)

# print("The model after conversion:\n{}".format(converted_model))

onnx.save(converted_model, args.output_model_path)
