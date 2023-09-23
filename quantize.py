import argparse
import logging
from pathlib import Path
import shutil
from typing import List, Optional

import numpy as np
import onnxruntime
from onnxruntime.quantization import quant_pre_process, quantize_static, CalibrationDataReader
from tqdm import tqdm
import yaml

from vv_core_inference.onnx_decode_forwarder import make_decode_forwarder
from vv_core_inference.onnx_yukarin_s_forwarder import make_yukarin_s_forwarder
from vv_core_inference.onnx_yukarin_sa_forwarder import make_yukarin_sa_forwarder
from vv_core_inference.onnx_yukarin_sosf_forwarder import make_yukarin_sosf_forwarder
from vv_core_inference.forwarder import Forwarder


class CalibrationDataset(CalibrationDataReader):
    def __init__(self, calibration_texts: List[str], forwarder: Forwarder, speaker_size: int):
        self.forwarder = forwarder
        self.speaker_size = speaker_size
        self.calibration_data = calibration_texts
        self.i = 0
        self.bar = tqdm(total=len(self.calibration_data))

    def __len__(self):
        return len(self.calibration_data)
    
    def get_next(self) -> dict:
        if self.i >= len(self.calibration_data):
            self.bar.close()
            return None
        sample_text = self.calibration_data[self.i]
        # self.bar.set_postfix_str(sample_text)
        _wave, intermediates = self.forwarder.forward(
            text = sample_text,
            speaker_id = self.i % self.speaker_size,
            f0_speaker_id = self.i % self.speaker_size,
            return_intermediate_results = True,
            run_until="decode"
        )
        self.i += 1
        self.bar.update(1)
        inputs = intermediates["yukarin_sosoa_input"]
        inputs["f0"] = np.asarray(inputs["f0"])
        inputs["phoneme"] = np.asarray(inputs["phoneme"])
        inputs["speaker_id"] = np.asarray(inputs["speaker_id"]).reshape((1,)).astype(np.int64)
        return inputs
        

def quantize_vocoder(onnx_dir: Path, output_dir: Path, speaker_size: int, use_gpu: bool, calibration_file: str):
    if use_gpu:
        assert onnxruntime.get_device() == "GPU", "Install onnxruntime-gpu if you want to use GPU."

    output_dir.mkdir(exist_ok=True)

    device = "cuda" if use_gpu else "cpu"
    logger = logging.getLogger("quantize")

    logger.info("loading calibration texts")
    with open(calibration_file, encoding="utf-8") as f:
        calibration_data = yaml.safe_load(f)["text"]
    logger.info(f"loaded {len(calibration_data)} texts")

    logger.info("loading forwarder")
    yukarin_s_forwarder = make_yukarin_s_forwarder(
        yukarin_s_model_dir=onnx_dir, device=device
    )
    yukarin_sa_forwarder = make_yukarin_sa_forwarder(
        yukarin_sa_model_dir=onnx_dir, device=device
    )
    if onnx_dir.joinpath("contour.onnx").exists():
        yukarin_sosf_forwarder = make_yukarin_sosf_forwarder(
            yukarin_sosf_model_dir=onnx_dir,
            device=device
        )
    else:
        yukarin_sosf_forwarder = None
    decode_forwarder = make_decode_forwarder(
        yukarin_sosoa_model_dir=onnx_dir,
        hifigan_model_dir=None,
        device=device
    )

    forwarder = Forwarder(
        yukarin_s_forwarder=yukarin_s_forwarder,
        yukarin_sa_forwarder=yukarin_sa_forwarder,
        yukarin_sosf_forwarder=yukarin_sosf_forwarder,
        decode_forwarder=decode_forwarder,
    )
    logger.info(f"loaded forwarder in {device} mode")

    dataset = CalibrationDataset(calibration_data[:30], forwarder, speaker_size)

    logger.info("preprocess decode.onnx (shape inference and optimization)")
    # TODO: apply symbolic shape (with_hn部分で止まってしまうので切り離す必要がある)
    quant_pre_process(
        str(onnx_dir.joinpath("decode.onnx")),
        str(output_dir.joinpath("decode_prequant.onnx")),
        skip_symbolic_shape=True)

    quantize_static(
        str(output_dir.joinpath("decode_prequant.onnx")),
        str(output_dir.joinpath("decode.onnx")),
        dataset,
        op_types_to_quantize=["Conv", "ConvTranspose"])

    shutil.copy(onnx_dir.joinpath("duration.onnx"), output_dir)
    shutil.copy(onnx_dir.joinpath("intonation.onnx"), output_dir)
    if onnx_dir.joinpath("contour.onnx").exists():
        shutil.copy(onnx_dir.joinpath("contour.onnx"), output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=Path, default="onnxmodel")
    parser.add_argument("--output_dir", type=Path, default="quantmodel")
    parser.add_argument("--speaker_size", type=int, default=1)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--calibration_file", type=Path, default="calibration_dataset.yaml")
    quantize_vocoder(**vars(parser.parse_args()))
