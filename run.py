import argparse
import time
from itertools import product
import numpy as np
from pathlib import Path
from typing import List

import soundfile

from vv_core_inference.forwarder import Forwarder


def run(
    yukarin_s_model_dir: Path,
    yukarin_sa_model_dir: Path,
    yukarin_sosoa_model_dir: Path,
    hifigan_model_dir: Path,
    use_gpu: bool,
    use_wgpu: bool,
    texts: List[str],
    speaker_ids: List[int],
    method: str,
):
    np.random.seed(0)
    device = "cpu"
    if use_gpu:
        device = "cuda"
    if use_wgpu:
        device = "wgpu"
    if method == "torch":
        from vv_core_inference.make_decode_forwarder import make_decode_forwarder
        from vv_core_inference.make_yukarin_s_forwarder import make_yukarin_s_forwarder
        from vv_core_inference.make_yukarin_sa_forwarder import (
            make_yukarin_sa_forwarder,
        )
    if method == "onnx":
        import onnxruntime

        from vv_core_inference.onnx_decode_forwarder import make_decode_forwarder
        from vv_core_inference.onnx_yukarin_s_forwarder import make_yukarin_s_forwarder
        from vv_core_inference.onnx_yukarin_sa_forwarder import (
            make_yukarin_sa_forwarder,
        )

        if use_gpu:
            assert onnxruntime.get_device() == "GPU", (
                "Install onnxruntime-gpu if you want to use GPU."
            )
        if use_wgpu:
            assert onnxruntime.get_device() == "CPU-WEBGPU", (
                "Build onnxruntime with --use_webgpu if you want to use WebGPU."
            )

    # yukarin_s
    yukarin_s_forwarder = make_yukarin_s_forwarder(
        yukarin_s_model_dir=yukarin_s_model_dir, device=device
    )

    # yukarin_sa
    yukarin_sa_forwarder = make_yukarin_sa_forwarder(
        yukarin_sa_model_dir=yukarin_sa_model_dir, device=device
    )

    # decoder
    decode_forwarder = make_decode_forwarder(
        yukarin_sosoa_model_dir=yukarin_sosoa_model_dir,
        hifigan_model_dir=hifigan_model_dir,
        device=device,
    )

    # Forwarder。このForwarderクラスの中を書き換えずに
    # yukarin_s_forwarder、yukarin_sa_forwarder、decode_forwarderを置き換えたい。
    forwarder = Forwarder(
        yukarin_s_forwarder=yukarin_s_forwarder,
        yukarin_sa_forwarder=yukarin_sa_forwarder,
        decode_forwarder=decode_forwarder,
    )

    times = []
    for text, speaker_id in product(texts, speaker_ids):
        current = time.time()
        wave = forwarder.forward(
            text=text, speaker_id=speaker_id, f0_speaker_id=speaker_id
        )
        times.append(time.time() - current)
        print(
            f"method={method}, text={text}, speaker_id={speaker_id}, "
            f"elapsed={times[-1]:.3f} seconds, "
        )
        if method == "torch" or method == "onnx":
            soundfile.write(
                f"{method}-{text}-{speaker_id}.wav", data=wave, samplerate=24000
            )

    print(f"Average time for all runs: {np.mean(times):.3f} seconds")
    print(f"Average time excluding first run: {np.mean(times[1:]):.3f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yukarin_s_model_dir", type=Path, default=Path("model/yukarin_s")
    )
    parser.add_argument(
        "--yukarin_sa_model_dir", type=Path, default=Path("model/yukarin_sa")
    )
    parser.add_argument(
        "--yukarin_sosoa_model_dir", type=Path, default=Path("model/yukarin_sosoa")
    )
    parser.add_argument("--hifigan_model_dir", type=Path, default=Path("model/hifigan"))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_wgpu", action="store_true")
    parser.add_argument("--texts", nargs="+", default=["こんにちは、どうでしょう"])
    parser.add_argument("--speaker_ids", nargs="+", type=int, default=[5, 9])
    parser.add_argument("--method", choices=["torch", "onnx"], default="torch")
    run(**vars(parser.parse_args()))
