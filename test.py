import argparse
from itertools import product
from pathlib import Path
from typing import List

import numpy as np

from vv_core_inference.forwarder import Forwarder

def run(
    yukarin_s_model_dir: Path,
    yukarin_sa_model_dir: Path,
    yukarin_sosoa_model_dir: Path,
    hifigan_model_dir: Path,
    use_gpu: bool,
    texts: List[str],
    speaker_ids: List[int],
    method: str,
):
    if method == "torch":
        from vv_core_inference.make_decode_forwarder import make_decode_forwarder
        from vv_core_inference.make_yukarin_s_forwarder import make_yukarin_s_forwarder
        from vv_core_inference.make_yukarin_sa_forwarder import make_yukarin_sa_forwarder
    if method == "onnx":
        from vv_core_inference.onnx_decode_forwarder import make_decode_forwarder
        from vv_core_inference.onnx_yukarin_s_forwarder import make_yukarin_s_forwarder
        from vv_core_inference.onnx_yukarin_sa_forwarder import make_yukarin_sa_forwarder

    np.random.seed(0)
    device = "cuda" if use_gpu else "cpu"
    result = {
        "s": None,
        "sa": None,
        "decode": None,
    }

    # yukarin_s
    yukarin_s_forwarder = make_yukarin_s_forwarder(
        yukarin_s_model_dir=yukarin_s_model_dir, device=device
    )
    def _s(**kwargs):
        x = yukarin_s_forwarder(**kwargs)
        result["s"] = x
        return x

    # yukarin_sa
    yukarin_sa_forwarder = make_yukarin_sa_forwarder(
        yukarin_sa_model_dir=yukarin_sa_model_dir, device=device
    )
    def _sa(**kwargs):
        x = yukarin_sa_forwarder(**kwargs)
        result["sa"] = x
        return x

    # decoder
    decode_forwarder = make_decode_forwarder(
        yukarin_sosoa_model_dir=yukarin_sosoa_model_dir,
        hifigan_model_dir=hifigan_model_dir,
        device=device,
    )
    def _decode(**kwargs):
        x = decode_forwarder(**kwargs)
        result["decode"] = x
        return x

    # Forwarder。このForwarderクラスの中を書き換えずに
    # yukarin_s_forwarder、yukarin_sa_forwarder、decode_forwarderを置き換えたい。
    forwarder = Forwarder(
        yukarin_s_forwarder=_s,
        yukarin_sa_forwarder=_sa,
        decode_forwarder=_decode,
    )

    for text, speaker_id in product(texts, speaker_ids):
        _wave = forwarder.forward(
            text=text, speaker_id=speaker_id, f0_speaker_id=speaker_id
        )
    return result

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
    parser.add_argument("--texts", nargs="+", default=["こんにちは、どうでしょう"])
    parser.add_argument("--speaker_ids", nargs="+", type=int, default=[5, 9])

    torch_result = run(**vars(parser.parse_args()), method="torch")
    onnx_result = run(**vars(parser.parse_args()), method="onnx")

    for key in ["s", "sa", "decode"]:
        print(key, np.allclose(torch_result[key], onnx_result[key]))

    print(np.abs(torch_result["decode"] - onnx_result["decode"]).max())
    print(np.abs(torch_result["decode"] - onnx_result["decode"]).max() / np.abs(torch_result["decode"]).max())