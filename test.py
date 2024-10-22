import argparse
from itertools import product
from pathlib import Path
import time
from typing import List

import numpy as np
from tqdm import tqdm
import yaml

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
        yukarin_sosf_forwarder=None,
        decode_forwarder=decode_forwarder,
    )

    result = []
    proctime = []
    for text, speaker_id in tqdm(product(texts, speaker_ids), desc=method):
        tic = time.process_time()
        wave = forwarder.forward(
            text=text, speaker_id=speaker_id, f0_speaker_id=speaker_id
        )
        tac = time.process_time()
        result.append(wave)
        proctime.append(tac - tic)
    return result, proctime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path, default=Path("model"), help="path to the folder of torch models"
    )
    parser.add_argument(
        "--model", type=Path, default=Path("onnxmodel"), help="path to the folder of onnx models"
    )
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path("test_dataset.yaml"))
    parser.add_argument("--speaker_id", type=int, default=5)
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        test_data = yaml.safe_load(f)["text"]
    print(f"loaded {len(test_data)} texts")

    torch_waves, torch_times = run(
        yukarin_s_model_dir=args.baseline/"yukarin_s",
        yukarin_sa_model_dir=args.baseline/"yukarin_sa",
        yukarin_sosoa_model_dir=args.baseline/"yukarin_sosoa",
        hifigan_model_dir=args.baseline/"hifigan",
        use_gpu=args.use_gpu,
        texts=test_data,
        speaker_ids=[args.speaker_id],
        method="torch"
    )
    onnx_waves, onnx_times = run(
        yukarin_s_model_dir=args.model,
        yukarin_sa_model_dir=args.model,
        yukarin_sosoa_model_dir=args.model,
        hifigan_model_dir=args.model,
        use_gpu=args.use_gpu,
        texts=test_data,
        speaker_ids=[args.speaker_id],
        method="onnx"
    )
    torch_alltime = np.sum(torch_times)
    onnx_alltime = np.sum(onnx_times)
    
    torch_signal = np.concatenate(torch_waves)
    onnx_signal = np.concatenate(onnx_waves)
    peak_pow = (torch_signal.max() - torch_signal.min()) ** 2
    noise_pow = np.mean((torch_signal - onnx_signal) ** 2)

    print(f"=== time for processing {len(test_data)} texts ===")
    print(f"baseline: {torch_alltime:.3f} sec")
    print(f"model: {onnx_alltime:.3f} sec")
    print(f"x{torch_alltime / onnx_alltime:.3f} faster")
    print("=== model's PSNR (higher is better) ===")
    print(10 * np.log10(peak_pow / noise_pow), "dB")