import argparse
from itertools import product
from pathlib import Path
from typing import List

import soundfile

from vv_core_inference.forwarder import Forwarder
from vv_core_inference.make_decode_forwarder import make_decode_forwarder
from vv_core_inference.make_yukarin_s_forwarder import make_yukarin_s_forwarder
from vv_core_inference.make_yukarin_sa_forwarder import make_yukarin_sa_forwarder


def run(
    yukarin_s_model_dir: Path,
    yukarin_sa_model_dir: Path,
    yukarin_sosoa_model_dir: Path,
    hifigan_model_dir: Path,
    use_gpu: bool,
    texts: List[str],
    speaker_ids: List[int],
):
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

    forwarder = Forwarder(
        yukarin_s_forwarder=yukarin_s_forwarder,
        yukarin_sa_forwarder=yukarin_sa_forwarder,
        decode_forwarder=decode_forwarder,
    )

    for text, speaker_id in product(texts, speaker_ids):
        wave = forwarder.forward(
            text=text, speaker_id=speaker_id, f0_speaker_id=speaker_id
        )
        soundfile.write(f"{text}-{speaker_id}.wav", data=wave, samplerate=24000)


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
    run(**vars(parser.parse_args()))
