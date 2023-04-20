from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime
from numpy import ndarray


def make_yukarin_sosf_forwarder(yukarin_sosf_model_dir: Path, device, convert=False):
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    session = onnxruntime.InferenceSession(
        str(yukarin_sosf_model_dir.joinpath("contour.onnx")), providers=providers
    )

    def _dispatcher(
        length: int,
        f0: np.ndarray,
        phoneme: np.ndarray,
        speaker_id: Optional[np.ndarray] = None,
    ):
        length = np.asarray(length).astype(np.int64)
        f0 = np.asarray(f0)
        phoneme = np.asarray(phoneme)
        if speaker_id is not None:
            speaker_id = np.asarray(speaker_id).astype(np.int64)

        f0, voiced = session.run(
            ["f0", "voiced"],
            {
                "length": length,
                "f0": f0,
                "phoneme": phoneme,
                "speaker_id": speaker_id,
            },
        )
        return f0, voiced

    return _dispatcher
