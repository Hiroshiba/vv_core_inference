from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
import onnxruntime


def make_yukarin_s_forwarder(yukarin_s_model_dir: Path, device, convert=False):
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    session = onnxruntime.InferenceSession(
        str(yukarin_s_model_dir.joinpath("duration.onnx")), providers=providers
    )

    def _dispatcher(length: int, phoneme_list: ndarray, speaker_id: Optional[ndarray]):
        phoneme_list = np.asarray(phoneme_list)
        if speaker_id is not None:
            speaker_id = np.asarray(speaker_id)
            speaker_id = speaker_id.reshape((1,)).astype(np.int64)
        return session.run(
            ["phoneme_length"],
            {
                "phoneme_list": phoneme_list,
                "speaker_id": speaker_id,
            },
        )[0]

    return _dispatcher
