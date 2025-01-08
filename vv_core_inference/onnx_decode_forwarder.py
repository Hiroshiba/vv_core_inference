from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
import onnxruntime

def make_decode_forwarder(yukarin_sosoa_model_dir: Path, hifigan_model_dir: Path, device, convert=False):
    providers = ['CPUExecutionProvider']
    if device == "cuda":
      providers.insert(0, 'CUDAExecutionProvider')
    elif device == "dml":
        providers.insert(0, 'DmlExecutionProvider')
    session = onnxruntime.InferenceSession(
      str(yukarin_sosoa_model_dir.joinpath("decode.onnx")),
      providers=providers
    )

    def _dispatcher(
        length: int,
        phoneme_size: int,
        f0: ndarray,
        phoneme: ndarray,
        speaker_id: Optional[ndarray] = None,
    ):
        f0 = np.asarray(f0)
        phoneme = np.asarray(phoneme)
        if speaker_id is not None:
            speaker_id = np.asarray(speaker_id)
            speaker_id = speaker_id.reshape((1,)).astype(np.int64)
        wave = session.run(["wave"], {
            "f0": f0,
            "phoneme": phoneme,
            "speaker_id": speaker_id,
        })[0]
        return None, wave
    return _dispatcher
