from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
import onnxruntime

def make_decode_forwarder(yukarin_sosoa_model_dir: Path, hifigan_model_dir: Path, device, convert=False):
    session_sosoa = onnxruntime.InferenceSession(str(yukarin_sosoa_model_dir.joinpath("yukarin_sosoa.onnx")))
    session_hifi = onnxruntime.InferenceSession(str(hifigan_model_dir.joinpath("hifigan_modified.onnx")))

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
        spec = session_sosoa.run(["spec"], {
            "f0": f0,
            "phoneme": phoneme,
            "speaker_id": speaker_id,
        })[0]
        return session_hifi.run(["wave"], {
            "spec": spec,
        })[0]
    return _dispatcher
