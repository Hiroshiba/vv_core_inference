from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
import onnxruntime

def make_yukarin_sa_forwarder(yukarin_sa_model_dir: Path, device, convert=False):
    providers = ['CPUExecutionProvider']
    if device == "cuda":
      providers.insert(0, 'CUDAExecutionProvider')
    elif device == "dml":
        providers.insert(0, 'DmlExecutionProvider')
    session = onnxruntime.InferenceSession(
      str(yukarin_sa_model_dir.joinpath("intonation.onnx")),
      providers=providers
    )

    def _dispatcher(
        length: int,
        vowel_phoneme_list: ndarray,
        consonant_phoneme_list: ndarray,
        start_accent_list: ndarray,
        end_accent_list: ndarray,
        start_accent_phrase_list: ndarray,
        end_accent_phrase_list: ndarray,
        speaker_id: Optional[ndarray],
    ):
        length = np.asarray(length).astype(np.int64)
        vowel_phoneme_list = np.asarray(vowel_phoneme_list)
        consonant_phoneme_list = np.asarray(consonant_phoneme_list)
        start_accent_list = np.asarray(start_accent_list)
        end_accent_list = np.asarray(end_accent_list)
        start_accent_phrase_list = np.asarray(
            start_accent_phrase_list
        )
        end_accent_phrase_list = np.asarray(end_accent_phrase_list)

        if speaker_id is not None:
            speaker_id = np.asarray(speaker_id)
            speaker_id = speaker_id.reshape((-1,)).astype(np.int64)

        output = session.run(["f0_list"], {
            "length": length,
            "vowel_phoneme_list": vowel_phoneme_list,
            "consonant_phoneme_list": consonant_phoneme_list,
            "start_accent_list": start_accent_list,
            "end_accent_list": end_accent_list,
            "start_accent_phrase_list": start_accent_phrase_list,
            "end_accent_phrase_list": end_accent_phrase_list,
            "speaker_id": speaker_id,
        })[0]
        return output
    return _dispatcher
