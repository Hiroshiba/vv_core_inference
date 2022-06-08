import json
from pathlib import Path
from typing import Optional

import numpy
import torch
from hifi_gan.models import Generator as HifiGanPredictor
from torch import nn

from vv_core_inference.make_yukarin_sosoa_forwarder import make_yukarin_sosoa_wrapper
from vv_core_inference.utility import OPSET, to_tensor


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class WrapperDecodeForwarder(nn.Module):
    def __init__(
        self,
        yukarin_sosoa_forwarder: nn.Module,
        hifi_gan_forwarder: nn.Module,
    ):
        super().__init__()
        self.yukarin_sosoa_forwarder = yukarin_sosoa_forwarder
        self.hifi_gan_forwarder = hifi_gan_forwarder

    @torch.no_grad()
    def forward(
        self,
        f0: torch.Tensor,
        phoneme: torch.Tensor,
        speaker_id: torch.Tensor,
    ):
        # forward sosoa
        spec = self.yukarin_sosoa_forwarder(
            f0=f0, phoneme=phoneme, speaker_id=speaker_id
        )

        # forward hifi gan
        x = spec.transpose(1, 0)
        wave = self.hifi_gan_forwarder(x.unsqueeze(0))[0, 0]
        return wave


def make_decode_forwarder(
    yukarin_sosoa_model_dir: Path, hifigan_model_dir: Path, device, convert=False
):
    # yukarin_sosoa
    yukarin_sosoa_wrapper = make_yukarin_sosoa_wrapper(
        yukarin_sosoa_model_dir=yukarin_sosoa_model_dir, device=device
    )

    # hifi-gan
    vocoder_model_config = AttrDict(
        json.loads((hifigan_model_dir / "config.json").read_text())
    )

    hifi_gan_predictor = HifiGanPredictor(vocoder_model_config).to(device)
    checkpoint_dict = torch.load(
        # hifigan_model_dir.joinpath("model.pth"),
        hifigan_model_dir.joinpath("hoge.pt"),
        map_location=device,
    )
    hifi_gan_predictor.load_state_dict(checkpoint_dict["generator"])
    hifi_gan_predictor.eval()
    hifi_gan_predictor.remove_weight_norm()
    print("hifi-gan loaded!")

    decode_forwarder = WrapperDecodeForwarder(
        yukarin_sosoa_forwarder=yukarin_sosoa_wrapper,
        hifi_gan_forwarder=hifi_gan_predictor,
    )

    def _dispatcher(
        length: int,
        phoneme_size: int,
        f0: numpy.ndarray,
        phoneme: numpy.ndarray,
        speaker_id: Optional[numpy.ndarray] = None,
    ):
        f0 = to_tensor(f0, device=device)
        phoneme = to_tensor(phoneme, device=device)
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
        if convert:
            torch.onnx.export(
                decode_forwarder,
                (f0, phoneme, speaker_id),
                yukarin_sosoa_model_dir.joinpath("decode.onnx"),
                opset_version=OPSET,
                do_constant_folding=True,
                input_names=["f0", "phoneme", "speaker_id"],
                output_names=["wave"],
                # dynamic_axes={
                #     "f0": {0: "length"},
                #     "phoneme": {0: "length"},
                #     "wave": {0: "outlength"},
                # },
            )
            print("decode has been converted to ONNX")
        return decode_forwarder(f0, phoneme, speaker_id).cpu().numpy()

    return _dispatcher
