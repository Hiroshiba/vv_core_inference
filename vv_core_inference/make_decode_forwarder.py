import json
from pathlib import Path
from typing import Optional

import numpy
import torch
from hifi_gan.models import Generator as HifiGanPredictor
from torch import nn

from vv_core_inference.make_yukarin_sosoa_forwarder import make_yukarin_sosoa_forwarder
from vv_core_inference.utility import to_tensor


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class WrapperDecodeForwarder(nn.Module):
    def __init__(
        self,
        yukarin_sosoa_forwarder: nn.Module,
        hifi_gan_forwarder: nn.Module,
        device,
    ):
        super().__init__()
        self.yukarin_sosoa_forwarder = yukarin_sosoa_forwarder
        self.hifi_gan_forwarder = hifi_gan_forwarder
        self.device = device

    @torch.no_grad()
    def forward(
        self,
        length: int,
        phoneme_size: int,
        f0: numpy.ndarray,
        phoneme: numpy.ndarray,
        speaker_id: Optional[numpy.ndarray] = None,
    ):
        f0 = to_tensor(f0, device=self.device)
        phoneme = to_tensor(phoneme, device=self.device)

        # forward sosoa
        spec = self.yukarin_sosoa_forwarder(
            f0=f0, phoneme=phoneme, speaker_id=speaker_id
        )

        # forward hifi gan
        x = spec.T
        wave = self.hifi_gan_forwarder(x.unsqueeze(0)).squeeze()
        return wave.cpu().numpy()


def make_decode_forwarder(
    yukarin_sosoa_model_dir: Path, hifigan_model_dir: Path, device
):
    # yukarin_sosoa
    yukarin_sosoa_forwarder = make_yukarin_sosoa_forwarder(
        yukarin_sosoa_model_dir=yukarin_sosoa_model_dir, device=device
    )

    # hifi-gan
    vocoder_model_config = AttrDict(
        json.loads((hifigan_model_dir / "config.json").read_text())
    )

    hifi_gan_predictor = HifiGanPredictor(vocoder_model_config).to(device)
    checkpoint_dict = torch.load(
        hifigan_model_dir.joinpath("model.pth"),
        map_location=device,
    )
    hifi_gan_predictor.load_state_dict(checkpoint_dict["generator"])
    hifi_gan_predictor.eval()
    hifi_gan_predictor.remove_weight_norm()
    print("hifi-gan loaded!")

    decode_forwarder = WrapperDecodeForwarder(
        yukarin_sosoa_forwarder=yukarin_sosoa_forwarder,
        hifi_gan_forwarder=hifi_gan_predictor,
        device=device,
    )

    return decode_forwarder
