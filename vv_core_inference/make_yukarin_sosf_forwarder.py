from pathlib import Path
from typing import List, Optional

import numpy
import torch
import yaml
from torch import Tensor, nn
from yukarin_sosf.config import Config
from yukarin_sosf.network.predictor import Predictor, create_predictor

from vv_core_inference.make_yukarin_sosoa_forwarder import (
    RelPositionalEncoding,
    WrapperPostnet,
)
from vv_core_inference.utility import remove_weight_norm, to_tensor


class WrapperYukarinSosf(nn.Module):
    def __init__(self, predictor: Predictor):
        super().__init__()

        self.speaker_embedder = predictor.speaker_embedder
        self.phoneme_embedder = predictor.phoneme_embedder
        self.pre = predictor.pre
        self.encoder = predictor.encoder
        self.post = predictor.post
        self.postnet = WrapperPostnet(predictor.postnet)

    @torch.no_grad()
    def forward(
        self,
        f0_discrete: Tensor,
        phoneme: Tensor,
        speaker_id: Tensor,
    ):
        f0_discrete = f0_discrete.unsqueeze(0)
        phoneme = phoneme.unsqueeze(0)

        phoneme = self.phoneme_embedder(phoneme)  # (B, L, ?)

        speaker_id = self.speaker_embedder(speaker_id)
        speaker_id = speaker_id.unsqueeze(dim=1)  # (B, 1, ?)
        speaker_feature = speaker_id.expand(
            speaker_id.shape[0], f0_discrete.shape[1], speaker_id.shape[2]
        )  # (B, L, ?)

        h = torch.cat((f0_discrete, phoneme, speaker_feature), dim=2)  # (B, L, ?)
        h = self.pre(h)

        mask = torch.ones_like(f0_discrete).squeeze()
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)

        f0_contour, voiced = output2[:, :, 0], output2[:, :, 1]

        return f0_contour[0], voiced[0]


def make_yukarin_sosf_wrapper(yukarin_sosf_model_dir: Path, device) -> nn.Module:
    with yukarin_sosf_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    pe = predictor.encoder.embed[-1]
    predictor.encoder.embed[-1] = RelPositionalEncoding(
        pe.d_model, pe.dropout.p
    )  # Use my dynamic positional encoding version
    state_dict = torch.load(
        yukarin_sosf_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    predictor.apply(remove_weight_norm)
    print("yukarin_sosf loaded!")
    return WrapperYukarinSosf(predictor)


def make_yukarin_sosf_forwarder(yukarin_sosf_model_dir: Path, device):
    yukarin_sosf_forwarder = make_yukarin_sosf_wrapper(yukarin_sosf_model_dir, device)

    def _dispatcher(
        f0_discrete: numpy.ndarray,
        phoneme: numpy.ndarray,
        speaker_id: Optional[numpy.ndarray] = None,
    ):
        f0_discrete = to_tensor(f0_discrete, device=device)
        phoneme = to_tensor(phoneme, device=device)
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
        f0_contour, voiced = yukarin_sosf_forwarder(f0_discrete, phoneme, speaker_id)
        return f0_contour.cpu().numpy(), voiced.cpu().numpy()

    return _dispatcher
