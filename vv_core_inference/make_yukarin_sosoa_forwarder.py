from pathlib import Path
from typing import List, Optional

import numpy
import torch
import yaml
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from yukarin_sosoa.config import Config
from yukarin_sosoa.network.predictor import Predictor, create_predictor

from vv_core_inference.utility import remove_weight_norm, to_tensor


def make_pad_mask(lengths: Tensor):
    bs = lengths.shape[0]
    maxlen = lengths.max()

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: Tensor):
    return ~make_pad_mask(lengths)


class WrapperPostnet(nn.Module):
    def __init__(self, net: Postnet):
        super().__init__()
        self.postnet = net.postnet

    def forward(self, xs):
        for net in self.postnet:
            xs = net(xs)
        return xs


class WrapperYukarinSosoa(nn.Module):
    def __init__(self, predictor: Predictor):
        super().__init__()

        self.speaker_embedder = predictor.speaker_embedder
        self.pre = predictor.pre
        self.encoder = predictor.encoder
        self.post = predictor.post
        self.postnet = WrapperPostnet(predictor.postnet)

    @torch.no_grad()
    def forward(
        self,
        f0: Tensor,
        phoneme: Tensor,
        speaker_id: Tensor,
    ):
        f0 = f0.unsqueeze(0)
        phoneme = phoneme.unsqueeze(0)

        h = torch.cat((f0, phoneme), dim=2)  # (batch_size, length, ?)

        speaker_id = self.speaker_embedder(speaker_id)
        speaker_id = speaker_id.unsqueeze(dim=1)  # (batch_size, 1, ?)
        speaker_feature = speaker_id.expand(
            speaker_id.shape[0], h.shape[1], speaker_id.shape[2]
        )  # (batch_size, length, ?)
        h = torch.cat((h, speaker_feature), dim=2)  # (batch_size, length, ?)

        h = self.pre(h)

        mask = torch.ones_like(f0).squeeze()
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        return output2[0]

def make_yukarin_sosoa_wrapper(yukarin_sosoa_model_dir: Path, device) -> nn.Module:
    with yukarin_sosoa_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    state_dict = torch.load(
        yukarin_sosoa_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    predictor.apply(remove_weight_norm)
    predictor.encoder.embed[0].pe = predictor.encoder.embed[0].pe.to(device)
    print("yukarin_sosoa loaded!")
    return WrapperYukarinSosoa(predictor)

def make_yukarin_sosoa_forwarder(yukarin_sosoa_model_dir: Path, device):
    yukarin_sosoa_forwarder = make_yukarin_sosoa_wrapper(yukarin_sosoa_model_dir, device)

    def _dispatcher(
        f0: Tensor,
        phoneme: Tensor,
        speaker_id: Optional[numpy.ndarray] = None,
    ):
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
        return yukarin_sosoa_forwarder(f0, phoneme, speaker_id)

    return _dispatcher
