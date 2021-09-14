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
    def __init__(self, predictor: Predictor, device):
        super().__init__()

        predictor.encoder.embed[0].pe = predictor.encoder.embed[0].pe.to(device)

        self.speaker_embedder = predictor.speaker_embedder
        self.pre = predictor.pre
        self.encoder = predictor.encoder
        self.post = predictor.post
        self.postnet = WrapperPostnet(predictor.postnet)
        self.device = device

    @torch.no_grad()
    def forward(
        self,
        f0_list: List[Tensor],
        phoneme_list: List[Tensor],
        speaker_id: Optional[numpy.ndarray] = None,
    ):
        length_list = [f0.shape[0] for f0 in f0_list]

        length = torch.tensor(length_list).to(f0_list[0].device)
        f0 = pad_sequence(f0_list, batch_first=True)
        phoneme = pad_sequence(phoneme_list, batch_first=True)

        h = torch.cat((f0, phoneme), dim=2)  # (batch_size, length, ?)

        if self.speaker_embedder is not None and speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=self.device)
            speaker_id = self.speaker_embedder(speaker_id)
            speaker_id = speaker_id.unsqueeze(dim=1)  # (batch_size, 1, ?)
            speaker_feature = speaker_id.expand(
                speaker_id.shape[0], h.shape[1], speaker_id.shape[2]
            )  # (batch_size, length, ?)
            h = torch.cat((h, speaker_feature), dim=2)  # (batch_size, length, ?)

        h = self.pre(h)

        mask = make_non_pad_mask(length).to(length.device).unsqueeze(-2)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        return [output2[i, :l] for i, l in enumerate(length_list)]


def make_yukarin_sosoa_forwarder(yukarin_sosoa_model_dir: Path, device):
    with yukarin_sosoa_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    state_dict = torch.load(
        yukarin_sosoa_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    predictor.apply(remove_weight_norm)
    print("yukarin_sosoa loaded!")

    yukarin_sosoa_forwarder = WrapperYukarinSosoa(predictor, device=device)
    return yukarin_sosoa_forwarder
