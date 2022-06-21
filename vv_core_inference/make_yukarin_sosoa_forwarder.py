import math
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


class RelPositionalEncoding(torch.nn.Module):
    """Variant of espnet_pytorch_library/transformer/embedding.py#RelPositionalEncoding
    copyright 2019 shigeki karita
    apache 2.0  (http://www.apache.org/licenses/license-2.0)
    """
    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Construct an PositionalEncoding object."""
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model//2, 2)
        pe_negative = torch.zeros(x.size(1), self.d_model//2, 2)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, :, 0] = torch.sin(position * div_term)
        pe_positive[:, :, 1] = torch.cos(position * div_term)
        pe_negative[:, :, 0] = torch.sin(-1 * position * div_term)
        pe_negative[:, :, 1] = torch.cos(-1 * position * div_term)

        pe_positive = pe_positive.view(x.size(1), self.d_model)
        pe_negative = pe_negative.view(x.size(1), self.d_model)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)

        x = x * self.xscale
        pos_emb = pe[
            :,
            pe.size(1) // 2 - x.size(1) + 1 : pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)

def make_pad_mask(lengths: Tensor):
    bs = lengths.shape[0]
    maxlen = lengths.max()

    seq_range = torch.arange(0, maxlen, dtype=torch.int32, device=lengths.device)
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

        # mask = torch.ones_like(f0).squeeze()
        # h, _ = self.encoder(h, mask)
        h, _ = self.encoder(h, None)

        output1 = self.post(h)
        output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        return output2[0]

def make_yukarin_sosoa_wrapper(yukarin_sosoa_model_dir: Path, device) -> nn.Module:
    with yukarin_sosoa_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    pe = predictor.encoder.embed[-1]
    predictor.encoder.embed[-1] = RelPositionalEncoding(pe.d_model, pe.dropout.p) # Use my dynamic positional encoding version
    state_dict = torch.load(
        yukarin_sosoa_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    predictor.apply(remove_weight_norm)
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
