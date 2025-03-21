from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor, nn
from yukarin_sa.config import Config
from yukarin_sa.network.predictor import Predictor, create_predictor

from vv_core_inference.utility import remove_weight_norm, to_tensor, OPSET


class WrapperUniGRU(nn.Module):
    def __init__(self, rnn: nn.GRU):
        super().__init__()
        self.rnn = rnn

    def forward(self, x: Tensor, hidden: Optional[Tensor] = None):
        output, hidden = self.rnn(x.transpose(1, 2), hidden)
        return output.transpose(1, 2), hidden


class WrapperYukarinSa(nn.Module):
    def __init__(self, predictor: Predictor):
        super().__init__()
        self.phoneme_embedder = predictor.phoneme_embedder
        self.speaker_embedder = predictor.speaker_embedder
        self.encoder = predictor.encoder
        self.ar_encoder = WrapperUniGRU(predictor.ar_encoder.rnn)
        self.post = predictor.post
        _rnn = self.ar_encoder.rnn
        num_directions = 2 if _rnn.bidirectional else 1
        self.ar_encoder_hidden_shape = (
            _rnn.num_layers * num_directions,
            1, _rnn.hidden_size)

    def forward(
        self,
        length: Tensor,
        vowel_phoneme_list: Tensor,
        consonant_phoneme_list: Tensor,
        start_accent_list: Tensor,
        end_accent_list: Tensor,
        start_accent_phrase_list: Tensor,
        end_accent_phrase_list: Tensor,
        speaker_id: Tensor,
    ):
        batch_size = 1

        vowel_phoneme_list = vowel_phoneme_list.unsqueeze(0)
        consonant_phoneme_list = consonant_phoneme_list.unsqueeze(0)
        start_accent_list = start_accent_list.unsqueeze(0)
        end_accent_list = end_accent_list.unsqueeze(0)
        start_accent_phrase_list = start_accent_phrase_list.unsqueeze(0)
        end_accent_phrase_list = end_accent_phrase_list.unsqueeze(0)

        ph = self.phoneme_embedder(vowel_phoneme_list + 1) + self.phoneme_embedder(
            consonant_phoneme_list + 1
        )  # (batch_size, length, _phenome_emb)
        ph = ph.transpose(1, 2)  # (batch_size, _phenome_emb, length)

        ah = torch.stack(
            [
                start_accent_list,
                end_accent_list,
                start_accent_phrase_list,
                end_accent_phrase_list,
            ],
            dim=1,
        ).to(
            ph.dtype
        )  # (batch_size, ?, length)

        h = torch.cat((ph, ah), dim=1)  # (batch_size, ?, length)

        speaker_id = self.speaker_embedder(speaker_id)  # (batch_size, _speaker_emb)
        speaker_id = speaker_id.unsqueeze(2)  # (batch_size, _speaker_emb, 1)
        speaker = speaker_id.expand(
            speaker_id.shape[0], speaker_id.shape[1], ph.shape[2]
        )  # (batch_size, _speaker_emb, length)
        encoder_input = torch.cat((h, speaker), dim=1)  # (batch_size, encoder_emb = _phoneme_emb + 4 + speaker_emb, length)
        encoder_input = encoder_input.view(1, -1, length)

        h = self.encoder(encoder_input)  # (batch_size, encoder_emb, length)

        f0_one = torch.zeros(
            batch_size, 1, 1, dtype=h.dtype, device=h.device
        )  # (batch_size, 1, 1)

        hidden = torch.zeros(
            self.ar_encoder_hidden_shape,
            device=h.device)
        f0 = []
        for i in range(int(length)):
            h_one = h[:, :, i : i + 1]  # (batch_size, encoder_emb, 1)
            ar_encoder_input = torch.cat((h_one, f0_one), dim=1)  # (batch_size, encoder_emb+1, 1)
            h_one, hidden = self.ar_encoder(
                ar_encoder_input, hidden=hidden
            )  # (batch_size, ?, 1)
            f0_one = self.post(h_one)  # (batch_size, 1, 1)

            f0 += [f0_one[:, 0, 0]]

        return torch.stack(f0, dim=1)[0]  # (length,)


def make_yukarin_sa_forwarder(yukarin_sa_model_dir: Path, device):
    with yukarin_sa_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    state_dict = torch.load(
        yukarin_sa_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    predictor.apply(remove_weight_norm)
    print("yukarin_sa loaded!")
    wrapper = WrapperYukarinSa(predictor)

    
    @torch.no_grad()
    def _dispatcher(
        length: int,
        vowel_phoneme_list: Tensor,
        consonant_phoneme_list: Tensor,
        start_accent_list: Tensor,
        end_accent_list: Tensor,
        start_accent_phrase_list: Tensor,
        end_accent_phrase_list: Tensor,
        speaker_id: Optional[Tensor],
    ):
        length = to_tensor(length, device=device).to(torch.int64)
        vowel_phoneme_list = to_tensor(vowel_phoneme_list, device=device)
        consonant_phoneme_list = to_tensor(consonant_phoneme_list, device=device)
        start_accent_list = to_tensor(start_accent_list, device=device)
        end_accent_list = to_tensor(end_accent_list, device=device)
        start_accent_phrase_list = to_tensor(
            start_accent_phrase_list, device=device
        )
        end_accent_phrase_list = to_tensor(end_accent_phrase_list, device=device)

        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
            speaker_id = speaker_id.reshape((-1,)).to(torch.int64)

        args = (
            length,
            vowel_phoneme_list,
            consonant_phoneme_list,
            start_accent_list,
            end_accent_list,
            start_accent_phrase_list,
            end_accent_phrase_list,
            speaker_id,
        )
        output = wrapper(*args)
        return output.cpu().numpy()
    return _dispatcher

