from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor, nn
from yukarin_sa.config import Config
from yukarin_sa.network.predictor import Predictor, create_predictor

from vv_core_inference.utility import remove_weight_norm, to_tensor


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

    @torch.no_grad()
    def forward(
        self,
        length: int,
        vowel_phoneme_list: Tensor,
        consonant_phoneme_list: Tensor,
        start_accent_list: Tensor,
        end_accent_list: Tensor,
        start_accent_phrase_list: Tensor,
        end_accent_phrase_list: Tensor,
        speaker_id: Optional[Tensor],
    ):
        vowel_phoneme_list = vowel_phoneme_list.unsqueeze(0)
        consonant_phoneme_list = consonant_phoneme_list.unsqueeze(0)
        start_accent_list = start_accent_list.unsqueeze(0)
        end_accent_list = end_accent_list.unsqueeze(0)
        start_accent_phrase_list = start_accent_phrase_list.unsqueeze(0)
        end_accent_phrase_list = end_accent_phrase_list.unsqueeze(0)

        batch_size = 1
        length = vowel_phoneme_list.shape[1]

        ph = self.phoneme_embedder(vowel_phoneme_list + 1) + self.phoneme_embedder(
            consonant_phoneme_list + 1
        )  # (batch_size, length, ?)
        ph = ph.transpose(1, 2)  # (batch_size, ?, length)

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

        if speaker_id is not None:
            speaker_id = self.speaker_embedder(speaker_id)  # (batch_size, ?)
            speaker_id = speaker_id.unsqueeze(2)  # (batch_size, ?, 1)
            speaker = speaker_id.expand(
                speaker_id.shape[0], speaker_id.shape[1], ph.shape[2]
            )  # (batch_size, ?, length)
            h = torch.cat((h, speaker), dim=1)  # (batch_size, ?, length)

        h = self.encoder(h)  # (batch_size, ?, length)

        if self.ar_encoder is not None:
            f0 = torch.zeros(
                batch_size, length, dtype=h.dtype, device=h.device
            )  # (batch_size, length)

            f0_one = torch.zeros(
                batch_size, 1, 1, dtype=h.dtype, device=h.device
            )  # (batch_size, 1, 1)
            hidden: Optional[Tensor] = None
            for i in range(length):
                h_one = h[:, :, i : i + 1]  # (batch_size, ?, 1)
                h_one = torch.cat((h_one, f0_one), dim=1)  # (batch_size, ?, 1)
                h_one, hidden = self.ar_encoder(
                    h_one, hidden=hidden
                )  # (batch_size, ?, 1)
                f0_one = self.post(h_one)  # (batch_size, 1, 1)

                f0[:, i] = f0_one[:, 0, 0]  # (batch_size, length)

        else:
            h = self.post(h)  # (batch_size, 1, length)
            f0 = h[:, 0, :]  # (batch_size, length)

        return f0[0]  # (length,)


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
            speaker_id
        )
        return wrapper(*args).cpu().numpy()
    return _dispatcher

