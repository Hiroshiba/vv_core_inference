from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor, nn
from yukarin_s.config import Config
from yukarin_s.network.predictor import Predictor, create_predictor

from vv_core_inference.utility import to_tensor


class WrapperYukarinS(nn.Module):
    def __init__(self, predictor: Predictor, device):
        super().__init__()
        self.predictor = predictor
        self.device = device

    @torch.no_grad()
    def forward(self, length: int, phoneme_list: Tensor, speaker_id: Optional[Tensor]):
        phoneme_list = to_tensor(phoneme_list, device=self.device)

        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=self.device)
            speaker_id = speaker_id.reshape((1,)).to(torch.int64)

        output = self.predictor(
            phoneme_list=phoneme_list.unsqueeze(0), speaker_id=speaker_id
        )[0]
        return output.cpu().numpy()


def make_yukarin_s_forwarder(yukarin_s_model_dir: Path, device):
    with yukarin_s_model_dir.joinpath("config.yaml").open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    predictor = create_predictor(config.network)
    state_dict = torch.load(
        yukarin_s_model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    print("yukarin_s loaded!")

    yukarin_s_forwarder = WrapperYukarinS(predictor, device)
    return yukarin_s_forwarder
