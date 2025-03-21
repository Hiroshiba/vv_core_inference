from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import Tensor, nn
from yukarin_s.config import Config
from yukarin_s.network.predictor import Predictor, create_predictor

from vv_core_inference.utility import to_tensor, OPSET


class WrapperYukarinS(nn.Module):
    def __init__(self, predictor: Predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, phoneme_list: Tensor, speaker_id: Tensor):
        output = self.predictor(
            phoneme_list=phoneme_list.unsqueeze(0), speaker_id=speaker_id
        )[0]
        return output


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

    yukarin_s_forwarder = WrapperYukarinS(predictor)
    @torch.no_grad()
    def _dispatcher(length: int, phoneme_list: Tensor, speaker_id: Optional[Tensor]):
        phoneme_list = to_tensor(phoneme_list, device=device)
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
            speaker_id = speaker_id.reshape((1,)).to(torch.int64)
        return yukarin_s_forwarder(phoneme_list, speaker_id).cpu().numpy()

    return _dispatcher
