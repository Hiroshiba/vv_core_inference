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

    @torch.no_grad()
    def forward(self, phoneme_list: Tensor, speaker_id: Tensor):
        output = self.predictor(
            phoneme_list=phoneme_list.unsqueeze(0), speaker_id=speaker_id
        )[0]
        return output


def make_yukarin_s_forwarder(yukarin_s_model_dir: Path, device, convert=False):
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
    def _dispatcher(length: int, phoneme_list: Tensor, speaker_id: Optional[Tensor]):
        phoneme_list = to_tensor(phoneme_list, device=device)
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)
            speaker_id = speaker_id.reshape((1,)).to(torch.int64)
        if convert:
            torch.onnx.export(
                yukarin_s_forwarder,
                (phoneme_list, speaker_id),
                yukarin_s_model_dir.joinpath("yukarin_s.onnx"),
                opset_version=OPSET,
                do_constant_folding=True,  # execute constant folding for optimization
                input_names=["phoneme_list", "speaker_id"],
                output_names=["phoneme_length"],
                dynamic_axes={
                    "phoneme_list": {0: "sequence"},
                    "phoneme_length" : {0: "sequence"}})
            print("yukarin_s has been converted to ONNX") 
        return yukarin_s_forwarder(phoneme_list, speaker_id).cpu().numpy()

    return _dispatcher
