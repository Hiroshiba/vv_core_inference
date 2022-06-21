import json
from pathlib import Path
from typing import Optional

import numpy
import torch
from hifi_gan.models import Generator as HifiGanPredictor
from torch import nn

from vv_core_inference.make_yukarin_sosoa_forwarder import make_yukarin_sosoa_wrapper
from vv_core_inference.utility import to_tensor, OPSET
from vv_core_inference.surgeon import surgeon


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class WrapperHifiGan(nn.Module):
    def __init__(
        self,
        hifi_gan_forwarder: nn.Module,
    ):
        super().__init__()
        self.hifi_gan_forwarder = hifi_gan_forwarder
    
    @torch.no_grad()
    def forward(
        self,
        spec: torch.Tensor,
    ):
        # forward hifi gan
        x = spec.transpose(1, 0)
        wave = self.hifi_gan_forwarder(x.unsqueeze(0))[0, 0]
        return wave

def make_hifi_gan_wrapper(hifigan_model_dir: Path, device) -> nn.Module:
    config = AttrDict(json.load(hifigan_model_dir.joinpath("config.json").open()))
    predictor = HifiGanPredictor(config).to(device)
    checkpoint_dict = torch.load(
        hifigan_model_dir.joinpath("model.pth"),
        map_location=device,
    )
    predictor.load_state_dict(checkpoint_dict["generator"])
    predictor.eval()
    predictor.remove_weight_norm()
    print("hifi-gan loaded!")
    return WrapperHifiGan(predictor)


def make_decode_forwarder(
    yukarin_sosoa_model_dir: Path, hifigan_model_dir: Path, device, convert=False
):
    # yukarin_sosoa
    yukarin_sosoa_wrapper = make_yukarin_sosoa_wrapper(
        yukarin_sosoa_model_dir=yukarin_sosoa_model_dir, device=device
    )

    # hifi-gan
    hifi_gan_wrapper = make_hifi_gan_wrapper(
        hifigan_model_dir=hifigan_model_dir, device=device
    )

    def _dispatcher(
        length: int,
        phoneme_size: int,
        f0: numpy.ndarray,
        phoneme: numpy.ndarray,
        speaker_id: Optional[numpy.ndarray] = None,
    ):
        f0 = to_tensor(f0, device=device)
        phoneme = to_tensor(phoneme, device=device)
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id, device=device)

        spec = yukarin_sosoa_wrapper(
            f0=f0, phoneme=phoneme, speaker_id=speaker_id
        )
        wave = hifi_gan_wrapper(spec)

        if convert:
            torch.onnx.export(
                yukarin_sosoa_wrapper,
                (f0, phoneme, speaker_id),
                yukarin_sosoa_model_dir.joinpath("yukarin_sosoa.onnx"),
                opset_version=OPSET,
                do_constant_folding=True,
                input_names=["f0", "phoneme", "speaker_id"],
                output_names=["spec"],
                dynamic_axes={
                    "f0": {0: "length"},
                    "phoneme": {0: "length"},
                    "spec": {0: "length"}
                })
            print("decode/yukarin_sosoa has been converted to ONNX")
            torch.onnx.export(
                hifi_gan_wrapper,
                (spec,),
                hifigan_model_dir.joinpath("hifigan.onnx"),
                opset_version=OPSET,
                do_constant_folding=True,
                input_names=["spec"],
                output_names=["wave"],
                dynamic_axes={
                    "spec": {0: "length"},
                    "wave": {0: "outlength"}
                })
            fname = str(hifigan_model_dir.joinpath("hifigan.onnx"))
            surgeon(fname, fname)
            print("decode/hifigan has been converted to ONNX")
        return wave.cpu().numpy()

    return _dispatcher