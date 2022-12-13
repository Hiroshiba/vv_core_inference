"""
Pytorchモデルを読み込んでONNXモデルに変換する。
モデルを複数指定した場合は、それらを並列に接続し入力によって実行パスが切り替わるモデルが作成される。
例えば6話者のmodel1と10話者のmodel2が接続された場合は、入力speaker_idが0-5の時はmodel1が、6-15の時はmodel2が動作するONNXモデルが生成される。
各モデルに含まれている話者数はconfigファイルから推定される。
複数指定可能なモデルはduration, intonation, spectrogram推定。
vocoderは並列化することができず、全ての入力パターンに対してモデルが共有される。
"""
import argparse
import logging
from pathlib import Path
from typing import List

import torch
import onnx
import onnx.helper
import onnx.compose
import yaml

from vv_core_inference.utility import to_tensor, OPSET
from vv_core_inference.make_decode_forwarder import make_decode_forwarder
from vv_core_inference.make_yukarin_s_forwarder import make_yukarin_s_forwarder
from vv_core_inference.make_yukarin_sa_forwarder import make_yukarin_sa_forwarder
from vv_core_inference.forwarder import Forwarder


def convert_duration(model_dir: Path, device: str, offset: int, working_dir: Path, sample_input):
    """duration推定モデル(yukarin_s)のONNX変換"""
    from vv_core_inference.make_yukarin_s_forwarder import WrapperYukarinS
    from yukarin_s.config import Config
    from yukarin_s.network.predictor import create_predictor

    logger = logging.getLogger("duration")
    with model_dir.joinpath("config.yaml").open() as f:
        model_config = Config.from_dict(yaml.safe_load(f))
    size = model_config.network.speaker_size
    
    predictor = create_predictor(model_config.network)
    state_dict = torch.load(
        model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    logger.info("duration model is loaded!")
    logger.info("speaker size: %d" % size)

    phoneme_list = to_tensor(sample_input["phoneme_list"], device=device)
    speaker_id = to_tensor(sample_input["speaker_id"], device=device).reshape((1,)).long()

    forwarder = WrapperYukarinS(predictor)
    outpath = working_dir.joinpath(f"duration-{offset:03d}.onnx")
    torch.onnx.export(
        forwarder,
        (phoneme_list, speaker_id),
        outpath,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["phoneme_list", "speaker_id"],
        output_names=["phoneme_length"],
        dynamic_axes={
            "phoneme_list": {0: "sequence"},
            "phoneme_length" : {0: "sequence"}}
    )
    return outpath, size


def convert_intonation(model_dir: Path, device: str, offset: int, working_dir: Path, sample_input):
    """intonation推定モデル(yukarin_sa)のONNX変換"""
    from vv_core_inference.make_yukarin_sa_forwarder import WrapperYukarinSa
    from yukarin_sa.config import Config
    from yukarin_sa.network.predictor import create_predictor
    from vv_core_inference.utility import remove_weight_norm

    logger = logging.getLogger("intonation")
    with model_dir.joinpath("config.yaml").open() as f:
        model_config = Config.from_dict(yaml.safe_load(f))
    size = model_config.network.speaker_size

    predictor = create_predictor(model_config.network)
    state_dict = torch.load(
        model_dir.joinpath("model.pth"), map_location=device
    )
    predictor.load_state_dict(state_dict)
    predictor.eval().to(device)
    predictor.apply(remove_weight_norm)
    logger.info("intonation model is loaded!")
    logger.info("speaker size: %d" % size)
    wrapper = WrapperYukarinSa(predictor)

    args = (
        to_tensor(sample_input["length"], device=device).long(),
        to_tensor(sample_input["vowel_phoneme_list"], device=device),
        to_tensor(sample_input["consonant_phoneme_list"], device=device),
        to_tensor(sample_input["start_accent_list"], device=device),
        to_tensor(sample_input["end_accent_list"], device=device),
        to_tensor(sample_input["start_accent_phrase_list"], device=device),
        to_tensor(sample_input["end_accent_phrase_list"], device=device),
        to_tensor(sample_input["speaker_id"], device=device).reshape((1,)).long()
    )

    output = wrapper(*args)
    outpath = working_dir.joinpath(f"intonation-{offset:03d}.onnx")
    torch.onnx.export(
        torch.jit.script(wrapper),
        args,
        outpath,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=[
            "length",
            "vowel_phoneme_list",
            "consonant_phoneme_list",
            "start_accent_list",
            "end_accent_list",
            "start_accent_phrase_list",
            "end_accent_phrase_list",
            "speaker_id",
        ],
        example_outputs=output,
        output_names=["f0_list"],
        dynamic_axes={
            "vowel_phoneme_list": {0: "length"},
            "consonant_phoneme_list": {0: "length"},
            "start_accent_list": {0: "length"},
            "end_accent_list": {0: "length"},
            "start_accent_phrase_list": {0: "length"},
            "end_accent_phrase_list": {0: "length"},
            "f0_list": {0: "length"}},
    )
    return outpath, size

def convert_spectrogram(model_dir: Path, device: str, offset: int, working_dir: Path, sample_input):
    """spectrogram推定モデル(decodeの前半, yukarin_sosoa)のONNX変換"""
    from vv_core_inference.make_yukarin_sosoa_forwarder import make_yukarin_sosoa_wrapper

    logger = logging.getLogger("spectrogram")

    wrapper = make_yukarin_sosoa_wrapper(model_dir, device)
    size = wrapper.speaker_embedder.num_embeddings
    logger.info("spectrogram model is loaded!")
    logger.info("speaker size: %d" % size)
    args = (
        to_tensor(sample_input["f0"], device=device),
        to_tensor(sample_input["phoneme"], device=device),
        to_tensor(sample_input["speaker_id"], device=device)
    )
    outpath = working_dir.joinpath(f"spectrogram-{offset:03d}.onnx")
    torch.onnx.export(
        wrapper,
        args,
        outpath,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=[
            "f0",
            "phoneme",
            "speaker_id"
        ],
        output_names=["spec"],
        dynamic_axes={
            "f0": {0: "length"},
            "phoneme": {0: "length"},
            "spec": {0: "row", 1: "col"}
        }
    )
    return outpath, size

def convert_vocoder(model_dir: Path, device: str, working_dir: Path, sample_input):
    """vocoder(decodeの後半, hifi_gan)のONNX変換"""
    from vv_core_inference.make_decode_forwarder import make_hifigan_wrapper

    logger = logging.getLogger("vocoder")

    wrapper = make_hifigan_wrapper(model_dir, device)
    logger.info("vocoder model is loaded!")
    args = (to_tensor(sample_input, device=device),)
    outpath = working_dir.joinpath(f"vocoder.onnx")
    torch.onnx.export(
        wrapper,
        args,
        outpath,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["spec"],
        output_names=["wave"],
        dynamic_axes={"spec": {0: "row", 1: "col"}}
    )
    return outpath


def get_sample_inputs(
    yukarin_s_model_dir: Path,
    yukarin_sa_model_dir: Path,
    yukarin_sosoa_model_dir: Path,
    hifigan_model_dir: Path,
    sample_text: str,
    sample_speaker: int,
    device: str
):
    """ONNX変換に必要な中間表現を収集する"""
    # yukarin_s
    yukarin_s_forwarder = make_yukarin_s_forwarder(
        yukarin_s_model_dir=yukarin_s_model_dir, device=device
    )

    # yukarin_sa
    yukarin_sa_forwarder = make_yukarin_sa_forwarder(
        yukarin_sa_model_dir=yukarin_sa_model_dir, device=device
    )

    # decoder
    decode_forwarder = make_decode_forwarder(
        yukarin_sosoa_model_dir=yukarin_sosoa_model_dir,
        hifigan_model_dir=hifigan_model_dir,
        device=device
    )

    forwarder = Forwarder(
        yukarin_s_forwarder=yukarin_s_forwarder,
        yukarin_sa_forwarder=yukarin_sa_forwarder,
        decode_forwarder=decode_forwarder,
    )

    _wave, intermediates = forwarder.forward(
        text = sample_text,
        speaker_id = sample_speaker,
        f0_speaker_id = sample_speaker,
        return_intermediate_results = True
    )

    return intermediates


def concat(onnx_list: List[Path], offsets: List[int]):
    """
    複数のONNXモデル model[:] を並列に接続し、
    入力されるspeaker_idがどのoffsetに含まれるかによって実行パスが変わるONNXモデルを再生成する。
    ONNX上ではモデルiのノードXはm{i}.Xにリネームされ、上流のIFノードによって実行パスが変化する。

    goal is outputing an onnx equivalent to:
    
    def merged_model(speaker_id, **kwargs):
        if speaker_id < offset[1]:
            y = model[0](speaker_id - offset[0], **kwargs)
        else:
            if speaker_id < offset[2]:
                y = model[1](speaker_id - offset[1], **kwargs)
            else:
                if ...
        return y
    """
    logger = logging.getLogger("concat")
    models = [onnx.load(path) for path in onnx_list]
    logger.info("loaded models:")
    for path in onnx_list:
        logger.info(f"* {path}")

    opset = models[0].opset_import[0].version
    logger.info("opset: %d" % opset)

    # io name/nodes to be shared
    input_names = []
    input_nodes = []
    output_names = []
    output_nodes = []
    logger.info("input names:")
    for node in models[0].graph.input:
        logger.info(f"* {node.name}")
        input_names.append(node.name)
        input_nodes.append(node)

    assert "speaker_id" in input_names
    input_names.remove("speaker_id") # speaker_id input is not shared

    logger.info("output names:")
    for node in models[0].graph.output:
        logger.info(f"* {node.name}")
        output_names.append(node.name)
        output_nodes.append(node)

    def rename(graph, prefix: str, freeze_names: List[str]):
        for node in graph.node:
            for i, n in enumerate(node.input):
                if n not in freeze_names and n != "":
                    node.input[i] = prefix + n
            node.name = prefix + node.name
            if node.op_type == "Loop":
                for attr in node.attribute:
                    if attr.name == "body":
                        for subnode in attr.g.input:
                            subnode.name = prefix + subnode.name
                        for subnode in attr.g.output:
                            subnode.name = prefix + subnode.name
                        rename(attr.g, prefix, freeze_names)
            for i, n in enumerate(node.output):
                if n not in freeze_names and n != "":
                    node.output[i] = prefix + n
        for init in graph.initializer:
            init.name = prefix + init.name

    offset_consts = []
    for i, offset in enumerate(offsets):
        offset_consts.append(onnx.helper.make_tensor(
            name=f"offset_{i}",
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[offset]))

    for i, model in enumerate(models):
        prefix = f"m{i}."
        rename(model.graph, prefix, input_names + output_names) # all submodules share inputs and outputs.
 
    select_conds = []
    shifted_speaker_ids = []
    for i, model in enumerate(models):
        prefix = f"m{i}."
        speaker_offset = offset_consts[i]
        speaker_end = offset_consts[i+1]
        select_conds.append(onnx.helper.make_node(
            "Less",
            inputs=["speaker_id", speaker_end.name],
            outputs=[f"select_cond_{i}"]
        ))
        shifted_speaker_ids.append(onnx.helper.make_node(
            "Sub",
            inputs=["speaker_id", speaker_offset.name],
            outputs=[prefix + "speaker_id"]
        ))

    branches = []
    for i, m in enumerate(models):
        branches.append(onnx.helper.make_graph(
            nodes=[shifted_speaker_ids[i]] + list(m.graph.node),
            name=f"branch_m{i}",
            inputs=[],
            outputs=output_nodes,
            initializer=list(m.graph.initializer) + [offset_consts[i]]
        ))
    
    whole_graph = branches[-1]
    for i in range(len(branches)-2, -1, -1):
        if_node = onnx.helper.make_node(
            "If",
            inputs=[f"select_cond_{i}"],
            outputs=output_names,
            then_branch=branches[i],
            else_branch=whole_graph
        )
        # logger.info(f"else: {whole_graph.node}")
        whole_graph = onnx.helper.make_graph(
            nodes=[select_conds[i], if_node],
            name=f"branch{i}",
            inputs=[],
            outputs=output_nodes,
            initializer=[offset_consts[i+1]]
        )

    whole_graph = onnx.helper.make_graph(
        nodes=list(whole_graph.node),
        name="whole_model",
        inputs=input_nodes,
        outputs=output_nodes,
        initializer=list(whole_graph.initializer)
    )
    whole_model = onnx.helper.make_model(whole_graph, opset_imports=[onnx.helper.make_operatorsetid("", opset)])

    output_onnx_path = onnx_list[0].parent / (onnx_list[0].stem[:-4] + ".onnx")
    onnx.checker.check_model(whole_model)
    onnx.save(whole_model, output_onnx_path)
    logger.info(f"saved {output_onnx_path}")
    return output_onnx_path


def fuse(onnx1: Path, onnx2: Path):
    """ふたつのONNXモデルを直列に接続する。spectrogramとvocoderを接続するために利用する。"""
    # you can use onnx.compose.merge_models
    # https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#onnx-compose
    logger = logging.getLogger("fuse")
    model1 = onnx.load(onnx1)
    model2 = onnx.load(onnx2)
    opset = model1.opset_import[0].version
    logger.info("opset: %d" % opset)

    merged_graph = onnx.compose.merge_graphs(model1.graph, model2.graph, [("spec", "spec")])
    merged = onnx.helper.make_model(merged_graph, opset_imports=[onnx.helper.make_operatorsetid("", opset)])
    logger.info(f"fused {onnx1} and {onnx2}")
    output_onnx_path = onnx1.parent / "decode.onnx"
    onnx.checker.check_model(merged)
    onnx.save(merged, output_onnx_path)
    logger.info(f"saved {output_onnx_path}")

def run(
    yukarin_s_model_dir: List[Path],
    yukarin_sa_model_dir: List[Path],
    yukarin_sosoa_model_dir: List[Path],
    hifigan_model_dir: Path,
    working_dir: Path,
    text: str,
    speaker_id: int,
    use_gpu: bool,
):
    logger = logging.getLogger()
    device = "cuda" if use_gpu else "cpu"

    model_size = len(yukarin_s_model_dir)
    assert model_size == len(yukarin_sa_model_dir)
    assert model_size == len(yukarin_sosoa_model_dir)

    assert working_dir.exists() and working_dir.is_dir()

    logger.info("device: %s" % device)
    logger.info("onnx OPSET is %d." % OPSET)
    logger.info("vocoder model dir is %s" % str(hifigan_model_dir))
    logger.info("working on %s" % str(working_dir))

    logger.info("--- creating onnx models ---")

    duration_onnx_list = []
    intonation_onnx_list = []
    spectrogram_onnx_list = []

    offsets = [0]
    for idx, s_dir, sa_dir, sosoa_dir in zip(
        range(model_size), yukarin_s_model_dir, yukarin_sa_model_dir, yukarin_sosoa_model_dir
    ):
        logger.info(f"[{idx+1}/{model_size}] Start converting models")
        logger.info(f"duration: {s_dir}")
        logger.info(f"intonation: {sa_dir}")
        logger.info(f"spectrogram: {sosoa_dir}")

        sample_inputs = get_sample_inputs(
            s_dir, sa_dir, sosoa_dir, hifigan_model_dir, text, speaker_id, device
        )

        logger.info("duration model START")
        duration_onnx, duration_size = convert_duration(s_dir, device, offsets[-1], working_dir, sample_inputs["yukarin_s_input"])
        duration_onnx_list.append(duration_onnx)
        logger.info("duration model DONE")

        logger.info("intonation model START")
        intonation_onnx, intonation_size = convert_intonation(sa_dir, device, offsets[-1], working_dir, sample_inputs["yukarin_sa_input"])
        intonation_onnx_list.append(intonation_onnx)
        logger.info("intonation model DONE")

        logger.info("spec model START")
        spec_onnx, spec_size = convert_spectrogram(sosoa_dir, device, offsets[-1], working_dir, sample_inputs["yukarin_sosoa_input"])
        spectrogram_onnx_list.append(spec_onnx)
        logger.info("spec model DONE")

        assert duration_size == intonation_size
        assert duration_size == spec_size
        offsets.append(offsets[-1] + duration_size)
    
    logger.info(f"collected offsets: {offsets}")
    logger.info(f"[###] Start converting vocoder model: {hifigan_model_dir}")
    vocoder_onnx = convert_vocoder(hifigan_model_dir, device, working_dir, sample_inputs["hifigan_input"])
    logger.info("vocoder model DONE")

    logger.info("--- concatination ---")
    duration_merged_onnx = concat(duration_onnx_list, offsets)
    intonation_merged_onnx = concat(intonation_onnx_list, offsets)
    spectrogram_merged_onnx = concat(spectrogram_onnx_list, offsets)
    decoder_onnx = fuse(spectrogram_merged_onnx, vocoder_onnx)
    logger.info("--- DONE! ---")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yukarin_s_model_dir", type=Path, nargs="*", default=[Path("model/yukarin_s")]
    )
    parser.add_argument(
        "--yukarin_sa_model_dir", type=Path, nargs="*", default=[Path("model/yukarin_sa")]
    )
    parser.add_argument(
        "--yukarin_sosoa_model_dir", type=Path, nargs="*", default=[Path("model/yukarin_sosoa")]
    )
    parser.add_argument("--hifigan_model_dir", type=Path, default=Path("model/hifigan"))
    parser.add_argument(
        "--working_dir", type=Path, default="model"
    )
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--text", default="こんにちは、どうでしょう")
    parser.add_argument("--speaker_id", type=int, default=5)
    run(**vars(parser.parse_args()))
