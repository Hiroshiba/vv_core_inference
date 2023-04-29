set -eux -o pipefail
shopt -s inherit_errexit

working_dir=working/
rm -rf $working_dir
mkdir -p $working_dir

python run.py \
    --yukarin_s_model_dir "model/yukarin_s" \
    --yukarin_sa_model_dir "model/yukarin_sa" \
    --yukarin_sosf_model_dir "model/yukarin_sosf" \
    --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
    --hifigan_model_dir "model/hnhifigan" \
    --speaker_ids 5 9

python convert.py \
    --yukarin_s_model_dir "model/yukarin_s" \
    --yukarin_sa_model_dir "model/yukarin_sa" \
    --yukarin_sosf_model_dir "model/yukarin_sosf" \
    --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
    --hifigan_model_dir "model/hnhifigan" \
    --working_dir "$working_dir"

python run.py \
    --yukarin_s_model_dir "$working_dir" \
    --yukarin_sa_model_dir "$working_dir" \
    --yukarin_sosf_model_dir "$working_dir" \
    --yukarin_sosoa_model_dir "$working_dir" \
    --hifigan_model_dir "$working_dir" \
    --method onnx \
    --speaker_ids 5 9
