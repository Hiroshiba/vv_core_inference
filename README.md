# vv_core_inference

VOICEVOX のコア内で用いられているディープラーニングモデルの推論コード。

サンプルとして配布しているモデルは実際の VOICEVOX のものではなく、ほとんどノイズと変わらない音が生成されます。
含まれている config の値（層の数など）は仮の値で、VOICEVOX で使用されるモデルとは異なることがあります。

## 環境構築

Python 3.7.2 を使って開発しました。

```bash
# ５分くらいかかります
pip install -r requirements.txt
```

## モデルのダウンロード

```bash
wget https://github.com/Hiroshiba/voicevox/releases/download/0.0.1/model.zip
unzip model.zip
```

## 実行

```python
# 生成される音声はほぼノイズで、かろうじて母音がわかる程度だと思います
python run.py \
    --yukarin_s_model_dir "model/yukarin_s" \
    --yukarin_sa_model_dir "model/yukarin_sa" \
    --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
    --hifigan_model_dir "model/hifigan" \
    --speaker_ids 5 \
    --texts "おはようございます、こんにちは、こんばんは、どうでしょうか"
```
