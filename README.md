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
wget https://github.com/Hiroshiba/vv_core_inference/releases/download/0.0.1/model.zip
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

## C++のコードを python 側に持ってくる場合

Cyhton が便利です。

1. [VOICEVOX CORE](https://github.com/Hiroshiba/voicevox_core/tree/f4844efc65b1a4875442091955af84f671e16887)にある[core.h](https://github.com/Hiroshiba/voicevox_core/blob/f4844efc65b1a4875442091955af84f671e16887/core.h)をダウンロード
2. core.h に合うように C++ コードを書く
3. C++ コードから動的ライブラリをビルド
4. あとは[README.md](https://github.com/Hiroshiba/voicevox_core/tree/f4844efc65b1a4875442091955af84f671e16887#%E3%82%BD%E3%83%BC%E3%82%B9%E3%82%B3%E3%83%BC%E3%83%89%E3%81%8B%E3%82%89%E5%AE%9F%E8%A1%8C)にあるように`python setup.py install`などを実行
5. import して[このように](https://github.com/Hiroshiba/voicevox_core/blob/f4844efc65b1a4875442091955af84f671e16887/example/python/run.py#L21-L25)つなぎこむ

## ファイル構造

- `run.py` ･･･ エントリーポイント
- `vv_core_inference` ･･･ いろいろな処理
  - `forwarder.py`
    - VOICEVOX と同じインターフェースと処理。
    - このクラスを一切変更することなく、3 つの`forwarder`を与えられると完璧
  - `make_yukarin_s_forwarder.py`
    - 音素ごとの長さを求めるモデル`yukarin_s`用の`forwarder`を作る
  - `make_yukarin_sa_forwarder.py`
    - モーラごとの音高を求めるモデル`yukarin_sa`用の`forwarder`を作る
  - `make_yukarin_sosoa_forwarder.py`
    - `make_decode_forwarder`に必要な`yukarin_sosoa`用の`forwarder`を作る
  - `make_decode_forwarder.py`
    - 音声波形生成用の`forwarder`を作る
  - `acoustic_feature_extractor.py`
    - 音素情報やリサンプリング手法などが入っている。ディープラーニングとは関係ない。
  - `full_context_label.py`
    - フルコンテキストラベルの処理が入っている。ディープラーニングとは関係ない。
  - `utility.py`
    - 便利関数が多少ある
