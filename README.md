# vv_core_inference

VOICEVOX のコア内で用いられているディープラーニングモデルの推論コード。VOICEVOXコア用のonnxモデルを制作できる。

サンプルとして配布しているモデルは実際の VOICEVOX のものではなく、ほとんどノイズと変わらない音が生成されます。
含まれている config の値（層の数など）は仮の値で、VOICEVOX で使用されるモデルとは異なることがあります。

## 公開している意図

VOICEVOXコアでの音声合成をより高速・軽量にするための手法の議論や提案を受けられるようにするためです。

VOICEVOXコアはこのリポジトリで作ったonnxモデルを用いて計算処理（推論）が行われています。
onnxモデルをそのまま改良するのはかなり専門的な知識が必要なので、より多くの方に馴染みのあるpytorchのモデルとコードを公開しています。

## 環境構築

Python 3.10.14 で開発しました。 3.10 台なら動くと思います。

```bash
# ５分くらいかかります
pip install -r requirements.txt
```

## pytorchモデルのダウンロード

```bash
wget https://github.com/Hiroshiba/vv_core_inference/releases/download/0.0.1/model.zip
unzip model.zip
```

## 実行

```bash
# 生成される音声はほぼノイズで、かろうじて母音がわかる程度だと思います
python run.py \
  --yukarin_s_model_dir "model/yukarin_s" \
  --yukarin_sa_model_dir "model/yukarin_sa" \
  --yukarin_sosoa_model_dir "model/yukarin_sosoa" \
  --hifigan_model_dir "model/hifigan" \
  --speaker_ids 5 \
  --texts "おはようございます、こんにちは、こんばんは、どうでしょうか"
```

## モデルをonnxに変換
* `python convert.py --yukarin_s_model_dir "model/yukarin_s" --yukarin_sa_model_dir "model/yukarin_sa" --yukarin_sosoa_model_dir "model/yukarin_sosoa" --hifigan_model_dir "model/hifigan"` でonnxへの変換が可能。modelフォルダ内のyukarin_s, yukarin_sa, yukarin_sosoaフォルダにonnxが保存される。
  - `speaker_ids`オプションに指定する数値は自由。どの数値を指定しても生成されるonnxモデルは全ての`speaker_id`に対応しており、値を変えて実行しなおしたり、複数のidを指定したりする必要は無い。
  - yukarin_sosoaフォルダにはhifi_ganと合わせた`decode.onnx`が保存される
  - yukarin_sosfはオプショナルで、追加する場合は`--yukarin_sosf_model_dir "model/yukarin_sosf"`などを指定する

* onnxで実行したい場合は`run.py`を`--method=onnx`で実行する； `python run.py --yukarin_s_model_dir "model" --yukarin_sa_model_dir "model" --yukarin_sosoa_model_dir "model" --hifigan_model_dir "model"  --speaker_ids 5  --method=onnx`
  - `speaker_ids`に複数の数値を指定すれば、通常実行と同様に各話者の音声が保存される。

## ファイル構造

- `run.py` ･･･ 音声合成のためのエントリーポイント
- `convert.py` ･･･ モデル変換のためのエントリーポイント
- `vv_core_inference` ･･･ いろいろな処理
  - `forwarder.py`
    - VOICEVOX と同じインターフェースと処理。
    - この`Forwarder`クラスを一切変更することなく、3 つの`forwarder`を与えられると完璧
  - `make_yukarin_s_forwarder.py`
    - 音素ごとの長さを求めるモデル`yukarin_s`用の`forwarder`を作る
  - `make_yukarin_sa_forwarder.py`
    - モーラごとの音高を求めるモデル`yukarin_sa`用の`forwarder`を作る
  - `make_yukarin_sosf_forwarder.py`
    - モーラごとの音高からフレームごとの音高を求めるモデル`yukarin_sosf`用の`forwarder`を作る
  - `make_yukarin_sosoa_forwarder.py`
    - `make_decode_forwarder`に必要な`yukarin_sosoa`用の`forwarder`を作る
  - `make_decode_forwarder.py`
    - 音声波形生成用の`forwarder`を作る
  - `onnx_yukarin_s_forwarder.py`
    - onnxruntimeで動作する`yukarin_s`用の`forwarder`を作る
  - `onnx_yukarin_sa_forwarder.py`
    - onnxruntimeで動作する`yukarin_sa`用の`forwarder`を作る
  - `onnx_yukarin_sosf_forwarder.py`
    - onnxruntimeで動作する`yukarin_sosf`用の`forwarder`を作る
  - `onnx_decode_forwarder.py`
    - onnxruntimeで動作する音声波形生成用の`forwarder`を作る
    - `yukarin_sosoa`も内部に組み込まれている
  - `acoustic_feature_extractor.py`
    - 音素情報やリサンプリング手法などが入っている。ディープラーニングとは関係ない。
  - `full_context_label.py`
    - フルコンテキストラベルの処理が入っている。ディープラーニングとは関係ない。
  - `utility.py`
    - 便利関数が多少ある

## 自分で学習したモデルの onnx を作りたい場合

VOICEVOX をビルドするには以下の 3 つの onnx が必要です。
（predict_contourはオプショナルです。）

- predict_duration.onnx
  - 入力
    - phoneme_list
      - shape: [sequence]
      - dtype: int
      - 値は音素 id
    - speaker_id
      - shape: [1]
      - dtype: int
  - 出力
    - phoneme_length
      - shape: [sequence]
      - dtype: float
      - 値は秒単位の継続長
- predict_intonation.onnx
  - 入力
    - length
      - shape: int
      - 値は音素 id
    - vowel_phoneme_list
      - shape: [length]
      - dtype: int
      - 値は音素 id
    - consonant_phoneme_list
      - shape: [length]
      - dtype: int
      - 値は音素 id
    - start_accent_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - end_accent_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - start_accent_phrase_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - end_accent_phrase_list
      - shape: [length]
      - dtype: int
      - 値は 0 か 1
    - speaker_id
      - shape: [1]
      - dtype: int
  - 出力
    - f0_list
      - shape: [sequence]
      - dtype: float
- predict_contour.onnx
  - 入力
    - f0_discrete
      - shape: [length, 1]
      - dtype: float
      - 値は離散値だった f0 を長さ分だけ repeat したもの
    - phoneme
      - shape: [length]
      - dtype: int
      - 値は音素 id
    - speaker_id
      - shape: [1]
      - dtype: int
  - 出力
    - f0_contour
      - shape: [length]
      - dtype: float
      - 値は連続値の f0
    - voiced
      - shape: [length]
      - dtype: bool
      - 値は True か False
- decode.onnx
  - 入力
    - f0
      - shape: [length, 1]
      - dtype: float
    - phoneme
      - shape: [length, vocab_size]
      - dtype: int
      - onehot で表現された音素
    - speaker_id
      - shape: [1]
      - dtype: int
  - 出力
    - wave
      - shape: [outlength]
      - dtype: float
      - 値は [-1.0, 1.0] の音声波形
      - サンプリング周波数は 24kHz

音素 id は辞書に依存します。また predict_duration.onnx や predict_intonation.onnx の出力はコアによって変換されて decode.onnx の入力になります。コアを変更しない場合は phoneme_length を元に f0 と phoneme が 93.75(=24k/256)Hz になるように変換されます。
