# vv_core_inference

VOICEVOX のコア内で用いられているディープラーニングモデルの推論コード。

サンプルとして配布しているモデルは実際の VOICEVOX のものではなく、ほとんどノイズと変わらない音が生成されます。
含まれている config の値（層の数など）は仮の値で、VOICEVOX で使用されるモデルとは異なることがあります。

## 公開している意図

VOICEVOXのコアの軽量版を作りたいためです。

VOICEVOXのディスク容量の軽量化をしたいのですが、時間が取れずにいます。ディスク容量が大きいのは、コア内のディープラーニングモデルの推論にlibtorchを用いているためです。そこで該当箇所のpythonコードを公開し、libtorchの代替となる軽量な手法の議論や提案を受けられるようにしました。

技術的なこと以外の要件としては、諸事情により「第三者が簡単にモデルの内容を得られない」ようにする必要があります。pythonコードは容易にコードを推測できるので使えません。とりあえず推論コードが全部C++であれば大丈夫です。（こちら側で暗号化などを足します。）

## 環境構築

Python 3.7.2 で開発しました。 3.7 台なら動くと思います。

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

## モデルをonnxに変換
* `python run.py --yukarin_s_model_dir "model/yukarin_s" --yukarin_sa_model_dir "model/yukarin_sa" --yukarin_sosoa_model_dir "model/yukarin_sosoa" --hifigan_model_dir "model/hifigan"  --speaker_ids 5  --method=convert` でonnxへの変換が可能。modelフォルダ内のyukarin_s, yukarin_sa, yukarin_sosoaフォルダにonnxが保存される。
  - `speaker_ids`オプションに指定する数値は自由。どの数値を指定しても生成されるonnxモデルは全ての`speaker_id`に対応しており、値を変えて実行しなおしたり、複数のidを指定したりする必要は無い。
  - yukarin_sosoaフォルダにはhifi_ganと合わせた`decode.onnx`が保存される

* onnxで実行したい場合は`--method=onnx`とする； `python run.py --yukarin_s_model_dir "model/yukarin_s" --yukarin_sa_model_dir "model/yukarin_sa" --yukarin_sosoa_model_dir "model/yukarin_sosoa" --hifigan_model_dir "model/hifigan"  --speaker_ids 5  --method=onnx`
  - `speaker_ids`に複数の数値を指定すれば、通常実行と同様に各話者の音声が保存される。

## ファイル構造

- `run.py` ･･･ エントリーポイント
- `vv_core_inference` ･･･ いろいろな処理
  - `forwarder.py`
    - VOICEVOX と同じインターフェースと処理。
    - この`Forwarder`クラスを一切変更することなく、3 つの`forwarder`を与えられると完璧
  - `make_yukarin_s_forwarder.py`
    - 音素ごとの長さを求めるモデル`yukarin_s`用の`forwarder`を作る
  - `make_yukarin_sa_forwarder.py`
    - モーラごとの音高を求めるモデル`yukarin_sa`用の`forwarder`を作る
  - `make_yukarin_sosoa_forwarder.py`
    - `make_decode_forwarder`に必要な`yukarin_sosoa`用の`forwarder`を作る
  - `make_decode_forwarder.py`
    - 音声波形生成用の`forwarder`を作る
  - `onnx_yukarin_s_forwarder.py`
    - onnxruntimeで動作する`yukarin_s`用の`forwarder`を作る
  - `onnx_yukarin_sa_forwarder.py`
    - onnxruntimeで動作する`yukarin_sa`用の`forwarder`を作る
  - `onnx_decode_forwarder.py`
    - onnxruntimeで動作する音声波形生成用の`forwarder`を作る
    - `yukarin_sosoa`も内部に組み込まれている
  - `acoustic_feature_extractor.py`
    - 音素情報やリサンプリング手法などが入っている。ディープラーニングとは関係ない。
  - `full_context_label.py`
    - フルコンテキストラベルの処理が入っている。ディープラーニングとは関係ない。
  - `utility.py`
    - 便利関数が多少ある
