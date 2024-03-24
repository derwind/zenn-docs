---
title: "Qiskit で遊んでみる (24) — Qiskit Runtime local testing mode を使ってみる"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python"]
published: false
---

# 目的

[Qiskit Runtime local testing mode](https://docs.quantum.ibm.com/verify/local-testing-mode) に

> Local testing mode (available with `qiskit-ibm-runtime` 0.22.0 or later) can be used to help develop and test programs before fine-tuning them and sending them to real quantum hardware.

とあるのでこれを試してみたい。ざっくりとは、今までクラウド上で実機に対してジョブを投入する時の書き方がローカルでもできるようになって、バックエンドの指定をフェイクバックエンドやシミュレータから実機に切り替えるだけでコードを書き換えずにシミュレータ/実機で実験ができる、というものに見える。

# やってみた

## Qiskit のインストール

`qiskit-ibm-runtime` 0.22.0 が使えたら何でも良いのだが、とりあえずバージョンを指定してインストールしてみる。

```sh
%%bash

pip install -U "qiskit==1.0.2" "qiskit[visualization]==1.0.2" "qiskit-aer==0.13.3"
pip install -U qiskit-ibm-runtime=="0.22.0"
```

## 必要なモジュールの import

Example を参考に実装。比較用に、フェイクバックエンドも幾つかインポート。
[Migrate to the Qiskit Runtime V2 primitives](https://docs.quantum.ibm.com/api/migration-guides/v2-primitives) にあるように `Qiskit Runtime V2 primitives` もリリースされているので、これも使う。軽く見た感じではエラー緩和の指定の仕方などに変更があった。`SamplerV2` では実機側が対応している場合に[ダイナミックデカップリング](https://www.qcrjp.com/post/ibm433)が使えるようだが、今回はローカルでやるので使えない。

```python
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import (
    Session,
    SamplerV2 as Sampler,
    QiskitRuntimeService,
)
from qiskit_ibm_runtime.fake_provider import (
    FakeManilaV2,
    FakeSherbrooke,
    FakeTorino,
)
```

## 実験

Example のままに Bell 状態のサンプリング。

```python
# Bell Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# 事前にアカウント情報をセーブしてあることを想定
service = QiskitRuntimeService()

use_real = True
if use_real:
    real_backend = service.backend("ibm_brisbane")  # 実機
    backend = AerSimulator.from_backend(real_backend)  # の情報をシミュレータにロード
else:
    # backend = FakeManilaV2()  # 量子ビット数 5 のフェイクバックエンド
    backend = FakeTorino()  # 量子ビット数 133 のフェイクバックエンド
    # backend = AerSimulator()  # ノイズのないシミュレータ
 
# Run the sampler job locally using AerSimulator.
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_qc = pm.run(qc)
with Session(backend=backend) as session:
    sampler = Sampler(session=session)
    result = sampler.run([isa_qc]).result()
```

## 結果のチェック

```python
pub_result = result[0]
plot_histogram(pub_result.data.meas.get_counts())
```

で結果を見ることができる。

**ノイズのないシミュレータの場合**

勿論 $\frac{1}{\sqrt{2}}(\ket{00} + \ket{11})$ がとれる。

![](/images/dwd-qiskit24/001.png)

**ibm_brisbane (をシミュレータからロードした) の場合**

少しノイズが入った結果になる。`backend = real_backend` とすると、実機にジョブが飛んでいく。

![](/images/dwd-qiskit24/002.png)

## 回路の可視化

**パスマネージャを通していない回路**

そのまま。

```python
qc.draw("mpl", style="clifford")
```

![](/images/dwd-qiskit24/003.png)

**パスマネージャを通した回路**

パスマネージャが正直どういったものか [Qiskit v0.45 is here!](https://medium.com/qiskit/qiskit-v0-45-is-here-69e861fbfc88) を読んでも分からないが、example 同様に [Preset Passmanagers](https://docs.quantum.ibm.com/api/qiskit/transpiler_preset) の `generate_preset_pass_manager` をそのまま使えば良いのではないだろうか。

ということでトランスパイルされた回路が得られる。

```python
isa_qc.draw("mpl", style="clifford")
```

![](/images/dwd-qiskit24/004.png)

## ibm_brisbane のレイアウト

最後に `ibm_brisbane` のレイアウトを見て終わろう。今回は右上の 2 個の量子ビットだけ使った感じになる。

```python
real_backend.coupling_map.draw()
```

![](/images/dwd-qiskit24/005.png)

# まとめ

Qiskit のアップデートが早いので少しするとすぐに今までと違う書き方が登場している。たまにアップデートしないとよく分からなくなる。

今回、クラウド上と同じ書き方がローカルでできるようになったので、それを試してみた。
