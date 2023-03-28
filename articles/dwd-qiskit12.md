---
title: "Qiskit で遊んでみる (12) — Qiskit Aer にクラスを追加した話"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem"]
published: true
---

# 目的

[Qiskit で遊んでみる (11) — Qiskit Advocate Mentorship Program](/derwind/articles/dwd-qiskit11) で触れた Qiskit Advocate Mentorship Program Fall 2022 の活動の結果、[Qiskit Aer](https://github.com/Qiskit/qiskit-aer) に [Implement AerDensityMatrix](https://github.com/Qiskit/qiskit-aer/pull/1732) で `AerDensityMatrix` というクラスを実装し、無事 [0.12.0](https://github.com/Qiskit/qiskit-aer/releases/tag/0.12.0) で世の中にリリースされたので書き留めてみたい。

# Qiskit Aer とは

「現実的なノイズモデルを持つ高性能な量子コンピュータシミュレータを提供する」Qiskit のモジュールである。Qiskit は元々は _Terra_ (地)、_Aer_ (空気)、_Ignis_ (火)、_Aqua_ (水) の 4 つの主要コンポーネントから構成されており、それぞれ Qiskit のメインの土台、シミュレータ、雑音・エラー、アルゴリズムを担っていた。後者 2 つは細分化されここ数年で DEPRECATED になり、現在でもアクティブに開発が進んでいるのは Terra と Aer である。詳細については [Qiskit の要素](https://qiskit.org/documentation/locale/ja/the_elements.html) が参考になりそうである。

QAMP (Qiskit Advocate Mentorship Program) の活動で、この Aer のシミュレーション用クラスを拡充しようとする assignment があり、それに取り組んだ。

# 色々なシミュレータ

メインコンポーネント [Qiskit Terra](https://github.com/Qiskit/qiskit-terra) は基本的にはベーシックな実装のものが組み込まれており、[BasicAer: Python-based Simulators](https://qiskit.org/documentation/apidoc/providers_basicaer.html) に紹介のある一番よく使う？シミュレータ

- [BasicAer](https://github.com/Qiskit/qiskit-terra/blob/0.23.3/qiskit/providers/basicaer/__init__.py#L73)

や、[状態ベクトル](https://ja.wikipedia.org/wiki/%E9%87%8F%E5%AD%90%E7%8A%B6%E6%85%8B)および[密度行列](https://ja.wikipedia.org/wiki/%E5%AF%86%E5%BA%A6%E8%A1%8C%E5%88%97)に特化したシミュレータの

- [Statevector](https://github.com/Qiskit/qiskit-terra/blob/0.23.3/qiskit/quantum_info/states/statevector.py)
- [DensityMatrix](https://github.com/Qiskit/qiskit-terra/blob/0.23.3/qiskit/quantum_info/states/densitymatrix.py)

が存在する。この辺のクラスは大なり小なり [IBM Certified Associate Developer](https://www.ibm.com/training/certification/C0010300) 試験においては頻出のモジュールであり、基本的な機能について把握していることが求められる。

但し、実際の使用においては全体として中程度の速度であるという問題がある。理由としては、これらは CPython や [NumPy](https://numpy.org/) のみで実装されており、NumPy も高速とは言え汎用的なモジュールで量子計算に特化はしていないからということになる。

これに対し、高速シミュレーションを提供する Aer においては、Terra に対するオプションとして

- [AerSimulator](https://github.com/Qiskit/qiskit-aer/blob/0.12.0/qiskit_aer/backends/aer_simulator.py)
- [AerStatevector](https://github.com/Qiskit/qiskit-aer/blob/0.12.0/qiskit_aer/quantum_info/states/aer_statevector.py)
- [AerDensityMatrix](https://github.com/Qiskit/qiskit-aer/blob/0.12.0/qiskit_aer/quantum_info/states/aer_densitymatrix.py)

が対応して存在する。AerStatevector は Aer 0.11.0 で、AerDensityMatrix は今回 0.12.0 で導入された。これらは CPU 上でも Terra 版より高速に動作するが、更に GPU が利用できる環境では GPU 上で実行することで更にシミュレーションを加速することが可能となっている[^1]。

[^1]: GPU については別の記事にわけたい。

# AerStatevector の設計

課題に取り組むにあたって洗い出したシーケンスは以下のようなものであった。幾らか手が入ったので現在では多少異なっているかもしれないが、大きな流れはこのようになっているはずである。

![](/images/dwd-qiskit12/001.png)

基本的には、Python のガワで受け取った情報 (`QuantumCircuit`, `numpy.ndarray`) を [pybind11](https://github.com/pybind/pybind11) を経由して、C++ 層に渡し、その中で

- マルチスレッド
- SIMD 命令
- GPU 計算 (有効化されている場合)

を活用した効率的な計算が実行されることになる。

なお、QAMP の課題としてそこまでこなすと勿論とてつもない工数になることもあり、**課題としては範囲外であり今回も高速化実装は対応していない**。既に実装していただいた**高速化の処理を Python のガワにつないで Python クラスとして利用可能にする**部分が課題であった。要するに上記の AerStatevector のシーケンスで言うと、一番左のライフラインが実装範囲である。

# AerDensityMatrix の設計

**既に AerStatevector の実装は提供されていたので、基本的にはそれを分析して倣う**ということになるので、それほど難しくはない・・・というのが最初の見解であったが、実際にはソフトウェア実装である以上もろもろの問題はあった。

- 状態ベクトルは 1 次元ベクトルだが、密度行列は 2 次元行列なので、メモリレイアウトを意識する必要があった
- 色々な PR が飛び交う中で、merge を待っているとキリがないので、自分に必要な PR をピックアップして作業ブランチに取り込んで整合性をとって開発を進める必要があった
- 結局何だかんだで C++ の層に潜り込んで、障害が起こっている箇所を特定して、暫定修正する必要があった

といったところが大きめの問題であった。が、ソフトウェア開発としてはよくある事なので、その意味ではやはりそれほど難しい課題ではなかったとは言える。

# AerDensityMatrix のパフォーマンス

下回りの C++ 実装については、高速化の実装に詳しい Qiskit の専門家が実装されていたので、Python のガワを繋いだ後に n 量子ビットのランダムな回路上でパフォーマンスを測定すると以下のようになっていた。

![](/images/dwd-qiskit12/002.png)

y 軸を対数にするとより分かり易いのだが、Terra 版に比べて CPU 動作で約 3 倍高速な結果になっている。

# AerDensityMatrix のコード例

最後に少しコードを記載して終わりたい。Terra 版の `DensityMatrix` でも同様であるが、密度行列のシミュレーションを実行するクラスであるので、[混合状態](https://ja.wikipedia.org/wiki/%E9%87%8F%E5%AD%90%E7%8A%B6%E6%85%8B#%E6%B7%B7%E5%90%88%E7%8A%B6%E6%85%8B)を表現できる。以下は量子雑音として分極解消チャンネルの影響がある中でのシミュレーションの例である。

```python
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit_aer.noise.errors.standard_errors import depolarizing_error
from qiskit_aer.quantum_info import AerDensityMatrix as DensityMatrix

depola_1q = depolarizing_error(1e-2, 1).to_instruction()
depola_2q = depolarizing_error(1e-2, 2).to_instruction()

circ = QuantumCircuit(2)
circ.h(0)
circ.append(depola_1q, [0], [])
circ.cx(0, 1)
circ.append(depola_2q, [0, 1], [])
circ.measure_all()

display(circ.draw(scale=0.7))
print(AerSimulator().run(circ).result().get_counts(), '\n')
dm = DensityMatrix(circ.remove_final_measurements(False))
display(dm.draw('latex'))
```

![](/images/dwd-qiskit12/003.png)

雑音がない場合、4x4 の密度行列は四隅に約 0.5 の値が出るが、行列の中心部に雑音の影響が反映された行列になっている。

ついでに些細だが便利なポイントとしては、

```python
from qiskit_aer.quantum_info import AerDensityMatrix as DensityMatrix
```

と書いたように、基本的に Terra 実装を踏襲しているのでインターフェイスに互換性があり、既存実装の import 部分を差し替えるだけで高速化できる (はず) ということである。

また、コードやシーケンス図からも分かるように、`QuantumCircuit` を渡して、`AerDensityMatrix` インスタンスを作ると、初期化処理の中で時間発展のシミュレーションが実行されるので、インスタンスが作成された時点で時間発展後の状態が得られている。

# まとめ

去年の時点で draft PR 状態まで実装を進めていたので、実際のリリースまではそれほどやることはなかった。但し、多少のクリーンナップなどはあり、Python と C++ のファサード？と言うか多少の責任範囲の見直しなど細かい事もあった。とは言え、実際に正式にリリースされると、何かバグレポートが来るんじゃないだろうか？とか、CI でテストはされているものの、ちゃんと動くだろうか？といった心配はあった。

今までそれほど大きな OSS に PR を送ったこともなかったし、簡単な機能実装くらいしか PR したことがなかったので、今回グローバルに使われる SDK で、わりとまとまった実装規模になって良い経験になったと思う。
