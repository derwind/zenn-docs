---
title: "Google Cirq で遊んでみる (1) — グローバー探索アルゴリズム"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Cirq", "poem", "Python"]
published: false
---

# 目的

[Qiskit で遊んでみる (4)](/derwind/articles/dwd-qiskit04) と [cuQuantum で遊んでみる (2) — グローバー探索アルゴリズム](/derwind/articles/dwd-cuquantum02) に引き続き、Google Cirq を使ってグローバー探索アルゴリズムを実行してみたい。

別にグローバー探索アルゴリズムに特別な思い入れがあるわけではなくて、フレームワークの試運転の題材として使っているだけなので、それ自身にはそれほどの意味はない[^1]。

[^1]: なお、Cirq はまだ使い出して 2 日目なので、良い書き方ができているかはよく分からない・・・。

# 基本的なパッケージのインポート

とりあえず以下くらいをインポートしておく。`SVGCircuit` を使わないと回路の可視化がかなり素っ気なくなるので、これは使っていきたい。

```python
import numpy as np
import cirq
from cirq.contrib.svg import SVGCircuit
%matplotlib inline
```

# オラクルの作成

以下のような書き方で良いのかは分からないのだが、`cirq.Moment` を使うと “同じ時間” でのゲート作用がざっくりかけて便利かもしれない。また、マルチ制御ゲートの記述がとても楽でゲートに `.controlled()` をぶら下げることで実現できてしまう。なお、Qiskit の時は量子ビットの順序を踏まえて `for i, digit in enumerate(state[::-1]):` で逆順にしたが今回は不要・・・なはずなのでそれはしていない。量子ビット 0 が MSB になるようにしている。

```python
def revserse_phase(circuit, qubits, state: str):
    n_qubits = len(qubits)
    qr = []
    for i, digit in enumerate(state):
        if digit == '0':
            qr.append(i)

    circuit.append([
        cirq.Moment(*[cirq.X(qubits[i]) for i in qr]),
        # MCZ start (HXH = Z)
        cirq.Z.controlled(n_qubits-1).on(
            *[qubits[i] for i in range(1, n_qubits)], qubits[0]),
        # MCZ end
        cirq.Moment(*[cirq.X(qubits[i]) for i in qr])
    ])

def define_oracle(n_qubits, solutions):
    # Create the oracle
    qubits = cirq.LineQubit.range(n_qubits)
    oracle = cirq.Circuit()

    for sol in solutions:
        revserse_phase(oracle, qubits, sol)

    return oracle
```

# Diffuser の作成

こちらもざっくりと書ける。Keras における `Sequential` みたいなノリで書けるので結構楽かもしれない。

```python
def define_diffuser(n_qubits):
    qubits = cirq.LineQubit.range(n_qubits)
    diffuser = cirq.Circuit()
    diffuser.append([
        cirq.Moment(*[cirq.H(qubits[i]) for i in range(n_qubits)]),
        cirq.Moment(*[cirq.X(qubits[i]) for i in range(n_qubits)]),
        # MCZ start (HXH = Z)
        cirq.Z.controlled(n_qubits-1).on(
            *[qubits[i] for i in range(1, n_qubits)], qubits[0]),
        # MCZ end
        cirq.Moment(*[cirq.X(qubits[i]) for i in range(n_qubits)]),
        cirq.Moment(*[cirq.H(qubits[i]) for i in range(n_qubits)])
    ])

    return diffuser
```

# 5 量子ビットに挑戦！

今回の問題でもまたまた 5 桁の 2 進数で解は `00101` と `11111` の 2 つとしてみた。まずは定番の回路の可視化をする。

```python
n_qubits = 5
solutions = ['00101', '11111']

oracle = define_oracle(n_qubits, solutions)
SVGCircuit(oracle)
```

Qiskit に慣れると実に味気ない感じに見えるが、まぁそれは本質的ではないので気にしないことにする。

![](/images/dwd-cirq-qsim01/001.png)

```python
diffuser = define_diffuser(n_qubits)
SVGCircuit(diffuser)
```

![](/images/dwd-cirq-qsim01/002.png)

意図した通りの内容になっているように見える。

# 問題を解いてみる

```python
N = 2**n_qubits
angle = np.arcsin(np.sqrt(len(solutions) / N))
counts = int((np.pi/2 - angle) / (2*angle) + 0.5)
print(f'{angle=}, {np.pi/2=}, {counts=}')
```

> angle=0.25268025514207865, np.pi/2=1.5707963267948966, counts=3

ということで、量子振幅増幅を 3 回くらい反復すると、解に対応する確率振幅が最大になっているはずである。

```python
qubits = cirq.LineQubit.range(n_qubits)
grover = cirq.Circuit()
# initialize |s>
grover.append([
    cirq.Moment(*[cirq.H(qubits[i]) for i in range(n_qubits)])
])
for _ in range(counts):
    grover += oracle
    grover += diffuser
SVGCircuit(grover)
```

![](/images/dwd-cirq-qsim01/003.png)

まぁ、こんな感じで・・・。

# qsim シミュレータを使う

[qsim](https://quantumai.google/qsim) を読むと

> Optimized quantum circuit simulators
>
> **qsim**
> qsim is a full wave function simulator written in C++. It uses gate fusion, AVX/FMA vectorized instructions and multi-threading using OpenMP to achieve state of the art simulations of quantum circuits. qsim is integrated with Cirq and can be used to run simulations of up to 40 qubits on a 90 core Intel Xeon workstation.

ということで、最適化された強いシミュレータということのようだ。今回これを使う。次回かは分からないけど、[Get started with qsimcirq](https://quantumai.google/qsim/tutorials/qsimcirq) の左ペインに見えている [GPU-based quantum simulation](https://quantumai.google/qsim/tutorials/gcp_gpu) から `cuQuantum` に繋ぎたいという狙いがあるのである。

```python
import qsimcirq
import matplotlib.pyplot as plt

def binary_labels(num_qubits):
    return [bin(x)[2:].zfill(num_qubits) for x in range(2 ** num_qubits)]

qubits = cirq.LineQubit.range(n_qubits)
grover.append(cirq.measure(qubits[:]))

simulator = qsimcirq.QSimSimulator()
result = simulator.run(grover, repetitions=1000)
# 確率として見たい場合には以下のようにすれば良い。
#result = cirq.get_state_histogram(result)
#result = result / np.sum(result)
_ = cirq.plot_state_histogram(
    result, plt.subplot(), title = 'Measurement results',
    xlabel = 'State', ylabel = 'Count',
    tick_label=binary_labels(n_qubits))
plt.xticks(rotation=70)
plt.show()
```

![](/images/dwd-cirq-qsim01/004.png)

なんとなく Qiskit 風味の出力にしてみた。この図より解 `00101` と `11111` の確率振幅が極めて大きくなっているのが分かる。

# まとめ

駆け足で `Cirq` を使ってグローバー探索アルゴリズムを使ってみた。何となく全般に Keras っぽさがあるので、Qiskit や Blueqat のように回路に対するメソッド呼び出しでゲートを追加するのではなく、ゲートを作って、それをディープラーニングのモデルに対するレイヤのように append していくイメージになるようだ。この辺は考え方の問題だと思うので、適当に頭の中で世界観を切り替えて使えば良いだろう。
