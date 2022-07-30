---
title: "Qiskit で遊んでみる (6) — QGSS2022 より"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: false
---

# 目的

[前回](/derwind/articles/dwd-qiskit05) に引き続き [Qiskit Global Summer School 2022: Quantum Simulations](https://qiskit.org/events/summer-school/) で興味を持ったテーマを復習する。

今回はハードウェアノイズのシミュレーションを題材にする。

# 準備

以下を import しているとする。

```python
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, coherent_unitary_error, depolarizing_error
from qiskit.circuit.library import RXGate
import matplotlib.pylab as plt
```

# 実機での計測

1 量子ビットの回路で $X$ ゲートを偶数回適用して $\ket{0}$ の確率振幅を測定する。$\ket{0}$ → $\ket{1}$ → $\ket{0}$ → $\ket{1}$ と flip するだけなので、理論上は結果は 1 である。

実際に試してみると:

```python
def measure_0(n_xgates, backend):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    # 指定の回数だけ X ゲートを適用する
    for _ in range(n_xgates):
        qc.x(0)
    qc.measure(qr, cr)
    shots = 1024
    counts = backend.run(qc, shots=shots).result().get_counts()
    count0 = 0
    if '0' in counts:
        count0 = counts['0']
    return count0/shots

IBMQ.load_account()
provider = IBMQ.get_provider()
backend = provider.get_backend('ibmq_belem')
prob0 = measure_0(20, backend)
print(prob0)
```

> [0.9599609375]

どうやら 5% くらい $\ket{1}$ が観測されたらしい。[^1]

[^1]: もっと色んな回数で試したかったが実機の待ち行列が厳しいので 1 回だけにした。

もし理想的な FTQC 上で実験ができたとすれば、以下のシミュレータの結果のようになるはずである（横は $X$ ゲートの適用回数、縦は $\ket{0}$ の測定結果で得た確率である）:

![](/images/dwd-qiskit06/001.png)

# 実機のフェイクでの測定

流石に実機で沢山の回数の試行を試すのは厳しいので、実機のノイズモデルを使ったシミュレータで測定してみよう。[Fake Provider](https://qiskit.org/documentation/apidoc/providers_fake_provider.html) を参考に `ibmq_belem` のフェイクを作成する。

```python
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_belem')
backend_sim = AerSimulator.from_backend(backend)

probs = []
for i in range(100):
    prob0 = measure_0(i, backend_sim)
    probs.append(prob0)
plt.plot(list(range(len(probs))), probs)
plt.show()
```

すると、FTQC とは異なり以下のように少しずつ確率が下がっていっている。

![](/images/dwd-qiskit06/002.png)

# 前回のおさらい

[前回](/derwind/articles/dwd-qiskit05)、

> 次回の記事の伏線の形で、ハミルトニアン $H = X$ の時間発展について考えてみる。これは $U(t) = \exp(-itH)$ であるが、$R_X(\theta) = \exp(-i \frac{\theta}{2} X)$ を思い出すと、$U(t) = R_X(2t)$ であることが分かる。

といったことを書いたが、時間 $t = \frac{\pi}{2}$ 経過後に $U(\frac{\pi}{2}) = R_X(\pi) = -i X$ なので、初期状態 $\ket{0}$ は Bloch 球の反対側、即ち $\ket{1}$ に到着するのである。この Bloch 球の旅路において僅かな誤差が生じるとする。つまり、$R_X(\pi)$ の実行が実際には $\tilde{R}_X(\pi) = R_X(\pi + \epsilon)$ になるといったことである。

# ノイズモデルの適用

$X$ ゲートがノイズの乗った回転ゲート $\tilde{R}_X(\pi)$ で実装されているとする。このために `measure_0` を少し書き直す:

```python
def measure_0_with_noise(n_xgates, backend, noise_model=None):
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    # 指定の回数だけ X ゲートを適用する
    # 但し、ノイズが乗っているので少しずつ理想からはずれた形になっていく
    for _ in range(n_xgates):
        # qc.x(np.pi, 0)
        qc.rx(np.pi, 0)
    qc.measure(qr, cr)
    shots = 1024
    counts = backend.run(qc, shots=shots).result().get_counts()
    count0 = 0
    if '0' in counts:
        count0 = counts['0']
    return count0/shots
```

次に、以下のような第 1 量子ビットの $R_X$ ゲートに作用する 2 つのノイズを定義する:

```python
def add_coherent_noise(noise_model):
    epsilon = 3*np.pi/180
    epsilon_rotation = RXGate(epsilon).to_matrix()
    over_rotation = coherent_unitary_error(epsilon_rotation)
    noise_model.add_quantum_error(over_rotation, ['rx'],
                                  qubits=[0], warnings=False)

def add_depolarizing_noise(noise_model):
    error = depolarizing_error(.02, 1)
    noise_model.add_quantum_error(error, ['rx'], qubits=[0], warnings=False)
```

これを使って、実験をしてみよう。

## コヒーレントノイズだけの時

```python
noise_model = NoiseModel()
add_coherent_noise(noise_model)
sim = AerSimulator(noise_model=noise_model)

probs = []
for i in range(100):
    prob0 = measure_0_with_noise(i, sim, noise_model)
    probs.append(prob0)
plt.plot(list(range(len(probs))), probs)
plt.show()
```

![](/images/dwd-qiskit06/003.png)

これは大分極端な結果になっているが、上に掲載した `ibmq_belem` のフェイクの時と同様に徐々に測定結果として $\ket{0}$ が得られる確率が減衰し、遂には反転してしまっている。もの凄い深い回路の場合やエラーが多いゲートを多用するとこういうことがあり得ることが示唆されている。

## 脱分極ノイズも追加する

```python
noise_model = NoiseModel()
add_coherent_noise(noise_model)
add_depolarizing_noise(noise_model)
sim = AerSimulator(noise_model=noise_model)

probs = []
for i in range(100):
    prob0 = measure_0_with_noise(i, sim, noise_model)
    probs.append(prob0)
plt.plot(list(range(len(probs))), probs)
plt.show()
```

![](/images/dwd-qiskit06/004.png)

これも大袈裟になってしまっているが、大分酷い結果になってしまった。

# まとめ

大分粗い設定かもしれないが、ある種のノイズモデルを定義して単純な回路のシミュレーションを行ってみた。これにより完璧ではないにせよ、どういったことが起こり得るかの知見が少し得られたように思う。また [Qiskit で遊んでみる (4)](/derwind/articles/dwd-qiskit04) の時に試したグローバーのアルゴリズムでは実際にかなり酷い結果になってしまった。

NISQ の抱える現実的な問題を踏まえつつ、測定エラーの軽減 (error mitigation) といったことも学んで最適なアプローチを模索したい。
