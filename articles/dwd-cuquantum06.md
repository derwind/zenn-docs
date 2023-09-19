---
title: "cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "qubo", "Python"]
published: true
---

# 目的

[cuQuantum で遊んでみる (5) — VQE その 2](/derwind/articles/dwd-cuquantum05) で VQE を無理やり実行してみたが、今回は 3 種類の方法で最大カット問題を解いてみたい。

# 最大カット問題

[Maximum cut](https://en.wikipedia.org/wiki/Maximum_cut) にあるように、頂点と辺がつながったグラフにおいて、頂点を黒と白に塗分ける類のものである。辺で繋がった頂点同士が異なる色で塗られている場合に、その辺は “カット可能” という解釈をする。カットできる箇所を最大化するという組み合わせ最適化問題である。

明らかに [最大カット問題](https://colab.research.google.com/drive/1cPM7qx-mTIqxHQztKHJuW3EaCHn5hegg?usp=sharing) のほうが明らかに優しい雰囲気の解説である。

# 今回試す手法

- [TYTAN SDK](https://github.com/tytansdk/tytan): 疑似量子アニーリング
- [Qiskit](https://github.com/Qiskit/qiskit): ゲート方式量子コンピューティング SDK
- [cuQuantum](https://github.com/NVIDIA/cuQuantum): GPU シミュレータ (今回はテンソルネットワークシミュレータの `cuTensorNet` を使う)

# TYTAN SDK

TYTAN SDK は疑似量子アニーリングによって QUBO (Quadratic Unconstrained Binary Optimization) を解く SDK である。

## 実装内容

Wikipedia の問題は、バイナリ変数 $q_i \in \{0, 1\}$ に対して、イジング変数を $z_i := 1 - 2 q_i \in \{-1, +1\}$ と置くことで、

$$
\begin{align*}
H = z_0 z_1 + z_0 z_2 + z_1 z_3 + z_2 z_3 + z_2 z_4 + z_3 z_4
\end{align*}
$$

を最小化する $z_0, \cdots, z_4$ を求める問題に帰着する。仮に -1 を白、+1 を黒と解釈すれば良い。バイナリ変数の目線で言うと、1 が白で 0 が黒になる。画像っぽくて分かりやすい気がする。

## 実装

まずは必要なモジュールを import する:

```python
import numpy as np
from tytan import (
    symbols_list,
    Compile,
    sampler,
)
```

[TYTAN SDK で遊んでみる (1) — 入門の入門](/derwind/articles/dwd-tytansdk01) で書いたような感じで、以下のような実装で解くことができる:

```python
# バイナリ変数を
q = np.array(symbols_list(5, 'q{}'))
# イジング変数にする
z0, z1, z2, z3, z4 = (1 - 2*q).tolist()

# 最小化するターゲット (ハミルトニアン)
H = z0*z1 + z0*z2 + z1*z3 + z2*z3 + z2*z4 + z3*z4

# 最適化ルーチン
qubo, offset = Compile(H).get_qubo()

solver = sampler.SASampler()
result = solver.run(qubo, shots=1000)

# 解を表示
for r in result:
    value = "".join([str(v) for v in r[0].values()])
    energy = round(float(r[1] + offset), 3)
    print(value, energy, r[2])
```

> 01100 -4.0 247
> 01101 -4.0 258
> 10010 -4.0 254
> 10011 -4.0 241

これらが求める解になっている。`01100` は頂点 0 が黒、頂点 1 が白、頂点 2 が白、頂点 3 が黒、頂点 4 が黒ということである。

これで解けたのだが、別の 2 種類の SDK でも解いてみる。

# Qiskit

ゲート方式の量子計算においては、メタヒューリスティックスである QAOA (Quantum Approximate Optimazation Algorithm) が用いられる。過去に書いたまったくもって分かりにくい記事 [QAOA を眺めてみる (2)](/derwind/articles/dwd-qiskit-qaoa02) が対応する。

## 実装する内容

まず、上記で用意したハミルトニアンのイジング変数 $z_i$ を $i$ 番目の量子ビットに作用する Pauli $Z$ ゲート $Z_i$ として読み換えて、以下のような問題ハミルトニアンを用意する。

$$
\begin{align*}
H_P = Z_0 \otimes Z_1 + Z_0 \otimes Z_2 + Z_1 \otimes Z_3 + Z_2 \otimes Z_3 + Z_2 \otimes Z_4 + Z_3 \otimes Z_4
\end{align*}
$$

更に、今回のように制約条件がない場合には、$\ket{\psi} = H^{\otimes n} \ket{0}^{\otimes n} = \ket{+}^{\otimes n}$ を固有ベクトルに持つようなミキシングハミルトニアン

$$
\begin{align*}
H_B = X_0 \otimes X_1 \otimes X_2 \otimes X_3 \otimes X_4
\end{align*}
$$

を用意する。

ここで、本来なら量子アニーリングに倣って量子断熱計算を行いたいが、量子ゲートが深くなるという理由で、似たような形の以下のパラメータ付き量子回路で代用する:

パラメータ $\beta = (\beta_1,\cdots,\beta_p) \in \R^p$ と $\gamma = (\gamma_1,\cdots,\gamma_p) \in \R^p$ に対して

$$
\begin{align*}
U(\beta, \gamma) = \underbrace{\left( e^{-i \beta_p H_B} e^{-i \gamma_p H_P} \right) \left( e^{-i \beta_{p-1} H_B} e^{-i \gamma_{p-1} H_P} \right) \cdots \left( e^{-i \beta_1 H_B} e^{-i \gamma_1 H_P} \right)}_{p}
\end{align*}
$$

を ansatz として $H_P$ の最小固有値を求める、或は $\ket{\psi({\beta, \gamma})} \!=\! U(\beta, \gamma) H^{\otimes n} \ket{0}$ として期待値

$$
\begin{align*}
\braket{\psi({\beta, \gamma}) | H_P | \psi({\beta, \gamma})}
\end{align*}
$$

を最小化するという計算を行わせる。変分原理より期待値の最小値は最小固有値に一致するので、やることは同じである。

## 実装

まずは必要なモジュールを import する:

```python
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator
```

Ansatz を実装する。上記で言うと $p=3$ 程度がほどほどに良い結果であったので `n_reps = 3` とする:

```python
n_qubits = 5
n_reps = 3

betas = ParameterVector("β", n_reps)
gammas = ParameterVector("γ", n_reps)
beta_idx = iter(range(n_reps))
bi = lambda: next(beta_idx)
gamma_idx = iter(range(n_reps))
gi = lambda: next(gamma_idx)

# Ansatz の回路
qc = QuantumCircuit(n_qubits)
# 初期状態の準備: H^n |0>
qc.h(qc.qregs[0][:])
for _ in range(n_reps):
    # 問題ハミルトニアンの時間発展演算子: exp(-γ/2 x 2H_p)
    gamma = gammas[gi()]
    qc.rzz(2*gamma, 0, 1)
    qc.rzz(2*gamma, 0, 2)
    qc.rzz(2*gamma, 1, 3)
    qc.rzz(2*gamma, 2, 3)
    qc.rzz(2*gamma, 2, 4)
    qc.rzz(2*gamma, 3, 4)
    qc.barrier()
    # ミキシングハミルトニアンの時間発展演算子: exp(-β/2 x 2H_B)
    beta = betas[bi()]
    for i in range(n_qubits):
        qc.rx(2*beta, i)

qc.draw("mpl", fold=-1)
```

![](/images/dwd-cuquantum06/001.png)

上記を使って、問題ハミルトニアンの期待値計算を最小化する。オプティマイザには COBYLA を用いる:

```python
%%time

def comnpute_expectation(params, *args):
    qc, hamiltonian, estimator = args
    qc = qc.bind_parameters(params)
    return estimator.run([qc], [hamiltonian]).result().values[0]

rng = np.random.default_rng(42)
init = rng.random(qc.num_parameters) * np.pi
pauli_list = ["IIIZZ", "IIZIZ", "IZIZI", "IZZII", "ZIZII", "ZZIII"]
hamiltonian = SparsePauliOp(pauli_list)
estimator = Estimator()

result = minimize(
    comnpute_expectation,
    init,
    args=(qc, hamiltonian, estimator),
    method="COBYLA",
    options={
        "maxiter": 500
    },
)

print(result.message)
print(f"opt value={round(result.fun, 3)}")
```

> Maximum number of function evaluations has been exceeded.
> opt value=-3.013
> CPU times: user 2.65 s, sys: 80.9 ms, total: 2.73 s
> Wall time: 3.01 s

最適化したハミルトニアンのエネルギー値は -3.013 で、疑似量子アニーリングで得た -4.0 とは少し乖離しているが、これについては `n_reps` を大きくすれば改善はする。今回はこの程度で満足することにする。

さて、この最小値を実現するパラメータによって、どういう量子状態が得られているのであろうか？Qiskit は LSB が右であるので表示の差異には文字列を反転させて表示することに注意して欲しい。

```python
opt_qc = qc.bind_parameters(result.x)
opt_qc.measure_all()

sim = AerSimulator()
t_qc = transpile(opt_qc, backend=sim)
counts = sim.run(t_qc).result().get_counts()
for k, n in sorted(counts.items(), key=lambda k_v: -k_v[1]):
    if n < 100:
        continue
    print(k[::-1], n)
```

> 01101 168
> 10011 165
> 01100 160
> 10010 156

要するに、かなりざっくりと下位の頻度の解を無視すれば、概ね “$\frac{1}{4} \ket{01101} + \frac{1}{4} \ket{10011} + \frac{1}{4} \ket{01100} + \frac{1}{4} \ket{10010}$” のような形で、最適解を与える状態の重ね合わせとして求まっている。ここで相対位相は求めていないので正確な状態はよく分からず、この式はあくまで雰囲気である。

疑似量子アニーリングで得た結果は以下であったので、最初と最後の解が得られた形である。なお、メタヒューリスティクスなので、解が得られたり得られなかったりする[^1]。

[^1]: 実際、2 番目と 3 番目の解が得られていないわけである。

[アニーリングの結果]

> 01100 -4.0 247
> 01101 -4.0 258
> 10010 -4.0 254
> 10011 -4.0 241

# cuQuantum

今回は、cuQuantum のうち、`cuTensorNet` を用いる。`cuTensorNet` は量子回路をテンソルネットワークに変換して、テンソル計算で数学的に同等の計算をする SDK である。この時に GPU 上にテンソルを乗せて、深層学習のような感じで計算を行うことになる。

今回は、まったくパフォーマンスを無視した実装を行うので、速度の観点ではまったく GPU のメリットを得られない。

実装は公式のチュートリアルである [qiskit_advanced.ipynb](https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/cutensornet/circuit_converter/qiskit_advanced.ipynb) を参考にした。内容的には、仮にテンソルが巨大になった場合でも、マルチ GPU 構成であれば、テンソルのスライシングを用いて各 GPU にテンソルを分散させることで大変巨大なテンソル計算ができる仕組みになっている・・・はずである。

## 実装の内容

Qiskit でやったことと同じ内容をテンソルの用語に置き換えるだけであり、数学的には同値である。

## 実装

まずは必要なモジュールを import する:

```python
import cupy as cp
from cuquantum import CircuitToEinsum, contract
```

`cuTensorNet` では、ハミルトニアンを構成する Pauli 演算子の並びを左を LSB と解釈して与える必要があるので、Qiskit のコードで使ったものを左右反転していることに注意して欲しい。

```python
pauli_list_cutn = [pauli[::-1] for pauli in pauli_list]

losses = []

def comnpute_expectation_tn(params, *args):
    qc, pauli_list_cutn = args
    qc = qc.bind_parameters(params)
    converter = CircuitToEinsum(qc)

    energy = 0.
    # Z_0 Z_1 + Z_0 Z_2 + Z_1 Z_3 + Z_2 Z_3 + Z_2 Z_4 + Z_3 Z_4 をばらして、
    # 部分期待値の足し合わせで全体の期待値を計算する
    for pauli_string in pauli_list_cutn:
        expr, operands = converter.expectation(pauli_string)
        energy += cp.asnumpy(contract(expr, *operands).real)

    losses.append(energy)

    return energy
```

最適化のルーチンは以下である。GPU を使っているので高速なイメージはあるが、パフォーマンスのチューニングを一切していないので、期待するより遥かに遅い[^2]。

[^2]: パフォーマンスの最適化を狙う場合、テンソルを直接操作しないとならないはずで、大変泥臭いのでここでは扱わない。

```python
%%time

rng = np.random.default_rng(42)
init = rng.random(qc.num_parameters) * np.pi

result = minimize(
    comnpute_expectation_tn,
    init,
    args=(qc, pauli_list_cutn),
    method="COBYLA",
    options={
        "maxiter": 500
    },
)

print(result.message)
print(f"opt value={round(result.fun, 3)}")
```

> Maximum number of function evaluations has been exceeded.
> opt value=-3.013
> CPU times: user 1min 51s, sys: 1.16 s, total: 1min 52s
> Wall time: 1min 52s

今回も上記と同程度の精度の最小エネルギーが求まった。

コストの動きも見てみよう。

```python
import matplotlib.pyplot as plt

plt.figure()
x = np.arange(0, len(losses), 1)
plt.plot(x, losses, color="blue")
plt.show()
```

![](/images/dwd-cuquantum06/002.png)

綺麗にコストが下がっていっている。より良い最小エネルギーを得るには、`maxiter` ではなく `n_reps` を増やすほうが良さそうである。

最適パラメータを用いてどういう状態が得られているか確認しよう。

```python
opt_qc = qc.bind_parameters(result.x)
opt_qc.measure_all()

sim = AerSimulator()
t_qc = transpile(opt_qc, backend=sim)
counts = sim.run(t_qc).result().get_counts()
for k, n in sorted(counts.items(), key=lambda k_v: -k_v[1]):
    if n < 100:
        continue
    print(k[::-1], n)
```

> 10010 169
> 01100 166
> 10011 149
> 01101 144

再度アニーリングの結果を掲載すると、以下だったのですべての解が求まったことになる。

[アニーリングの結果]

> 01100 -4.0 247
> 01101 -4.0 258
> 10010 -4.0 254
> 10011 -4.0 241

念のため、`cuTensorNet` の API でも求めてみよう。Qiskit の測定演算子があると計算できないので、それを外してからテンソルに変換する。今回は 5 量子ビットと規模が小さいので状態ベクトルを求めることにする。

```python
opt_qc.remove_final_measurements()
converter = CircuitToEinsum(opt_qc)
expr, operands = converter.state_vector()
result = contract(expr, *operands)
```

状態ベクトルから確率振幅を取り出して確率に変換して表示してみる。

```python
import itertools

d = {}
for i in range(2**5):
    k = bin(i)[2:].zfill(5)
    index = tuple(c for c in k)
    amplitude = result[index]
    d[k] = float(abs(amplitude)**2)
for k, prob in sorted(d.items(), key=lambda k_v: -k_v[1]):
    if prob < 0.01:
        continue
    print(k, round(prob, 3))
```

> 10011 0.156
> 01100 0.156
> 01101 0.156
> 10010 0.156
> 00110 0.059
> 11001 0.059
> 01001 0.046
> 10110 0.046
> 10001 0.046
> 01110 0.046

本質的に同じ結果が得られた。

# まとめ

疑似量子アニーリング、ゲート方式の量子計算、その GPU シミュレーションの 3 種類で最大カット問題を解いてみて、ほぼ同等の結果を得ることができた。
