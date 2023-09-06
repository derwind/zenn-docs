---
title: "cuQuantum で遊んでみる (9) — 少し規模が大きめの QAOA を計算する"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "Python"]
published: true
---

# 目的

[cuQuantum で遊んでみる (8) — QAOA の期待値計算高速化](/derwind/articles/dwd-cuquantum08) で大分計算周りをマシにしたのでもう少し本格的な計算をしてみたいというもの。

~~なお、今回の計算、どこか間違えているようなのだが間違えているなりにそこそこ行けてそうなので、一旦記事にしてしまって後日修正したい。~~ 修正した[^1]。

[^1]: QUBO 式をイジング変数で書き直す時に式の変形を間違えていた。

実験は Google Colab 上で T4 を使って行った。そんな手軽な環境で本格的な QAOA も VQE も実行できるのである。

# お題

TYTANSDK のチュートリアル [線形回帰](https://colab.research.google.com/drive/1Zt9FFF48S0tYRgpoiTOaLxiaHpWjTgLg?usp=sharing) を拝借したい。因みにこの問題は疑似量子アニーリングで計算したらすぐに結果が得られる。

# QAOA に書き直す

必要なモジュールを import する。

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
import matplotlib.pyplot as plt

from tytan import symbols_list, Compile, Auto_array

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit_aer import AerSimulator

import cupy as cp
from cuquantum import CircuitToEinsum, contract
```

また、`Rx_Rxdag` などの関数は [cuQuantum で遊んでみる (8) — QAOA の期待値計算高速化](/derwind/articles/dwd-cuquantum08) の通りとする。

## QUBO 定式化

[線形回帰](https://colab.research.google.com/drive/1Zt9FFF48S0tYRgpoiTOaLxiaHpWjTgLg?usp=sharing) の内容を QUBO で定式化する。

```python
n_qubits = 16
q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15 = \
    symbols_list(n_qubits, "q{}")

a = 10 + 10 * ((128*q0 + 64*q1 + 32*q2 + 16*q3 + 8*q4 + 4*q5 + 2*q6 + q7) / 256)
b = 0 + 1 * ((128*q8 + 64*q9 + 32*q10 + 16*q11 + 8*q12 + 4*q13 + 2*q14 + q15) / 256)

H = 0
H += (5.75 - (a*0.31 + b))**2

H += (8.56 - (a*0.4 + b))**2
H += (8.42 - (a*0.47 + b))**2
H += (7.78 - (a*0.4 + b))**2
H += (10.25 - (a*0.54 + b))**2
H += (6.79 - (a*0.36 + b))**2
H += (11.51 - (a*0.56 + b))**2
H += (7.66 - (a*0.43 + b))**2
H += (6.99 - (a*0.32 + b))**2
H += (10.61 - (a*0.6 + b))**2

qubo, offset = Compile(H).get_qubo()
```

この後、

$$
\begin{align*}
q_i q_j &= \frac{1 - z_i}{2} \frac{1 - z_j}{2} \\
&= \begin{cases}
\frac{1 - z_i}{2} \quad (i = j) \\
\frac{1}{4}(z_i z_j - z_i - z_j + 1) \quad (i \neq j)
\end{cases}
\end{align*}
$$

を用いて、イジング変数で書き直す。単に面倒くさいだけで `qubo` 辞書の内容を `ising_dict` 辞書に置き換えている。

```python
ising_dict = {}
additional_offset = 0
for k, v in qubo.items():
    left, right = k
    ln = int(left[1:])
    rn = int(right[1:])
    if rn < ln:
        ln, rn = rn, ln
    if ln == rn:
        new_k = (f"z{ln}",)
        ising_dict.setdefault(new_k, 0.0)
        ising_dict[new_k] += -v/2
        additional_offset += 1/2
    else:
        new_k = (f"z{ln}", f"z{rn}")
        ising_dict.setdefault(new_k, 0.0)
        ising_dict[new_k] += v/4
        new_k = (f"z{ln}",)
        ising_dict.setdefault(new_k, 0.0)
        ising_dict[new_k] += -v/4
        new_k = (f"z{rn}",)
        ising_dict.setdefault(new_k, 0.0)
        ising_dict[new_k] += -v/4
        additional_offset += 1/4

def calc_key(k: tuple[int] | tuple[int, int]):
    if len(k) == 1:
        left = k[0]
        ln = int(left[1:])
        return 256*ln - 1
    else:
        left, right = k
        ln = int(left[1:])
        rn = int(right[1:])
        return 256*ln + 16*rn

ising_dict = dict(sorted(ising_dict.items(), key=lambda k_v: calc_key(k_v[0])))

#for k in ising_dict:
#    print(k)

#print(additional_offset)
#for k, v in ising_dict.items():
#    k = str(k)
#    n_pad = 15 - len(k)
#    print(k, " "*n_pad, round(v, 5))
```

## QAOA 回路実装

この辺は [cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) と同様である。

```python
n_reps = 3

def rzz(qc, theta, qubit1, qubit2):
    qc.cx(qubit1, qubit2)
    qc.rz(theta, qubit2)
    qc.cx(qubit1, qubit2)

beta = ParameterVector("β", n_qubits * n_reps)
gamma = ParameterVector("γ", len(ising_dict) * n_reps)
beta_idx = iter(range(n_qubits * n_reps))
bi = lambda: next(beta_idx)
gamma_idx = iter(range(len(ising_dict) * n_reps))
gi = lambda: next(gamma_idx)

qc = QuantumCircuit(n_qubits)
qc.h(qc.qregs[0][:])
for _ in range(n_reps):
    for k in ising_dict:
        if len(k) == 1:
            left = k[0]
            ln = int(left[1:])
            rn = None
        else:
            left, right = k
            ln = int(left[1:])
            rn = int(right[1:])
            assert ln <= rn

        if rn is None:
            qc.rz(gamma[gi()], ln)
        else:
            rzz(qc, gamma[gi()], ln, rn)
    qc.barrier()
    for i in range(n_qubits):
        qc.rx(beta[bi()], i)

#qc.draw()
```

## テンソルネットワーク実装

[cuQuantum で遊んでみる (7) — 期待値計算再考](/derwind/articles/dwd-cuquantum07) で触れた高速計算用のハミルトニアンを作成する。

```python
I = cp.eye(2, dtype=complex)
Z = cp.array([
    [1, 0],
    [0, -1]
], dtype=complex)

def to_hamiltonian(ising_dict):
    hamiltonian = []
    for i in range(n_qubits):
        row = []
        for k in ising_dict.keys():
            if len(k) == 1:
                left = k[0]
                ln = int(left[1:])
                rn = None
            else:
                left, right = k
                ln = int(left[1:])
                rn = int(right[1:])
            if ln == i or rn == i:
                row.append(Z)
            else:
                row.append(I)
        hamiltonian.append(cp.array(row))
    return hamiltonian

hamiltonian = to_hamiltonian(ising_dict)
# 今回のハミルトニアンはごちゃごちゃした係数を持っている。
coefficients = np.array(list(ising_dict.values()))
```

量子回路にダミーのハミルトニアンを与えて一旦、`cuTensorNet` のテンソルネットワークに変換してからハミルトニアンを差し替える。やや回路が深いので計算に少し時間がかかる。

```python
%%time

dummy_hamiltonian = "Z" * n_qubits
expr, operands, pname2locs = circuit_to_einsum_expectation(qc, dummy_hamiltonian)
hamiltonian_locs = find_dummy_hamiltonian(operands)
es = expr.split("->")[0].split(",")
for loc in hamiltonian_locs:
    es[loc] = "食" + es[loc]
expr = ",".join(es) + "->食"

for ham, locs in zip(hamiltonian, hamiltonian_locs):
    operands[locs] = ham
```

> CPU times: user 2min 29s, sys: 351 ms, total: 2min 30s
> Wall time: 2min 33s

`expr` の中身が酷くて、眺めてみると面白い。「食 (hamu)」の部分にハミルトニアンを埋め込んでいる。

![](/images/dwd-cuquantum09/001.png)

コスト関数を定義する。

```python
param_names = [p.name for p in qc.parameters]

def comnpute_expectation_tn(params, *args):
    expr, operands, pname2locs = args
    energy = 0.

    pname2theta = dict(zip(param_names, params))
    parameterized_operands = replace_pauli(operands, pname2theta, pname2locs)

    # 純粋な期待値と係数のアダマール積をとって加算する。
    return np.sum(
        cp.asnumpy(contract(expr, *parameterized_operands).real) * coefficients
    )
```

# 最適化を回す

適当な `"maxiter"` で回す。

```python
%%time

init = np.random.randn(qc.num_parameters) * 2*np.pi

result = minimize(
    comnpute_expectation_tn,
    init,
    args=(expr, operands, pname2locs),
    method="COBYLA",
    options={
        "maxiter": 500
    },
)

print(result.message)
print(f"opt value={round(result.fun, 3)}")
```

> Maximum number of function evaluations has been exceeded.
> opt value=-1.068
> CPU times: user 1h 19min 46s, sys: 7min 12s, total: 1h 26min 59s
> Wall time: 1h 27min 6s

結構時間がかかったが、まだ常識的な範囲ではあろう。そろそろ途中経過の可視化を考えたほうが良い。コールバックから値を引っ張ってきて動的にグラフを書かせるか、`TensorBoard` を使うといったところか。

# 結果を考察する

そこそこの規模だとメモリを圧迫するので、念のため `AerSimulator` をテンソルネットワークをバックエンドにして実行する。今回のケースは `shots` を大きくしてもわりとすぐに完了する。理由はよく分かっていない。

```python
%%time

opt_qc = qc.bind_parameters(result.x)
opt_qc.measure_all()

sim = AerSimulator(device="GPU", method="tensor_network")
counts = sim.run(opt_qc, shots=1024*50).result().get_counts()
```

> CPU times: user 1min 37s, sys: 566 ms, total: 1min 37s
> Wall time: 1min 38s

Qiskit は右が LSB なので、左を LSB にする形で上位 20 の結果を表示してみる。

```python
topk = 20

states = []
for i, (k, n) in enumerate(sorted(counts.items(), key=lambda k_v: -k_v[1])):
    if n < 5:
        continue
    state = k[::-1]  # reverse order
    print(state, n)
    states.append(state)
    i += 1
    if i >= topk:
        break
```

> 1011101110000100 13
> 1011011110011101 12
> 1100010000111111 11
> 0001001000111110 11
> 1010111010100011 11
> 1010111011011011 11
> 1101100011111000 10
> 0101110011011010 10
> 1100010000000100 10
> 0000111111000101 10
> 1110011110100000 10
> 0000110011010001 10
> 0000101011001111 10
> 1100001100110110 10
> 1000101110100000 10
> 1000111111001110 9
> 1011001001010111 9
> 0111011101110100 9
> 0111110110100000 9
> 1000010111000001 9

線形回帰の係数を求める関数。

```python
def calc_ab(state: str):
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15 = [int(c)
        for c in state]
    a = 10 + 10 * ((128*q0 + 64*q1 + 32*q2 + 16*q3 + 8*q4 + 4*q5 + 2*q6 + q7) / 256)
    b = 0 + 1 * ((128*q8 + 64*q9 + 32*q10 + 16*q11 + 8*q12 + 4*q13 + 2*q14 + q15) / 256)
    return a, b
```

# 線形回帰のグラフを描く

ビリリダマ 10 匹のデータ:

```python
bibiridama = np.array([
    [0.31, 5.75],
    [0.4, 8.56],
    [0.47, 8.42],
    [0.4, 7.78],
    [0.54, 10.25],
    [0.36, 6.79],
    [0.56, 11.51],
    [0.43, 7.66],
    [0.32, 6.99],
    [0.6, 10.61],
])
```

グラフに描き出す。これが面白いくらい妥当な見栄えになっている。

```python
xs = bibiridama[:, 0]
ys = bibiridama[:, 1]
plt.figure()
plt.scatter(xs, ys)
print(states[0])
a_, b_ = calc_ab(states[0])
print(a_, b_)
x = np.arange(0.3, 0.6, 0.01)
y = a_*x + b_
plt.plot(x, y, color="red")
plt.show()
```

> 1011101110000100
> a = 17.3046875
> b = 0.515625

![](/images/dwd-cuquantum09/002.png)

2～3 番目の解でも同じようなグラフになっている。TYTANSDK のチュートリアルで疑似量子アニーリングで求める場合、

> a = 17.1875
> b = 0.88671875

らしいので、そこそこ良さそうである。因みに 6 番手の「1010111011011011」だと

> a = 16.796875
> b = 0.85546875

と言った感じで、より雰囲気が出る。要するに、top-1 が必ずしも最適解というわけではないという感じである。

# まとめ

疑似量子アニーリングの題材を使って 16 量子ビットの QAOA を実行してみた。

解の候補一覧から分かることだが、16 量子ビット程度の QAOA で既に最適解の確率振幅は驚く程小さい。僅差でトップになっているに過ぎない。Deutsch-Jozsa のアルゴリズムや Grover のアルゴリズムといった FTQC 用の解の絞り込みが工夫されたちゃんとしたアルゴリズムと異なり、NISQ 向けのメタヒューリスティクスなので仕方ない部分もあるかもしれないのだが、かなり大らかな気持ちで「解に近いものが得られたら御の字」くらいでやらないと厳しいのかもしれない。
