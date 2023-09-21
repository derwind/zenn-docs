---
title: "Qiskit で遊んでみる (21) — QAOA でお絵描き"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "量子機械学習", "機械学習", "poem"]
published: true
---

# 目的

[TYTAN tutorial おすすめ5（お絵かきロジック）](https://colab.research.google.com/drive/1WwsQkrIGS7YMz26BvrExIBD3MvpxEhzT?usp=sharing) という [TYTANSDK](https://github.com/tytansdk/tytan) のチュートリアルがある。原理的には QAOA でいけるはずなので試してみたい。

`cuTensorNet` での実験は使われたし、何だかんだで時間がかかっていつ解けるのであろうかという感じであったので、今回は思い切って `cuStateVec` を用いる。

以下の内容は Google Colab の T4 を用いて計算した。

# 今回解きたい問題

必要なモジュールを import する。

```python
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
import matplotlib.pyplot as plt

from tytan import symbols_list, Compile, sampler, Auto_array

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator as AerEstimator
```

## TYTYAN を用いた QUBO の定式化

```python
from tytan import *
import numpy as np

n_qubits = 25

q00, q01, q02, q03, q04, \
q05, q06, q07, q08, q09, \
q10, q11, q12, q13, q14, \
q15, q16, q17, q18, q19, \
q20, q21, q22, q23, q24 = symbols_list(n_qubits, "q{}")

H = 0
H += (q00 + q01 + q02 + q03 + q04 - 2)**2
H += (q05 + q06 + q07 + q08 + q09 - 3)**2
H += (q10 + q11 + q12 + q13 + q14 - 3)**2
H += (q15 + q16 + q17 + q18 + q19 - 3)**2
H += (q20 + q21 + q22 + q23 + q24 - 2)**2

H += (q00 + q05 + q10 + q15 + q20 - 3)**2
H += (q01 + q06 + q11 + q16 + q21 - 2)**2
H += (q02 + q07 + q12 + q17 + q22 - 5)**2
H += (q03 + q08 + q13 + q18 + q23 - 2)**2
H += (q04 + q09 + q14 + q19 + q24 - 1)**2

H += -0.1 * (q00 * q01) -0.1 * (q01 * q02) -0.1 * (q02 * q03) -0.1 * (q03 * q04)
H += -0.1 * (q05 * q06) -0.1 * (q06 * q07) -0.1 * (q07 * q08) -0.1 * (q08 * q09)
H += -0.1 * (q10 * q11) -0.1 * (q11 * q12) -0.1 * (q12 * q13) -0.1 * (q13 * q14)
H += -0.1 * (q15 * q16) -0.1 * (q16 * q17) -0.1 * (q17 * q18) -0.1 * (q18 * q19)

H += -0.1 * (q00 * q05) -0.1 * (q05 * q10) -0.1 * (q10 * q15) -0.1 * (q15 * q20)
H += -0.1 * (q01 * q06) -0.1 * (q06 * q11) -0.1 * (q11 * q16) -0.1 * (q16 * q21)
H += -0.1 * (q02 * q07) -0.1 * (q07 * q12) -0.1 * (q12 * q17) -0.1 * (q17 * q22)
H += -0.1 * (q03 * q08) -0.1 * (q08 * q13) -0.1 * (q13 * q18) -0.1 * (q18 * q23)

H += 0.1 * (q20 * q21) + 0.1 * (q21 * q22) + 0.1 * (q22 * q23) + 0.1 * (q23 * q24)
```

## 疑似量子アニーリングで解く

まずは、TYTANSDK で Simulated Annealing にて答えを確認する。

```python
qubo, offset = Compile(H).get_qubo()
print(f"{offset=}")

solver = sampler.SASampler()

result = solver.run(qubo)

def rename(qname):
    symbol = qname[0]
    params = int(qname[1:])
    return symbol + f"{params // 5}_{params % 5}"

for r in result[:5]:
    print(f'Energy {round(r[1] + offset, 2)}, Occurrence {r[2]}')

    renamed_dict = {
        rename(k): v for k, v in r[0].items()
    }

    arr, subs = Auto_array(renamed_dict).get_ndarray('q{}_{}')
    print(arr)

    img, subs = Auto_array(renamed_dict).get_image('q{}_{}')
    plt.figure(figsize=(2, 2))
    plt.imshow(img, cmap="gray")
    plt.show()
```

> offset=78
> Energy -1.50, Occurrence 10
> [[0 0 1 1 0]
>  [0 0 1 1 1]
>  [1 1 1 0 0]
>  [1 1 1 0 0]
>  [1 0 1 0 0]]

![](/images/dwd-qiskit21/001.png)

ということでワンコ (？) みたいな画像が得られることが今回のゴールである。

# QAOA への道

以下は QUBO の `Z` および `ZZ` ハミルトニアンへの変換用のユーティリティである。

```python
def _calc_key(num_qubits: int, k: tuple[str] | tuple[str, str]) -> int:
    if len(k) == 1:
        left = k[0]
        ln = int(left[1:])
        return num_qubits * ln - 1
    elif len(k) == 2:
        left, right = k
        ln = int(left[1:])
        rn = int(right[1:])
        return num_qubits * num_qubits * ln + num_qubits * rn
    else:
        raise ValueError(f"len(k) = {len(k)} must be one or two.")

def get_ising(
    qubo: dict[tuple[str, str], float], num_qubits: int
) -> tuple[dict[tuple[str] | tuple[str, str], float], float]:
    ising_dict: dict[tuple[str] | tuple[str, str], float] = {}
    offset = 0.0

    for k, v in qubo.items():
        left, right = k
        ln = int(left[1:])
        rn = int(right[1:])
        new_k: tuple[str] | tuple[str, str]
        if rn < ln:
            ln, rn = rn, ln
        if ln == rn:
            new_k = (f"z{ln}",)
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += -v / 2
            offset += v / 2
        else:
            new_k = (f"z{ln}", f"z{rn}")
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += v / 4
            new_k = (f"z{ln}",)
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += -v / 4
            new_k = (f"z{rn}",)
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += -v / 4
            offset += v / 4

    ising_dict = {k: v for k, v in ising_dict.items() if not np.isclose(v, 0)}
    ising_dict = dict(
        sorted(ising_dict.items(), key=lambda k_v: _calc_key(num_qubits, k_v[0]))
    )
    return ising_dict, offset


def get_hamiltonian(
    ising_dict: dict[tuple[str] | tuple[str, str], float],
    num_qubits: int
) -> tuple[list[str], np.ndarray]:
    hamiltonian: list[str] = []
    for k in ising_dict.keys():
        ham = ["I"] * num_qubits
        if len(k) == 1:
            left = k[0]
            ln = int(left[1:])
            ham[ln] = "Z"
        elif len(k) == 2:
            left, right = k
            ln = int(left[1:])
            rn = int(right[1:])
            ham[ln] = ham[rn] = "Z"
        else:
            raise ValueError(f"len(k) = {len(k)} must be one or two.")
        hamiltonian.append("".join(ham))

    return hamiltonian, np.array(list(ising_dict.values()))
```

上記を使ってハミルトニアン `qubit_op` を得る。

```python
ising_dict, additional_offset = get_ising(qubo, n_qubits)
hamiltonian, coefficients = get_hamiltonian(ising_dict, n_qubits)
# 係数を正規化して、2 π を何周もしないようにする。
coefficients /= np.max(abs(coefficients))

qubit_op = SparsePauliOp(
    [ham[::-1] for ham in hamiltonian],
    coefficients,
)
```

# QAOA を実行する

Ansatz を自分で実装するのも面倒くさいので Qiskit を用いて作る。Sampler Primitive を使う場合は [Qiskit で遊んでみる (20) — Qiskit Optimization での QAOA](/derwind/articles/dwd-qiskit20) の内容になるが、今回は `cuStateVec` を用いて、Estimator Primitive で状態ベクトルから期待値計算を行わせる。特に、GPU を用いるので、Qiskit Aer 側の Primitive を用いる。

## Ansatz の定義

`n_reps` が小さいと思うような解が得られないような気がしたので思い切って 10 にしたが、もっと削れるかもしれない。

```python
%%time

initial_state_circuit = QuantumCircuit(n_qubits)
initial_state_circuit.h(initial_state_circuit.qregs[0][:])

n_reps = 10

ansatz = QAOAAnsatz(
    cost_operator=qubit_op,
    reps=n_reps,
    initial_state=initial_state_circuit,
    name='QAOA',
    flatten=True,
)
```

> CPU times: user 1.51 ms, sys: 0 ns, total: 1.51 ms
> Wall time: 1.52 ms

どのくらいの回路の深さなのだろうか？

```python
ansatz.depth()
```

> 1491

結構深い・・・。たぶん NISQ デバイスではエラー緩和を用いても計算は無理だけど (実質ノイズになる)、今回は気にしないことにする。

## コスト関数の定義

```python
losses = []
count = 0

# Qiskit Aer の Estimator Primitive を使う。
# 内部で AerSimulator を作るのでそれに渡す GPU 用のオプションを設定する。
estimator = AerEstimator(
    backend_options = {
        "device": "GPU",
        "cuStateVec_enable": True,
    }
)

def compute_expectation(params, *args):
    global count

    (estimator, ansatz, qubit_op) = args
    time_start = time.time()
    energy = estimator.run([ansatz], [qubit_op], params).result().values[0]
    if count % 10 == 0:
        time_end = time.time()
        print(f"[{count}] {energy} (elapsed={round(time_end - time_start, 3)}s)")
    count += 1

    losses.append(energy)

    return energy
```

## 最適化を回す

準備ができたので計算する。今回 Powell 法で最適化する。あまりよく分かっていないが、COBYLA 法で最適化をかけると何故かわりとすぐに最適化を諦めてしまうので、何となくで Powell を用いただけである[^1]。

[^1]: SciPy の実装を読み解けなくて手当たり次第しかない・・・。

```python
%%time

rng = np.random.default_rng(42)
# 初期値は小さめの範囲のランダムな値にしておく。
init = rng.random(ansatz.num_parameters) * np.pi

result = minimize(
    compute_expectation,
    init,
    args=(estimator, ansatz, qubit_op),
    method="Powell",
    options={
        "maxiter": 1000
    },
)

print(result.message)
print(f"opt value={round(result.fun, 3)}")
```

> ...
> [1660] -6.324880292338709 (elapsed=3.565s)
> [1670] -2.5975932459677407 (elapsed=3.196s)
> [1680] -6.224010836693549 (elapsed=3.291s)
> [1690] -5.934916834677419 (elapsed=3.224s)
> [1700] -6.297379032258063 (elapsed=3.184s)
> [1710] -6.196005544354837 (elapsed=3.288s)
> Optimization terminated successfully.
> opt value=-6.272
> CPU times: user 1h 30min 6s, sys: 4min 34s, total: 1h 34min 40s
> Wall time: 1h 34min 55s

1 周につき複数回 `compute_expectation` を呼んでいるようで、`maxiter` 回の呼び出しでは終わらなかった。適当に検索して調べた感じでは、最適化の手法によってはそういうものらしい。今回は気にしないことにする。

# 最適状態を確認する

シミュレータも `cuStateVec` で計算する。25 量子ビットなので、T4 の VRAM 16GB だとそろそろ厳しい領域に近づきつつはあるが、まだ余裕はある。

```python
%%time

opt_ansatz = ansatz.bind_parameters(result.x)
opt_ansatz.measure_all()

sim = AerSimulator(device="GPU", method="statevector", cuStateVec_enable=True)
t_qc = transpile(opt_ansatz, backend=sim)
counts = sim.run(t_qc, shots=1024*50).result().get_counts()
```

> CPU times: user 6.89 s, sys: 193 ms, total: 7.09 s
> Wall time: 7.14 s

今回は `cuStateVec` の速さに心底驚かされた。やはり巨大な行列を並列計算でひたすら掛けるのが得意なのだな・・・。

## 可視化をする

```python
topk = 5

for i, (k, n) in enumerate(sorted(counts.items(), key=lambda k_v: -k_v[1])):
    if n < 2:
        continue
    state = k[::-1]  # reverse order
    result_dict = {f"q{i // 5}_{i % 5}": int(v) for i, v in enumerate(state)}

    img, subs = Auto_array(result_dict).get_image('q{}_{}')
    plt.figure(figsize=(2, 2))
    plt.imshow(img, cmap="gray")

    i += 1
    if i >= topk:
        break
```

![](/images/dwd-qiskit21/001.png)

![](/images/dwd-qiskit21/004.png)

![](/images/dwd-qiskit21/005.png)

![](/images/dwd-qiskit21/006.png)

![](/images/dwd-qiskit21/007.png)

ということで一番上に求めたい絵が出ていることが確認できた。

# コストの推移

念のためコスト関数の動きも見ておく。

```python
plt.figure()
x = np.arange(0, len(losses), 1)
plt.plot(x, losses, color="blue")
plt.show()
```

![](/images/dwd-qiskit21/003.png)

Powell 法の詳細を知らないが、余分に何度か `compute_expectation` を呼んでいるようだったので変な感じの絵になっているが、下側だけ追いかけていくと緩やかに下がっていくコストが見える・・・。のでそれで満足することにする。

因みに、COBYLA 法で最適化すると以下のような状態のところで諦めてしまい、最適化がどうしてもそれ以上進まなかった。

![](/images/dwd-qiskit21/002.png)

# まとめ

`cuStateVec` 最強伝説。とにかく 30 量子ビット程度までなら迷わずに初手は `cuStateVec` 一択で良いと思う。50 量子ビットクラスになると、`cuTensorNet` の出番にはなるだろうが、正直計算速度などはあまり期待できないかもしれない。GPU マルチノードで分散計算するなどをすれば高速になるのだろうか？

とりあえずワンコ (？) の絵が得られた良かった。
