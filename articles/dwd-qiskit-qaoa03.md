---
title: "QAOA を眺めてみる (3) ― グラフカラーリング問題と QAOA"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "QUBO"]
published: true
---

# 目的


blueqat さんのブログ記事 [HOBOソルバーでグラフカラーリング問題を効率化](https://blueqat.com/yuichiro_minato2/ae758ca8-27fe-43e8-8bdc-2171dfc3c01e) を敢えて QAOA で解いてみようというもの。

# グラフカラーリング問題

グラフカラーリング問題というのはグラフ（頂点、辺）において、辺で繋がった両端の頂点同士が異なる色になるように、指定の範囲の色で塗り分けるという問題である。

まず [HOBOソルバーでグラフカラーリング問題を効率化](https://blueqat.com/yuichiro_minato2/ae758ca8-27fe-43e8-8bdc-2171dfc3c01e) では以下のような 5 頂点のグラフの 4 色カラーリング問題を HOBO ソルバで解いている。頂点の辺での接続は以下の通りである:

![](/images/dwd-qiskit-qaoa03/001.png =500x)

# QAOA で解く

QAOA でも `Rzzz` ゲートなどを作ることで HOBO を直接扱えるが、今回は敢えて QUBO を扱ってみたい。この場合、扱いが少しややこしくなるため、まずは小さな問題を解きたい。**頂点を 4 つに減らし**、4 色での塗分けを考える。頂点の接続は以下とする。

![](/images/dwd-qiskit-qaoa03/002.png =500x)

基本的には [blueqatでXYミキサーを用いた制約付きQAOA](https://qiita.com/ryuNagai/items/1836601f4d3c5ec9e336) と同じはずだが、今回は補助量子ビットを使わない方法を用いた。

今回、後で最初の 5 頂点 4 色問題を解くことも想定して、GPU シミュレーションを行う。Colab 上で T4 を用いて計算を行いたい。

必要なモジュールのインストール:

```sh
pip install -qU qiskit qiskit[visualization] qiskit-aer-gpu tytan
```

念のため重要なパッケージのバージョンを表示しておく。

```sh
%%bash

pip list | egrep -e "(qiskit|tytan)"
```

> qiskit                           1.1.1
> qiskit-aer-gpu                   0.14.2
> tytan                            0.0.28

モジュールを import する:

```python
from __future__ import annotations

import pprint
import re
import sys

import numpy as np
import scipy as sp
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from tytan import (
    symbols, symbols_list, symbols_nbit, sampler, Auto_array, Compile
)

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.quantum_info import AerStatevector
```

## QUBO での定式化

参考資料として [tytan_tutorial](https://github.com/tytansdk/tytan_tutorial) の「初代基礎コース」の「tutorial04. グラフ分割問題、グラフカラーリング問題」を参考にする。

バイナリ変数として 2 次元の $q_{v,i}$ を考える。$v$ は頂点の番号で、今回 0, 1, 2, 3 である。$i$ は色の番号で、こちらも今回は 0, 1, 2, 3 である。

$$
\begin{align*}
q_{v,i} = \begin{cases}
1, \quad \text{頂点} v \text{が色} i \text{で塗られる} \\
0, \quad \text{otherwise}
\end{cases}
\end{align*}
$$

とする。

**ワンホット制約項**

同じ頂点 $v$ においては色はただ 1 つ決まる必要があるのでワンホット制約

$$
\begin{align*}
\left(\sum_{i=0}^3 q_{v,i} - 1\right)^2
\end{align*}
$$

を設定する。以下のコードでは `HA` が対応する。細かいテクニック類は Vignette & Clarity さんの記事 [21-12. 量子アニーリングのQUBOで設定可能な条件式まとめ（保存版）](https://vigne-cla.com/21-12/) が詳しい。

**コスト項**

辺 $(u, v)$ で接続された頂点同士が同じ色 $i$ で塗られることを禁止したいので、$q_{u,i} q_{v,i}$ にペナルティを設定したい。頂点の接続の集合 $E = {(u, v)}$ を考え以下のようなコスト

$$
\begin{align*}
\sum_{u, v \in E} \sum_{i=0}^3 q_{u,i} q_{v,i}
\end{align*}
$$

を設定する。以下のコードでは `HB` が対応する。

```python
%%time

n_vertices = 4
n_colors = 4

# vertex (v=A, B, C, D), color (i=0, 1, 2, 3)
q = symbols_list([n_vertices, n_colors], 'q{}_{}')

# ワンホット制約
HA = 0
for v in range(n_vertices):
    HA += (1 - sum(q[v][i] for i in range(n_colors))) ** 2

# 接続された頂点間の色が同一の場合に大きくなるコスト関数
HB = 0
E = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]  # AB, AC, BC, BD, CD
for u, v in E:
    for i in range(n_colors):
        HB += q[u][i] * q[v][i]
```

## QUBO からイジングハミルトニアンへの変換

[cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) でも触れたが、この QUBO 式を $z_i = 1 - 2q_i$ という変数変換により、$z_i$ を用いたイジングハミルトニアンに置き換える必要がある。

[Qiskit で遊んでみる (21) — QAOA でお絵描き](/derwind/articles/dwd-qiskit21) で使った関数を流用して以下のようなユーティリティを作った。以下を使う必要は特にはなく、とにかく QUBO をイジング式に変換できれば何でも良い。

**注意点（変更箇所）**:

- 引用記事では `"q{}"` という名前でバイナリ変数を定義したが、今回は `"q{}_{}"` という 2 つのプレースホルダーを持つ形にしたのでここを拡張している。
- `{("z1", "z3"): -2.5, ...}` みたいな辞書を返していた部分を、今回はインデックスだけにして `{(1, 3): -2.5, ...}` で返すようにした。

いずれ汎用的にしたい・・。

```python
# calc keys for double sort
def _calc_key(num_qubits: int, k: tuple[str] | tuple[str, str]) -> int:
    if len(k) == 1:
        ln = k[0]
        return num_qubits * ln - 1
    elif len(k) == 2:
        ln, rn = k
        return num_qubits * num_qubits * ln + num_qubits * rn
    else:
        raise ValueError(f"len(k) = {len(k)} must be one or two.")


def get_ising(
    qubo: dict[tuple[str, str], float], n_vertices: int, n_colors: int
) -> tuple[dict[tuple[str] | tuple[str, str], float], float]:
    ising_dict: dict[tuple[int] | tuple[int, int], float] = {}
    offset = 0.0

    num_qubits = n_vertices * n_colors

    for k, v in qubo.items():
        left, right = k
        vertex, color = [int(v) for v in left[1:].split("_")]
        ln = n_colors * vertex + color
        vertex, color = [int(v) for v in right[1:].split("_")]
        rn = n_colors * vertex + color
        new_k: tuple[str] | tuple[str, str]
        if rn < ln:
            ln, rn = rn, ln
        if ln == rn:
            new_k = (ln,)
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += -v / 2
            offset += v / 2
        else:
            new_k = (ln, rn)
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += v / 4
            new_k = (ln,)
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += -v / 4
            new_k = (rn,)
            ising_dict.setdefault(new_k, 0.0)
            ising_dict[new_k] += -v / 4
            offset += v / 4

    ising_dict = {k: v for k, v in ising_dict.items() if not np.isclose(v, 0)}
    ising_dict = dict(
        sorted(ising_dict.items(), key=lambda k_v: _calc_key(num_qubits, k_v[0]))
    )
    return ising_dict, offset
```

これを用いて QUBO をイジング形式に変換する。後述する理由で、今回は（`HA` は定義したもののこれは用いずに）`HB` だけ変換する。4 頂点 4 色なので、$4 \times 4 = 16$ 個の量子ビットへの対応となる。`(0, 4): 0.25` などは、$0.25 z_0 z_4$ に対応する。以下、$-0.5 z_0 + 0.25 z_0 z_4 + 0.25 z_0 z_8 + \cdots$ のようなイジング形式が得られている。

```python
qubo, offset = Compile(HB).get_qubo()
ising, ising_offset = get_ising(qubo, n_vertices, n_colors)
pprint.pprint(ising)
```

> {(0,): -0.5,
>  (0, 4): 0.25,
>  (0, 8): 0.25,
>  (1,): -0.5,
>  (1, 5): 0.25,
>  (1, 9): 0.25,
>  (2,): -0.5,
>  (2, 6): 0.25,
>  (2, 10): 0.25,
>  (3,): -0.5,
>  (3, 7): 0.25,
>  (3, 11): 0.25,
>  (4,): -0.75,
>  (4, 8): 0.25,
>  (4, 12): 0.25,
>  (5,): -0.75,
>  (5, 9): 0.25,
>  (5, 13): 0.25,
>  (6,): -0.75,
>  (6, 10): 0.25,
>  (6, 14): 0.25,
>  (7,): -0.75,
>  (7, 11): 0.25,
>  (7, 15): 0.25,
>  (8,): -0.75,
>  (8, 12): 0.25,
>  (9,): -0.75,
>  (9, 13): 0.25,
>  (10,): -0.75,
>  (10, 14): 0.25,
>  (11,): -0.75,
>  (11, 15): 0.25,
>  (12,): -0.5,
>  (13,): -0.5,
>  (14,): -0.5,
>  (15,): -0.5}

イジング形式を QAOA の問題ハミルトニアンに変換するのは以下のような実装で可能である。Qiskit を使うので、`SparsePauliOp` の書式に落とし込めば良い。

```python
def convert_to_z(indices: tuple[int, ...], num_qubits: int) -> str:
    paulis = ["I"] * num_qubits
    for i in indices:
        paulis[i] = "Z"
    return "".join(paulis[::-1])


def define_observables(
    ising: dict[tuple[int, ...], float], num_qubits: int
) -> SparsePauliOp:
    ham_coeffs = [(convert_to_z(k, num_qubits), v) for k, v in ising.items()]
    return SparsePauliOp.from_list(ham_coeffs)
```

## ワンホット制約

さて、次に `HA` であるが、これはワンホット制約を満たすための項であった。QAOA の場合 $XY$ ミキサーというものを使うことで（特に complete-$XY$ ミキサーとそこ固有状態である Hamming 重み 1 の Dicke 状態を使うことで）この制約を満たすことができる。この辺は長くなるので、Appendix にて後述した。


### Dicke 状態の作成

詳細は文献 [6] に委ねるとして実装は以下のようになる:

```python
def CCRYGate(theta: float) -> Gate:
    return RYGate(theta).control(2)


# https://arxiv.org/pdf/1904.07358 Figure 2.
def SCSn(num_qubits: int, first_block: bool = False):
    qc = QuantumCircuit(num_qubits)
    denominator = num_qubits + 1 if first_block else num_qubits

    for i, loc in enumerate(range(num_qubits - 2, -1, -1)):
        qc.cx(loc, num_qubits - 1)
        if loc + 1 == num_qubits - 1:
            qc.cry(2 * np.arccos(np.sqrt((i + 1) / denominator)), num_qubits - 1, loc)
        else:
            qc.append(CCRYGate(2 * np.arccos(np.sqrt((i + 1) / denominator))), [num_qubits - 1, loc + 1, loc])
        qc.cx(loc, num_qubits - 1)
    return qc


def make_dicke_circuit(n: int, k: int) -> QuantumCircuit:
    """make a Dicke circuit

    Andreas Bärtschi, Stephan Eidenbenz, Deterministic Preparation of Dicke States,
    https://arxiv.org/abs/1904.07358

    Args:
        n: number of qubits
        k: Hamming weight

    Returns:
        QuantumCircuit: a quantum circuit
    """

    if n < 2:
        raise ValueError(f"n ({n}) must be equal to or greater than 2.")
    if n < k:
        raise ValueError(f"n ({n}) must be equal to or greater than k ({k}).")

    dicke = QuantumCircuit(n)
    dicke.x(dicke.qregs[0][n-k:n])

    block = SCSn(n - 1, first_block=True)
    dicke = dicke.compose(block, range(1, n))

    for m in range(max(n - 1, 2), 1, -1):
        block = SCSn(m)
        dicke = dicke.compose(block, range(m))
    
    return dicke
```

今回欲しいケースだと各頂点において 4 色のうちの 1 色が選ばれて欲しいので以下のようなものとなる:

```python
dicke = make_dicke_circuit(4, 1)
display(dicke.draw("mpl", style="clifford", scale=0.7, fold=-1))
display(AerStatevector(dicke).draw("latex"))
```

![](/images/dwd-qiskit-qaoa03/003.png =500x)

$$
\begin{align*}
\frac{1}{2} (\ket{0001} + \ket{0010} + \ket{0100} + \ket{1000})
\end{align*}
$$

## Ansatz を作成する回路

イジングハミルトニアンを問題ハミルトニアンとし、ミキシングハミルトニアンに complete-$XY$ ミキサーを使う ansatz を作成する。

```python
def apply_rotated_multiple_z(qc, indices, theta, gamma):
    prev_idx = indices[0]
    for idx in indices[1:]:
        qc.cx(prev_idx, idx)
        prev_idx = idx
    qc.rz(theta * gamma, indices[-1])
    prev_idx = indices[-1]
    for idx in reversed(indices[:-1]):
        qc.cx(idx, prev_idx)
        prev_idx = idx


def define_qaoa_ansatz(
    ising: dict[tuple[int, ...], float], num_qubits: int, n_reps: int
) -> QuantumCircuit:
    gammas = ParameterVector("γ", n_reps)
    betas = ParameterVector("β", n_reps)
    ansatz = QuantumCircuit(num_qubits)
    for i in range(n_reps):
        gamma = gammas[i]
        beta = betas[i]
        # time evolution of the problem Hamiltonian: ZZ-hamiltonian
        for key, value in ising.items():
            apply_rotated_multiple_z(ansatz, key, value, gamma)
        ansatz.barrier()
        # time evolution of the mixing Hamiltonian: complete-XY mixer
        for vertex in range(n_vertices):
            offset = n_colors * vertex
            for j in range(offset, offset + n_colors - 1):
                for k in range(j + 1, offset + n_colors):
                    ansatz.rxx(beta, j, k)
                    ansatz.ryy(beta, j, k)
        ansatz.barrier()
    return ansatz
```

## 期待値計算

イジングハミルトニアンの ansatz による期待値を求める関数を定義する。[EstimatorV2](https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/qiskit_ibm_runtime.EstimatorV2) という新しい目のモジュールを用いるのでやや癖のある書き方だが、流儀に合わせておく。

```python
def compute_expectation(params, *args):
    (estimator, ansatz, observables, notify_loss) = args
    pubs = [(ansatz, observables, params)]
    job = estimator.run(pubs)  # V2
    energy = job.result()[0].data.evs
    energy = energy.item()  # np.ndarray -> float

    if notify_loss is not None:
        notify_loss(energy)

    return energy
```

これで全てユーティリティは揃ったので最適化ルーチンを回すことになる。

## 最適化

### Ansatz の作成

まず、初期状態の準備として、16 量子ビットのうち 4 量子ビットずつで Hamming 重み 1 の 4 量子ビットの Dicke 状態 $\ket{D_1^4}$ を作成する。

```python
num_qubits = n_vertices * n_colors  # 4x4 = 16
n_reps = 20

ansatz = QuantumCircuit(num_qubits)

dicke = make_dicke_circuit(4, 1)
for i in range(0, num_qubits, 4):
    ansatz = ansatz.compose(dicke, range(i, i + 4))

ansatz.barrier()

qaoa_ansatz = define_qaoa_ansatz(ising, num_qubits, n_reps)

ansatz = ansatz.compose(qaoa_ansatz)
ansatz.draw("mpl", style="clifford", scale=0.3, fold=60)
```

![](/images/dwd-qiskit-qaoa03/004.png)

`n_reps = 20` としたので 2 回目の繰り返しの途中までしか画像キャプチャできていないが、このような絵になる。左上に Dicke 状態 $\ket{D_1^4}$ が 4 量子ビットずつで 4 個見える。続いてイジングハミルトニアンの時間発展が $RZZ$ や $RZ$ ゲートを用いて階段状に連なっている。左下にうつって、complete-$XY$ ミキサーの時間発展が密集して、2 周目のブロックにうつっている。

### イジングハミルトニアンの作成

次にイジング形式からイジングハミルトニアンを作る。

```python
observables = define_observables(ising, num_qubits)
pprint.pprint(observables)
```

> SparsePauliOp(['IIIIIIIIIIIIIIIZ', 'IIIIIIIIIIIIIIZI', 'IIIIIIIIIIIIIZII', 'IIIIIIIIIIIIZIII', 'IIIIIIIIIIIZIIII', 'IIIIIIIIIIIZIIIZ', 'IIIIIIIIIIZIIIII', 'IIIIIIIIIZIIIIII', 'IIIIIIIIZIIIIIII', 'IIIIIIIZIIIIIIII', 'IIIIIIIZIIIIIIIZ', 'IIIIIIZIIIIIIIII', 'IIIIIZIIIIIIIIII', 'IIIIZIIIIIIIIIII', 'IIIZIIIIIIIIIIII', 'IIZIIIIIIIIIIIII', 'IZIIIIIIIIIIIIII', 'ZIIIIIIIIIIIIIII', 'IIIIIIIIIIZIIIZI', 'IIIIIIZIIIIIIIZI', 'IIIIIIIIIZIIIZII', 'IIIIIZIIIIIIIZII', 'IIIIIIIIZIIIZIII', 'IIIIZIIIIIIIZIII', 'IIIIIIIZIIIZIIII', 'IIIZIIIIIIIZIIII', 'IIIIIIZIIIZIIIII', 'IIZIIIIIIIZIIIII', 'IIIIIZIIIZIIIIII', 'IZIIIIIIIZIIIIII', 'IIIIZIIIZIIIIIII', 'ZIIIIIIIZIIIIIII', 'IIIZIIIZIIIIIIII', 'IIZIIIZIIIIIIIII', 'IZIIIZIIIIIIIIII', 'ZIIIZIIIIIIIIIII'],
>               coeffs=[-0.5 +0.j, -0.5 +0.j, -0.5 +0.j, -0.5 +0.j, -0.75+0.j,  0.25+0.j,
>  -0.75+0.j, -0.75+0.j, -0.75+0.j, -0.75+0.j,  0.25+0.j, -0.75+0.j,
>  -0.75+0.j, -0.75+0.j, -0.5 +0.j, -0.5 +0.j, -0.5 +0.j, -0.5 +0.j,
>   0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,
>   0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,
>   0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j,  0.25+0.j])

### トランスパイル

今回、Dicke 状態の作成で `RYGate(theta).control(2)` という実装を用いたが、これを分解しないと `AerSimulator` 上でも実行できないので、`AerSimulator` 向けにトランスパイルする[^1][^2]。

[^1]: 以下のような方法を使えば不要かもしれない。<br>- [CCCRY gate creation](https://quantumcomputing.stackexchange.com/questions/27546/cccry-gate-creation) <br>- [How can we construct a control-control y-rotation (CCRy) gate in Qiskit?](https://quantumcomputing.stackexchange.com/questions/16208/how-can-we-construct-a-control-control-y-rotation-ccry-gate-in-qiskit) <br>- [A problem with application of multi controlled rotation gates](https://quantumcomputing.stackexchange.com/questions/15475/a-problem-with-application-of-multi-controlled-rotation-gates/15478#15478)

[^2]: [Qiskit で遊んでみる (24) — Qiskit Runtime local testing mode を使ってみる](/derwind/articles/dwd-qiskit24) でも軽く触れたが、最近は 127 量子ビット超の超伝導量子コンピュータの実機上で何とか回路を実行するために細かいトランスパイルの制御をさせたいらしく、`PassManager` を使ったトランスパイル方法が推しのようである。[ISA回路とは](https://www.ibm.com/blogs/systems/jp-ja/isa-circuits/) に説明があるが、細かくは触れずに、ISA (Instruction Set Architecture) 回路へとトランスパイルして、イジングハミルトニアンもそのレイアウトへと合わせ込む[^3]。

[^3]: `PassManager` を深く理解してカスタマイズすると、オレオレゲートを定義して、それをトランスパイルするという芸当も実装可能である・・・。

以下は “こういうもの” として深く気にせずに実行する。

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


backend = AerSimulator()

pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_ansatz = pm.run(ansatz)
layout = isa_ansatz.layout
isa_observables = observables.apply_layout(layout)
```

### 最適化メインルーチン

念のために GPU (`cuStateVec`) を使う形でシミュレーションを実行する。

```python
%%time

rng = np.random.default_rng(42)
init = rng.random(ansatz.num_parameters) * np.pi

options = {
    "backend_options": {
        "device": "GPU",
        "cuStateVec_enable": True,
    }
}

estimator = EstimatorV2(options=options)

losses = []

def notify_loss(energy):
    global losses
    losses.append(energy)

maxiter = 10000

result = minimize(
    compute_expectation,
    init,
    args=(estimator, isa_ansatz, isa_observables, notify_loss),
    method="Powell",
    options={
        "maxiter": maxiter
    },
)
```

> CPU times: user 32min 29s, sys: 28.9 s, total: 32min 58s
> Wall time: 33min 12s

NVIDIA T4 で実験したが、30 分ちょっとで最適化が完了した。今回はナイーブに実行したが、文献 [3] を参考にパラメータスケジュールを実装すれば収束を早めることができるかもしれない。

## 結果の確認

まず、エネルギーの最小値を見てみる。

```python
print(result.message)
print(f"opt value={round(result.fun.item(), 3)}")
```

> Optimization terminated successfully.
> opt value=-4.582

エネルギーの推移を可視化する。

```python
plt.figure()
x = np.arange(0, len(losses), 1)
plt.plot(x, losses, color="blue")
plt.show()
```

![](/images/dwd-qiskit-qaoa03/005.png =500x)

もう少し下がりそうではあるが、今回はこれくらいで問題は解けた。

### サンプリングを行う

```python
%%time

opt_ansatz = isa_ansatz.assign_parameters(result.x)
opt_ansatz.measure_all()

sim = AerSimulator(device="GPU", method="statevector", cuStateVec_enable=True)
counts = sim.run(opt_ansatz, shots=1024*50).result().get_counts()

ok_count = 0
ng_count = 0


def bin2int(arr):
    for i, v in enumerate(arr[::-1]):
        if int(v) == 1:
            return i
    return -1


for state, count in list(sorted(counts.items(), key=lambda k_v: -k_v[1]))[:20]:
    state_ = list(state)
    state_ = np.array([state_[i:i + n_colors] for i in range(0, num_qubits, n_colors)], dtype=int)
    colors = [bin2int(stat) for stat in state_]
    D, C, B, A = colors
    ok = (A != B) and (A != C) and (B != C) and (B != D) and (C != D)
    print(state, f"{A=} {B=} {C=} {D=}", ok, f"({counts[state]} times)")

for state, count in list(sorted(counts.items(), key=lambda k_v: -k_v[1])):
    state_ = list(state)
    state_ = np.array([state_[i:i + n_colors] for i in range(0, num_qubits, n_colors)], dtype=int)
    if np.all(np.sum(state_, axis=1) == 1):
        colors = [bin2int(stat) for stat in state_]
        D, C, B, A = colors
        ok = (A != B) and (A != C) and (B != C) and (B != D) and (C != D)
        if ok:
            ok_count += 1
        else:
            ng_count += 1
    else:
        ng_count += 1
print(f"OK: {ok_count}, NG: {ng_count}")
```

> 0001100001000001 A=0 B=2 C=3 D=0 True (1846 times)
> 0001010010000001 A=0 B=3 C=2 D=0 True (1816 times)
> 1000000101001000 A=3 B=2 C=0 D=3 True (1718 times)
> 1000010000011000 A=3 B=0 C=2 D=3 True (1645 times)
> 0010010010000010 A=1 B=3 C=2 D=1 True (1581 times)
> 0010100001000010 A=1 B=2 C=3 D=1 True (1560 times)
> 0010010000010010 A=1 B=0 C=2 D=1 True (1389 times)
> 0010000101000010 A=1 B=2 C=0 D=1 True (1383 times)
> 0100000100100100 A=2 B=1 C=0 D=2 True (1233 times)
> 0100001000010100 A=2 B=0 C=1 D=2 True (1178 times)
> 1000000100011000 A=3 B=0 C=0 D=3 False (890 times)
> 0100001010000100 A=2 B=3 C=1 D=2 True (842 times)
> 0100100000100100 A=2 B=1 C=3 D=2 True (809 times)
> 0001010010000010 A=1 B=3 C=2 D=0 True (784 times)
> 0010010010000001 A=0 B=3 C=2 D=1 True (782 times)
> 0001001001000001 A=0 B=2 C=1 D=0 True (762 times)
> 0001100001000010 A=1 B=2 C=3 D=0 True (752 times)
> 1000000101000010 A=1 B=2 C=0 D=3 True (748 times)
> 0010100001000001 A=0 B=2 C=3 D=1 True (747 times)
> 0001010000100001 A=0 B=1 C=2 D=0 True (736 times)
> OK: 48, NG: 200

ざっと確認すると、可能性のある $2^{16} = 65,536$ 通りの候補のうち、実際にサンプリングされたのは 248 通りで、解けているケースは 48 通りということだった。上位 20 件を見ると、ほぼ解けているケースが並んでいる。十分な結果であろう。

# 5 頂点のグラフの 4 色カラーリング問題

冒頭で触れた本命のケースであるが、こちらは内容的にはほぼ同様の実装であるので詳細は割愛する。但し、NVIDIA A100 などのハイスペックの GPU で数時間くらい回さないと良い結果にならなさそうな感じであった。

エネルギーの推移は以下のような感じであった。

![](/images/dwd-qiskit-qaoa03/006.png =500x)

サンプリング結果は以下である。

> 00100001010010000010 A=1 B=3 C=2 D=0 E=1 True (858 times)
> 00010010001010000100 A=2 B=3 C=1 D=1 E=0 True (831 times)
> 10000010001001001000 A=3 B=2 C=1 D=1 E=3 True (739 times)
> 00101000100001000010 A=1 B=2 C=3 D=3 E=1 True (736 times)
> 00011000001001000001 A=0 B=2 C=1 D=3 E=0 True (595 times)
> 10000001000101000010 A=1 B=2 C=0 D=0 E=3 True (557 times)
> 01000001100000100100 A=2 B=1 C=3 D=0 E=2 True (474 times)
> 01001000100000010010 A=1 B=0 C=3 D=3 E=2 True (423 times)
> 00010100010000101000 A=3 B=1 C=2 D=2 E=0 True (399 times)
> 10000100001000011000 A=3 B=0 C=1 D=2 E=3 True (392 times)
> 00010100001010000010 A=1 B=3 C=1 D=2 E=0 True (362 times)
> 01000001000100100100 A=2 B=1 C=0 D=0 E=2 True (353 times)
> 01000001001010000010 A=1 B=3 C=1 D=0 E=2 True (335 times)
> 00010100010000100001 A=0 B=1 C=2 D=2 E=0 True (318 times)
> 01001000100000100100 A=2 B=1 C=3 D=3 E=2 True (289 times)
> 10000010010000100001 A=0 B=1 C=2 D=1 E=3 False (284 times)
> 10000100100000011000 A=3 B=0 C=3 D=2 E=3 False (283 times)
> 00011000000100100001 A=0 B=1 C=0 D=3 E=0 False (283 times)
> 00101000000100100100 A=2 B=1 C=0 D=3 E=1 False (281 times)
> 01000010100000010100 A=2 B=0 C=3 D=1 E=2 True (276 times)
> OK: 96, NG: 881

可能性のある $2^{20} = 1,048,576$ 通りの候補のうち、実際にサンプリングされたのは 977 通りで、解けているケースは 96 通りということだった。上位 20 件を見ると、ほぼ解けているケースが並んでいる。こちらも十分な結果であろう。

# まとめ

QAOA を用いてグラフカラーリング問題を解いてみた。解けるには解けるが結構大変だし、それなりに時間もかかることが分かった。

# Appendix

## QAOA

QAOA（文献 [1]）を素朴に書こう。文献 [2] にあるような断熱定理に基礎を置いた「断熱的量子アルゴリズム」のアルゴリズムがあり、問題を記述するハミルトニアンの基底状態のエネルギーと固有状態を求めることができるようなものである。それはより厳密な時間発展を記述するために量子回路が深くなるという問題がある。これに対して、量子近似最適化アルゴリズム (QAOA) は「断熱的量子アルゴリズム」にインスパイアされた構造を持ちつつも効率をあげて近似解を求めるヒューリスティックなアルゴリズムとなる[^a]。
“問題を記述するハミルトニアン” はある種の組み合わせ最適化問題から定式化され、これらを解くのに用いられるアルゴリズムとなる。

[^a]: 但し「断熱的量子アルゴリズム」と「量子近似最適化アルゴリズム (QAOA)」とは、完全に互換性のあるアルゴリズムというわけでもないので、これについてはこれ以上は掘り下げない。

文献 [3] が色々と詳しいので、以下ではこれを大いに参考にして概略を記載する。

QAOA では 2 つのハミルトニアンとして、問題ハミルトニアン（または位相ハミルトニアン） $\hat{H}_P$ とミキシングハミルトニアン $\hat{H}_M$ を用いる。まず量子状態として $\hat{H}_M$ の適当な基底状態 $\ket{\psi_0}$ を用意する[^b]。

[^b]: 難しい場合もあるようだが、正確に基底状態に設定していない場合には QAOA のパフォーマンスに影響が出る可能性が文献 [3] で示唆されている。

$\ket{\psi (\beta, \gamma)} := U(\beta,\gamma) \ket{\psi_0}$ という状態を作る回路を組み立てることになる。ここで、パラメータ $\beta = (\beta_1,\cdots,\beta_p) \in \R^p$ と $\gamma = (\gamma_1,\cdots,\gamma_p) \in \R^p$ に対して $U(\beta, \gamma)$ は以下のようなものである[^c]:

[^c]: 気持ち的には、「断熱的量子アルゴリズム」的に書いた場合の $$\hat{H}(t) = \left(1 - \frac{t}{T}\right) \hat{H}_M + \frac{t}{T} \hat{H}_P$$ の時間発展演算子 $\exp (-i t \hat{H}(t))$ の Trotter 分解のパラメータ付け版のような構造になっている。

$$
\begin{align*}
U(\beta, \gamma) = \underbrace{\left( e^{-i \beta_p \hat{H}_M} e^{-i \gamma_p \hat{H}_P} \right) \left( e^{-i \beta_{p-1} \hat{H}_M} e^{-i \gamma_{p-1} \hat{H}_P} \right) \cdots \left( e^{-i \beta_1 \hat{H}_M} e^{-i \gamma_1 \hat{H}_P} \right)}_{p}
\end{align*}
$$

この $U(\beta, \gamma)$ を用いて、期待値

$$
\begin{align*}
f(\beta, \gamma) = \braket{\psi (\beta, \gamma) | \hat{H}_P | \psi (\beta, \gamma)}
\end{align*}
$$

を考え、適当な古典オプティマイザ（COBYLA や Powell）で以下のように最適化を行う。

$$
\begin{align*}
\beta_\text{opt}, \gamma_\text{opt} = \argmin_{\beta, \gamma} f(\beta, \gamma)
\end{align*}
$$

この時、$\psi (\beta_\text{opt}, \gamma_\text{opt})$ は $\hat{H}_P$ の基底状態になっているというのが QAOA のアウトラインである。

## ミキシングハミルトニアン

ミキシングハミルトニアン $\hat{H}_M$ とは初期状態 $\ket{\psi_0}$ を基底状態に持つような使いやすいエルミート演算子ということになる。

**$X$ ミキサー**

QAOA の原典である文献 [1] では Pauli-$X$ ゲートを用いた、いわゆる「$X$ ミキサー」が導入された（文献 [1] Eq. (3)）。$X$ ゲートは量子計算のコンテキストでは

$$
\begin{align*}
X = \begin{pmatrix}
0 & 1 \\
1 & 0
\end{pmatrix}
\end{align*}
$$

と書かれるものであり、簡単な計算で固有状態として $\ket{+} := \frac{1}{\sqrt{2}} (\ket{0} + \ket{1}) = H \ket{0}$ を持つことが分かる（文献 [1] Eq. (5)）。ここで $H$ はアダマールゲート

$$
\begin{align*}
H = \frac{1}{\sqrt{2}} \begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix}
\end{align*}
$$

である。$X$ の時間発展演算子は $\exp (- i \frac{\theta}{2} X) =: RX (\theta)$ であり、いわゆる $RX$ で記述される（文献 [1] Eq. (4)）。

このため、QAOA のチュートリアル等では、まず回路の冒頭で一様にアダマールゲート $H$ を適用し、途中で横断的に $RX(\beta_i)$ を差し込む形になる[^d]。

[^d]: ところで計算すると分かるが、$X$ の固有値は $\{-1, 1\}$ であり、$-1$ に対応する固有ベクトルは $\ket{-} := \frac{1}{\sqrt{2}} (\ket{0} - \ket{1}) = ZH \ket{0}$ である。ここで、$Z$ は Pauli-$Z$ ゲート $$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$ である。これでは話がすっきりしないように感じられるかもしれないが、$\beta_i^\prime = -(2 \pi + \beta_i)$ とおくと、$RX(\beta_i^\prime) = -\exp (- i \frac{\beta_i}{2} (-X)) = -R(-X)(\beta_i)$ という関係が成り立つことが分かる。$p = 2 p^\prime$ の時には $(-1)^{p} = 1$ に注意すると、$$U(\beta, \gamma) = \underbrace{\left( e^{-i \beta_p^\prime (-X)} e^{-i \gamma_p \hat{H}_P} \right) \left( e^{-i \beta_{p-1}^\prime (-X)} e^{-i \gamma_{p-1} \hat{H}_P} \right) \cdots \left( e^{-i \beta_1^\prime (-X)} e^{-i \gamma_1 \hat{H}_P} \right)}_{p=2p^\prime}$$ と書けるので、“$-X$ ミキサー” を使っていると考えてもそれほど悪くはないであろう。$\ket{+}$ は確かに $-X$ の基底状態であるのだから。

**$XY$ ミキサー**

ミキシングハミルトニアンは $X$ ミキサー以外にも提案されており、$XY$ ミキサー（文献 [4]）や $RS$ ミキサーなどが提案されている。文献 [3] に加え、文献 [5] も参考になると思われる。

$XY$ ミキサーは大雑把には、2 量子ビットの場合だと $\frac{1}{\sqrt{2}} (\ket{01} + \ket{10})$ の状態を維持させるために用いられる。詳しい説明は [blueqatでXYミキサーを用いた制約付きQAOA](https://qiita.com/ryuNagai/items/1836601f4d3c5ec9e336) などにある。

文献 [3] では $XY$ ミキサーにも何種類かあることが書かれており、ring-$XY$ ミキサーや complete-$XY$ ミキサーがあるということになる。今回の記事では complete-$XY$ ミキサーを用いた。complete-$XY$ ミキサーとは

$$
\hat{H}_{S_\text{complete}}^{XY} = \sum_{(i,j) \in S_\text{complete}} (X_i X_j + Y_i Y_j), \\
\text{where} \ \ S_\text{complete} = \left\{ (i,j) | i < j; i, j \in \{1, \ldots, N\} \right\}
$$

である。ここで $Y$ は Pauli-$Y$ ゲートである。

さて、2 量子ビットの $\frac{1}{\sqrt{2}} (\ket{01} + \ket{10})$ に加え、3 量子ビットの場合には $\frac{1}{\sqrt{3}} (\ket{001} + \ket{010} + \ket{100})$、4 量子ビットの場合には $\frac{1}{2} (\ket{0001} + \ket{0010} + \ket{0100} + \ket{1000})$ を維持できるのであればワンホット制約は自動的に満たされるのである。このような状態を「Hamming 重み 1 の Dicke 状態」と呼ぶが、これらの状態を準備する方法が文献 [6] で知られている。簡単には論文の Figure 2 を実装すれば良いということになる。

状況を一言で書くと、「complete-$XY$ ミキサーは Hamming 重み 1 の Dicke 状態を固有状態に持つ」となる[^e]。

[^e]: 実際には、今回も complete-$XY$ ミキサーの基底状態というよりは最大固有値に対応する固有状態なのだが、パラメータをいじることで実質問題ないと思われる。

## 参考文献

[1] E. Farhi, J. Goldstone, and S. Gutmann, A quantum approximate optimization algorithm, Preprint at [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
[2] Edward Farhi, Jeffrey Goldstone, Sam Gutmann, Michael Sipser. Quantum computation by adiabatic evolution, 2000. [arXiv:quant-ph/0001106](https://arxiv.org/abs/quant-ph/0001106).
[3] Zichang He, Ruslan Shaydulin, Shouvanik Chakrabarti, Dylan Herman, Changhao Li, Yue Sun, Marco Pistoia. Alignment between Initial State and Mixer Improves QAOA Performance for Constrained Optimization, Preprint at [arXiv:2305.03857](https://arxiv.org/abs/2305.03857)
[4] Zhihui Wang, Nicholas C. Rubin, Jason M. Dominy, Eleanor G. Rieffel. $XY$-mixers: analytical and numerical results for QAOA, Preprint at [arXiv:1904.09314](https://arxiv.org/abs/1904.09314)
[5] Wenyang Qian, Robert A. M. Basili, Mary Eshaghian-Wilner, Ashfaq Khokhar, Glenn Luecke, James P. Vary. Comparative study of variations in quantum approximate optimization algorithms for the Traveling Salesman Problem, Preprint at [arXiv:2307.07243](https://arxiv.org/abs/2307.07243)
[6] Andreas Bärtschi, Stephan Eidenbenz. Deterministic Preparation of Dicke States, Preprint at [arXiv:1904.07358](https://arxiv.org/abs/1904.07358)
