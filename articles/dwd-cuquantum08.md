---
title: "cuQuantum で遊んでみる (8) — QAOA の期待値計算高速化"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "Python"]
published: false
---

# 目的

[cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) で QAOA の計算を `cuQuantum` の `cuTensorNet` で行ったがあまり速くなかった。

これについて、[cuQuantum で遊んでみる (7) — 期待値計算再考](/derwind/articles/dwd-cuquantum07) で QAOA の類のハミルトニアンのテンソルネットワークへの効率的な埋め込みについて考察した。

この上で更に GPU 計算への配慮をした実装をすると 2 倍以上の高速化ができるのでこれについて簡単にまとめる。実際は更に工夫すると 3 倍以上に高速化ができるが、コードも 3 倍汚くなるので本記事では 2 倍くらいで止めておく。

# 問題設定

[Maximum cut](https://en.wikipedia.org/wiki/Maximum_cut) のようなグラフ問題を解くことであった。

# テンソルネットワーク実装

必要なモジュールを import する。

```python
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import cupy as cp
from cuquantum import CircuitToEinsum, contract
```

## 必要なユーティリティの定義

以下は高速化だけを想定して書いたユーティリティなのでかなりわかりにくい。基本的に

- GPU メモリを使いまわす
- メモリを最小限で書き換える
- 同じ結果になる計算は二度行わない
- 計算結果をキャッシュする

ことを前提にしている。

```python
# cuTensorNet のテンソルネットワークの Rx ゲート部分を上書きする
# <ψ|H|ψ> の構造なので、Rx の反対側に Rx† があるので同時に書き換える。
def Rx_Rxdag(
    theta: float, mat: cp.ndarray, mat_dag: cp.ndarray
) -> None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)

    mat[0][0] = mat[1][1] = mat_dag[0][0] = mat_dag[1][1] = cos
    mat[0][1] = mat[1][0] = -sin * 1.0j
    mat_dag[0][1] = mat_dag[1][0] = sin * 1.0j
    return None

# cuTensorNet のテンソルネットワークの Rx ゲート部分を上書きする
# <ψ|H|ψ> の構造なので、Rz の反対側に Rz† があるので同時に書き換える。
def Rz_Rzdag(
    theta: float, mat: cp.ndarray, mat_dag: cp.ndarray
) -> None:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)

    mat[0][0] = mat_dag[1][1] = cos - sin * 1.0j
    mat_dag[0][0] = mat[1][1] = cos + sin * 1.0j
    mat[0][1] = mat[1][0] = mat_dag[0][1] = mat_dag[1][0] = 0
    return None

# Rx と Rz の行列を指定の角度で作成する。
def Rx_Rz(
    theta: float
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    cos = np.cos(theta / 2)
    sin = np.sin(theta / 2)

    rx = cp.array([
        [cos, -sin * 1.0j],
        [-sin * 1.0j, cos]
    ], dtype=complex)

    rz = cp.array([
        [cos - sin * 1.0j, 0],
        [0, cos + sin * 1.0j]
    ], dtype=complex)

    return rx, rz

# プレースホルダーの量子回路からプレースホルダーのテンソルネットワークを作成する。
def circuit_to_einsum_expectation(
    qc: QuantumCircuit, hamiltonian: str
) -> tuple[str, list[cp.ndarray], dict[str, tuple[list[int], list[int], Pauli]]]:
    length = len(qc.parameters)
    eps = 0.01
    # ダミーのパラメータを割り当てておいて
    params = np.arange(eps, np.pi - eps, (np.pi - 2 * eps) / length)[:length]
    name2param = {pvec.name: p for pvec, p in zip(qc.parameters, params)}
    qc = qc.bind_parameters(dict(zip(qc.parameters, params)))
    converter = CircuitToEinsum(qc)
    # ダミーのパラメータベースでテンソルネットワークを作成する。
    expr, operands = converter.expectation(hamiltonian)

    # ダミーのパラメータの位置とゲート種別をテンソルネットワークの中から探す。

    # Qiskit のパラメータ名から (ゲートの位置, 共役ゲートの位置, 上書き関数) への辞書
    pname2locs: dict[str, tuple[list[int], list[int], Callable]] = {}
    for name, p in name2param.items():
        rx, rz  = Rx_Rz(p)

        locs: list[int] = []
        dag_locs: list[int] = []
        make_paulis: Callable = None
        for i, op in enumerate(operands):
            # 半分より先は既に見つけたやつの共役行列があるのでいちいち探さない。
            if i >= len(operands) / 2:
                break

            if cp.allclose(op, rx):
                locs.append(i)
                dag_locs.append(len(operands) - i - 1)
                make_paulis = Rx_Rxdag
            elif cp.allclose(op, rz):
                locs.append(i)
                dag_locs.append(len(operands) - i - 1)
                make_paulis = Rz_Rzdag
        if locs and dag_locs:
            pname2locs[name] = (locs, dag_locs, make_paulis)
    return expr, operands, pname2locs

# 指定のゲート位置の行列を上書きする
def replace_pauli(
    operands: list[cp.ndarray],  # テンソルネットワーク
    pname2theta: dict[str, float],  # パラメータ名に対する新しいパラメータ値
    pname2locs: dict[str, tuple[list[int], list[int], Callable]],
) -> list[cp.ndarray]:
    for pname, theta in pname2theta.items():
        if pname not in pname2locs:
            continue
        locs, dag_locs, make_paulis = pname2locs[pname]
        for loc, dag_loc in zip(locs, dag_locs):
            make_paulis(theta, operands[loc], operands[dag_loc])

    return operands

# ダミーのハミルトニアン ZZZZZ の位置を見つける
def find_dummy_hamiltonian(operands: list[cp.ndarray]) -> list[int]:
    locs = []
    for i, op in enumerate(operands):
        if cp.all(op == Z):
            locs.append(i)
    return locs
```

ここまでで準備ができたので、実際に QAOA を解いていく。

## ansatz

QAOA の ansatz を作る:

```python
n_qubits = 5
n_reps = 3

def rzz(qc, theta, qubit1, qubit2):
    qc.cx(qubit1, qubit2)
    qc.rz(theta, qubit2)
    qc.cx(qubit1, qubit2)

beta = ParameterVector("β", n_qubits * n_reps)
gamma = ParameterVector("γ", 6 * n_reps)
beta_idx = iter(range(n_qubits * n_reps))
bi = lambda: next(beta_idx)
gamma_idx = iter(range(6 * n_reps))
gi = lambda: next(gamma_idx)

qc = QuantumCircuit(n_qubits)
qc.h(qc.qregs[0][:])
for _ in range(n_reps):
    rzz(qc, gamma[gi()], 0, 1)
    rzz(qc, gamma[gi()], 0, 2)
    rzz(qc, gamma[gi()], 1, 3)
    rzz(qc, gamma[gi()], 2, 3)
    rzz(qc, gamma[gi()], 2, 4)
    rzz(qc, gamma[gi()], 3, 4)
    qc.barrier()
    for i in range(n_qubits):
        qc.rx(beta[bi()], i)

qc.draw(fold=-1)
```

![](/images/dwd-cuquantum06/001.png)

## テンソルネットワークの構築

```python
I = cp.eye(2, dtype=complex)
Z = cp.array([
    [1, 0],
    [0, -1]
], dtype=complex)

# cuQuantum で遊んでみる (7) — 期待値計算再考
# で触れた横に積み上げたハミルトニアン
hamiltonian = [
    cp.array([Z, Z, I, I, I, I]),
    cp.array([Z, I, Z, I, I, I]),
    cp.array([I, Z, I, Z, Z, I]),
    cp.array([I, I, Z, Z, I, Z]),
    cp.array([I, I, I, I, Z, Z]),
]

# ダミーのハミルトニアン ZZZZZ で一旦テンソルネットワークを作って、
dummy_hamiltonian = "Z" * n_qubits
expr, operands, pname2locs = circuit_to_einsum_expectation(qc, dummy_hamiltonian)
# どこに埋め込まれたかを探して、
hamiltonian_locs = find_dummy_hamiltonian(operands)
# ハミルトニアンを差し替える。
for ham, locs in zip(hamiltonian, hamiltonian_locs):
    operands[locs] = ham

# cuQuantum で遊んでみる (7) — 期待値計算再考
# の方法で expr のハミルトニアンの位置の添え字を更新する。
# 「g」だとぶつかるのでここでは「ξ」にしている。
es = expr.split("->")[0].split(",")
for loc in hamiltonian_locs:
    es[loc] = "ξ" + es[loc]
expr = ",".join(es) + "->"
```

## コスト関数の定義

```python
# Qiskit のパラメータ名
param_names = [p.name for p in qc.parameters]

def comnpute_expectation_tn(params, *args):
    expr, operands, pname2locs = args
    energy = 0.

    pname2theta = dict(zip(param_names, params))
    # パラメータを更新して、
    parameterized_operands = replace_pauli(operands, pname2theta, pname2locs)
    # 縮約計算を実行する。
    return cp.asnumpy(contract(expr, *parameterized_operands).real)
```

# QAOA 実行

COBYLA で適当な回数まわす。

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
> opt value=-3.993
> CPU times: user 1min, sys: 1.65 s, total: 1min 2s
> Wall time: 1min 2s

結果を比較すると [cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) では

> Wall time: 2min 14s

だったのでかなりマシになったが、コードの汚さは半端なくなってしまった。

サンプリングしてどういう状態が得られているのかも確認しよう。ここで、`AerSimulator` もテンソルネットワークシミュレータを設定している。このくらいの規模なら問題ないのだが、40 量子ビット超の QAOA を実行すると Jupyter がクラッシュすることがあったので、予めテンソルネットワークで実行するようにしておく。

```python
from qiskit_aer import AerSimulator


opt_qc = qc.bind_parameters(result.x)
opt_qc.measure_all()

sim = AerSimulator(device="GPU", method="tensor_network")
counts = sim.run(opt_qc).result().get_counts()
for k, n in sorted(counts.items(), key=lambda k_v: -k_v[1]):
    if n < 100:
        continue
    print(k[::-1], n)
```

> 10011 388
> 01100 345
> 01101 145
> 10010 143

前々回の [cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) では

> 01101 336
> 10010 332
> 10011 178
> 01100 173

だったので、同等の結果が得られているのではないだろうか。

# まとめ

期待値計算の際の `contract` 呼び出しを 1 回に抑えることで、計算時間を短縮できた。更にコードを汚くすると、今のところベストで 39.8 s まで短縮できているがとても汚くて記事には不向きである。

元々は CPU 計算で 3.5 s で済んでいたものが、GPU 計算で、なおかつ高速化しても 1min 2s もかかっているので一見まったく意味がない。確かにこのサイズではまったく意味がない。

但し、量子ビット数をスケールした時に話が変わってくる。CPU の状態ベクトルシミュレータで計算すると、概ね 30 量子ビット前後でメモリを 16GB～64GB くらい積んだ普通の一般向け PC だとメモリが枯渇する。それにも拘わらず、

- テンソルネットワークシミュレーションだと実行可能で、
- たった 6GB 程度の GPU メモリでも 40 量子ビット以上のシミュレーションが動かせて、
- テンソルを上手く組み立てると、GPU の得意な領域なのでリーズナブルな時間でシミュレーションが完了する

というご利益がある[^1]。

[^1]: 勿論回路の形態がテンソルネットワークに向いている場合の話で、大変不向きな回路の場合にはその限りではない。

これでいままでよりは大規模なシミュレーションができるようになったと思うし、今後量子ビット数をスケールした場合に考えていかなければならない部分も見えてきそうである。
