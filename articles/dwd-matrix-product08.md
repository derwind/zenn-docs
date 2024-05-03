---
title: "行列積状態について考える (8) — Vidal の標準形 その 2"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "Qiskit"]
published: true
---

# 目的

「Vidal の標準形」と呼ばれる MPS (Matrix Product State; 行列積状態) の一つの表示形式について、[行列積状態について考える (6) — Vidal の標準形](/derwind/articles/dwd-matrix-product06) で考察したが、今回再考察したい。

現時点での問題点としては、

- Qiskit Aer の実装を移植しただけであまりよく分かっていないこと
- 内容的に [行列積状態について考える (3)](https://zenn.dev/derwind/articles/dwd-matrix-product03) で実装した `TT_SVD` と近そうだが、実装上の対称性がイマイチないこと

があり、これを解消したい。

# 実装

必要なモジュールを import する。

```python
from __future__ import annotations
from typing import Sequence
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.random import random_statevector
from qiskit_aer import AerSimulator
```

[The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477) の 4.6. Notations and conversions に実装方法が載っているので、これを前回の `TT_SVD` に被せる形で見てくれを似せた状態で再実装する。結合次元を与えることで情報圧縮もできるはずだが、いま必要ないので最大の結合次元を計算して使う:

```python
def TT_SVD_Vidal(
    C: np.ndarray, num_qubits: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """TT_SVD Vidal algorithm

    Args:
        C (np.ndarray): n-dimensional input tensor
        num_qubits (int | None): number of qubits

    Returns:
        list[np.ndarray]: Γs
        list[np.ndarray]: Λs
    """

    gammas = []
    lambdas = []

    if num_qubits is None:
        num_qubits = int(np.log2(np.prod(C.shape)))
    dims = (2,) * num_qubits
    C = C.reshape(dims)

    # 最大の結合次元の計算 c.f. TT_SVD
    r = []
    for sep in range(1, num_qubits):
        row_dim = np.prod(dims[:sep])
        col_dim = np.prod(dims[sep:])
        rank = np.linalg.matrix_rank(C.reshape(row_dim, col_dim))
        r.append(rank)

    for i in range(num_qubits - 1):
        if i == 0:
            ri_1 = 1
        else:
            ri_1 = r[i - 1]
        ri = r[i]
        C = C.reshape(ri_1 * dims[i], np.prod(dims[i + 1:]))
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        U = U[:, :ri]
        S = S[:ri]
        Vh = Vh[:ri, :]
        U = U.reshape(ri_1, dims[i], ri)
        if i > 0:
            for a in range(U.shape[0]):
                U[a, :, :] /= lambdas[-1][a]  # Eq. (161)
        gammas.append(U)
        lambdas.append(S)
        C = np.diag(S) @ Vh
    gammas.append(Vh)
    gammas[0] = gammas[0].reshape(dims[0], r[0])
    return gammas, lambdas
```

式 (161) の処理以外は `TT_SVD` (TT-分解) と本質的には同じ実装ということになる。

縮約計算用の添え字を作成する関数も定義する。一例として `"Aa,a,aBb,b,bC->ABC"` のようなものを作ることになる[^1]:

[^1]: Qiskit Aer の実装では `"Aa,a,Bab,b,Cb->ABC"` に相当するので、`TT_SVD_Vidal` に `numpy.transpose` 相当の実装が追加で必要になる。

```python
def make_expr(n_qubits: int) -> str:
    outer_indices = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    inner_indices = [chr(i) for i in range(ord("a"), ord("z") + 1)]

    expr = []
    prev_inner = ""
    for i, (outer_i, inner_i) in enumerate(zip(outer_indices, inner_indices)):
        if i + 1 < n_qubits:
            expr.extend([f"{prev_inner}{outer_i}{inner_i}", inner_i])
            prev_inner = inner_i
        else:
            expr.extend([f"{prev_inner}{outer_i}"])
            break
    return ",".join(expr) + "->" + "".join(outer_indices[:n_qubits])
```

MPS シミュレーションのデモ用に 1 量子ビットゲートを Vidal 標準形に適用するための関数を定義する。実装は [Efficient classical simulation of slightly entangled quantum computations](https://arxiv.org/abs/quant-ph/0301063) の Lemma 1 を用いる。

```python
def apply_one_qubit_gate(
    gammas: list[np.ndarray], U: np.ndarray, qubit: int
) -> None:
    gamma = gammas[qubit]
    if qubit == 0:  # expr: ia
        gamma = np.einsum("ij,ja->ia", U, gamma)
    elif qubit + 1 >= len(gammas):  # expr: ai
        gamma = np.einsum("ij,aj->ai", U, gamma)
    else:  # expr: aib
        gamma = np.einsum("ij,ajb->aib", U, gamma)
    gammas[qubit][:] = gamma


PauliX = np.array([[0, 1], [1, 0]], dtype=float)
PauliZ = np.array([[1, 0], [0, -1]], dtype=float)
Hadamard = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)


def apply_X(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliX, qubit)


def apply_Z(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliZ, qubit)


def apply_H(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, Hadamard, qubit)
```

# MPS シミュレーション

3 量子ビット回路の各量子ビットに `ZH` を適用して、
$\frac{1}{2\sqrt{2}} (\ket{0} - \ket{1}) \otimes (\ket{0} - \ket{1}) \otimes (\ket{0} - \ket{1})$ を作りたい。

```python
ket_ZERO = np.array([1, 0], dtype=float)
ket_ONE = np.array([0, 1], dtype=float)
state_minus = (ket_ZERO - ket_ONE) / np.sqrt(2)

state_000 = np.kron(np.kron(ket_ZERO, ket_ZERO), ket_ZERO)
state_mmm = np.kron(np.kron(state_minus, state_minus), state_minus)


num_qubits = 3
gammas, lambdas = TT_SVD_Vidal(state_000)
lambdas.append(None)

operands = []
for gamma, lam in zip(gammas, lambdas):
    operands.append(gamma)
    if lam is not None:
        operands.append(lam)
expr = make_expr(num_qubits)

for i in range(num_qubits):
    apply_H(gammas, i)
    apply_Z(gammas, i)
tensor = np.einsum(expr, *operands).flatten()

print(f"{state_mmm=}")
print(f"{tensor=}")
print(np.allclose(tensor, state_mmm))
```

> state_mmm=array([ 0.35355339, -0.35355339, -0.35355339,  0.35355339, -0.35355339,  0.35355339,  0.35355339, -0.35355339])
> tensor=array([ 0.35355339, -0.35355339, -0.35355339,  0.35355339, -0.35355339,  0.35355339,  0.35355339, -0.35355339])
> True

状態ベクトルシミュレーションの場合、$2^3 = 8$ であるので、$8 \times 8$ 行列を掛けこまないとならない計算が、標的となる量子ビットに対応するテンソルに対してサイズ $2 \times 2$ のテンソル (行列) を適用するだけで時間発展を実行できた。

# まとめ

- `TT_SVD_Vidal` の実装を `TT_SVD` と対称性を持たすことができた
- 簡単なケースのみだが、MPS シミュレーションを実装できた

# 参考文献

[O] [I. V. Oseledets, Tensor-Train Decomposition, SIAM J. Sci. Comput., 33(5), 2295–2317. (23 pages), 2011.](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition)
[S] [Ulrich Schollwoeck, The density-matrix renormalization group in the age of matrix product states, arXiv:1008.3477, 2010.](https://arxiv.org/abs/1008.3477)
[V] [Guifre Vidal, Efficient classical simulation of slightly entangled quantum computations, arXiv:quant-ph/0301063, 2003.](https://arxiv.org/abs/quant-ph/0301063)
