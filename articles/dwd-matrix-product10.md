---
title: "行列積状態について考える (10) — 50 量子ビットの期待値計算"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "Qiskit", "TensorNetwork", "cuQuantum"]
published: false
---

# 目的

[行列積状態について考える (9) — 100 量子ビットのもつれ状態](/derwind/articles/dwd-matrix-product09) で大量の量子ビットの行列積状態 (MPS) のもつれ状態を見た。

テンソルネットワーク計算で量子状態を求める場合、係数である波動関数ごとに計算する必要があり、すべての波動関数を計算しようとすると (量子ビット数について) 指数関数的な計算量になる。

ところがハミルトニアンの期待値計算の場合、実はたった 1 つの波動関数を求めれば良いので MPS の恩恵を得ることができる。今回はこれを見たい。

- Qiskit Aer
- cuTensorNet
- 自前実装

の 3 パターン実装して確認してみたい。

# 期待値計算

量子状態 $\ket{0}^{\otimes n}$ を単に $\ket{\mathbf{0}}$ と書くことにし、何かしらのユニタリゲートの連鎖によって $\ket{\psi} = U \ket{\mathbf{0}}$ になるとしよう。この時、ハミルトニアン $H$ の期待値は $\braket{H} = \braket{\psi | H | \psi}$ で与えられるのであった。

これをもとに以下のような状態を考える。

$$
\begin{align*}
\ket{\Psi} := U^\dagger H U \ket{\mathbf{0}} = \sum_{i_1,i_2, \cdots, i_d} \Psi_{i_1 i_2 \cdots i_d} \ket{i_1 i_2 \cdots i_d}
\end{align*}
$$

すると、$\braket{H} = \braket{\mathbf{0}| \Psi} = \Psi_{0 0 \cdots 0}$ であることが分かる。

[行列積状態について考える (6) — Vidal の標準形](/derwind/articles/dwd-matrix-product06) を思い出すと、以下の波動関数を求めることになる。

$$
\begin{align*}
\Psi_{0 0 \cdots 0} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} \Gamma^{[1] 0}_{\alpha_1} \lambda^{[1]}_{\alpha_1} \Gamma^{[2] 0}_{\alpha_1 \alpha_2} \lambda^{[2]}_{\alpha_2} \cdots \Gamma^{[d] 0}_{\alpha_{d-1}}
\end{align*}
$$

そういうわけで早速やってみよう。ハミルトニアンは何でも良いのだが、今回はありがちは $ZZ$ ハミルトニアンの形にしてみる。組み合わせ最適化などでよく見かけるやつだ。

# 実装

## Qiskit Aer

まずは必要なモジュールを import する。

```python
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
```

そして以下の回路を実行する。

```python
num_qubits = 50

qc = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    theta = (3/2)*math.pi/num_qubits * (i+1)
    qc.ry(theta, i)
for i in range(num_qubits-1):
    qc.cx(i, i+1)
qc.save_expectation_value(SparsePauliOp(["ZZ"]), qubits=[8, 10])
sim = AerSimulator(method="matrix_product_state")
expval = sim.run(qc, shots=1).result().data()["expectation_value"]
print(expval)
```

> 0.2992070369841501

これは一瞬で終わる。MPS シミュレータなので 50 量子ビットでもまったく平気なわけである。状態ベクトルシミュレーションなら本記事を書きながら手元の PC で気軽にシミュレーションなどできない。

## cuTensorNet

### Qiskit Aer 経由

まずは Qiskit Aer を経由して cuTensorNet を使ってみよう。以下を実行する。

```python
num_qubits = 50

qc = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    theta = (3/2)*math.pi/num_qubits * (i+1)
    qc.ry(theta, i)
for i in range(num_qubits-1):
    qc.cx(i, i+1)
qc.save_expectation_value(SparsePauliOp(["ZZ"]), qubits=[8, 10])
sim = AerSimulator(device="GPU", method="tensor_network")
expval = sim.run(qc, shots=1).result().data()["expectation_value"]
print(expval)
```

> 0.29920703698414985

これも GPU の初期化を除けば一瞬である。

### CircuitToEinsum 経由

よりダイレクトな API の使い方をしてみよう。必要なモジュールを追加で import する。

```python
from cuquantum import CircuitToEinsum
from cuquantum import contract as cuq_contract
```

そして以下を実行する[^1]。

[^1]: 注意として Qiskit のバージョンは現時点では `"qiskit==0.44.1"` までにしておかないと cuQuantum との嚙み合わせで問題が出る。c.f. [CircuitToEinsum fails for some qiskit QuantumCircuit](https://github.com/NVIDIA/cuQuantum/issues/99)

```python
num_qubits = 50

qc = QuantumCircuit(num_qubits)
for i in range(num_qubits):
    theta = (3/2)*math.pi/num_qubits * (i+1)
    qc.ry(theta, i)
for i in range(num_qubits-1):
    qc.cx(i, i+1)

converter = CircuitToEinsum(qc)
hamiltonian = "I"*8 + "ZIZ" + "I"*(num_qubits-11)
expr, operands = converter.expectation(hamiltonian)

expval = contract(expr, *operands).real
print(expval)
```

> 0.2992070369841499

## 自前実装

ここまで非常に良い感じで計算できている。最後に自前実装でも確認してみよう。結論としては、上記のライブラリの計算と一致するので、以下のような内容を自前で実装するようなことはせずに、素直にライブラリの API を叩くのが正しい。以下では学習的な観点で実装した形になる。

実装の大部分は [行列積状態について考える (9) — 100 量子ビットのもつれ状態](/derwind/articles/dwd-matrix-product09) をそのまま使うのだが、以下の変更を加えている:

- `make_expr` を 50 量子ビットまでいける実装に拡張。
- 1 量子ビットゲートの種類も追加。
- `dtype=complex` に変更。

まずは必要なモジュールを追加で import する。

```python
from opt_einsum import contract as oe_contract
```

以下が、MPS 関連の実装になる。

```python
def TT_SVD_Vidal(
    C: np.ndarray, num_qubits: int | None = None, dims: tuple[int] = None
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
    if num_qubits < 2:
        raise ValueError(f"num_qubits ({num_qubits}) must be larger than one.")

    if dims is None:
        dims = (2,) * num_qubits
    C = C.reshape(dims)

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
                U[a, :, :] /= lambdas[-1][a]
        gammas.append(U)
        lambdas.append(S)
        C = np.diag(S) @ Vh
    gammas.append(Vh)
    gammas[0] = gammas[0].reshape(dims[0], r[0])
    return gammas, lambdas

alphabets = [chr(uni)
    for uni in list(range(ord("A"), ord("Z")+1)) + list(range(ord("a"), ord("z")+1))]
greek_alphabets = [chr(uni)
    for uni in list(range(ord("Α"), ord("Ω")+1)) + list(range(ord("α"), ord("ω")+1))]

def make_expr(n_qubits: int) -> str:
    outer_indices = alphabets
    inner_indices = greek_alphabets

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


PauliX = np.array([[0, 1], [1, 0]], dtype=complex)
PauliY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PauliZ = np.array([[1, 0], [0, -1]], dtype=complex)
Hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def Rx(theta: float):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2) * 1j],
        [-np.sin(theta/2) * 1j, np.cos(theta/2)]
    ], dtype=complex)


def Ry(theta: float):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex)


def Rz(theta: float):
    return np.array([
        [np.cos(theta/2) - np.sin(theta/2), 0],
        [0, np.cos(theta/2) + np.sin(theta/2)]
    ], dtype=complex)


def apply_X(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliX, qubit)


def apply_Y(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliY, qubit)


def apply_Z(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliZ, qubit)


def apply_H(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, Hadamard, qubit)


def apply_Rx(gammas: list[np.ndarray], theta: float, qubit: int) -> None:
    apply_one_qubit_gate(gammas, Rx(theta), qubit)


def apply_Ry(gammas: list[np.ndarray], theta: float, qubit: int) -> None:
    apply_one_qubit_gate(gammas, Ry(theta), qubit)


def apply_Rz(gammas: list[np.ndarray], theta: float, qubit: int) -> None:
    apply_one_qubit_gate(gammas, Rz(theta), qubit)


def swap_ij_of_controlled_gate(mat: np.ndarray) -> np.ndarray:
    mat2 = np.zeros_like(mat)
    mat2[0, 0] = mat2[2, 2] = 1
    mat2[1, 1] = mat[2, 2]
    mat2[1, 3] = mat[2, 3]
    mat2[3, 1] = mat[3, 2]
    mat2[3, 3] = mat[3, 3]
    return mat2


def apply_two_qubits_gate(
    gammas: list[np.ndarray], lambdas: list[np.ndarray],
    U: np.ndarray, control: int, target: int
) -> None:
    U2 = swap_ij_of_controlled_gate(U).reshape(2, 2, 2, 2)
    U = U.reshape(2, 2, 2, 2)

    i, j = control, target

    if i+1 != j and j+1 != i:
        raise ValueError(f"only adjuscent qubits are supported.")

    reverse = False
    if j < i:
        i, j = j, i
        reverse = True

    if i == 0:
        if len(gammas) == 2:
            expr = "IJAB,Aa,a,aB->IJ"
        else:
            expr = "IJAB,Aa,a,aBb->IJb"
        left_dim = gammas[i].shape[0]
    elif j + 1 < len(gammas):
        expr = "IJAB,aAb,b,bBc->aIJc"
        left_dim = gammas[i].shape[0] * gammas[i].shape[1]
    else:
        expr = "IJAB,aAb,b,bB->aIJ"
        left_dim = gammas[i].shape[0] * gammas[i].shape[1]

    if not reverse:
        C = np.einsum(expr, U, gammas[i], lambdas[i], gammas[j])
    else:
        C = np.einsum(expr, U2, gammas[i], lambdas[i], gammas[j])

    updated_gammas, updated_lambdas = TT_SVD_Vidal(
        C, num_qubits=2, dims=(left_dim, -1)
    )
    if i > 0:
        updated_gammas[0] = updated_gammas[0].reshape(-1, 2, len(updated_lambdas[0]))
    if j + 1 < len(gammas):
        updated_gammas[1] = updated_gammas[1].reshape(len(updated_lambdas[0]), 2, -1)
    else:
        updated_gammas[1] = updated_gammas[1].reshape(-1, 2)
    gammas[i] = updated_gammas[0]
    lambdas[i] = updated_lambdas[0]
    gammas[i+1] = updated_gammas[1]


CX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=float)


CY = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, -1j],
    [0, 0, 1j, 0]
], dtype=complex)


CZ = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
], dtype=float)


def apply_CX(gammas, lambdas, i, j):
    apply_two_qubits_gate(gammas, lambdas, CX, i, j)

def apply_CY(gammas, lambdas, i, j):
    apply_two_qubits_gate(gammas, lambdas, CY, i, j)

def apply_CZ(gammas, lambdas, i, j):
    apply_two_qubits_gate(gammas, lambdas, CZ, i, j)
```

また、雑多なユーティリティとして以下を用いる。

```python
def zeros_mps(n: int):
    gammas = [np.array([[1.], [0.]], dtype=complex)]
    lambdas = [np.array([1.], dtype=complex)]
    for _ in range(1, n - 1):
        gammas.append(np.array([[[1.], [0.]]], dtype=complex))
        lambdas.append(np.array([1.], dtype=complex))
    gammas.append(np.array([[1., 0.]], dtype=complex))
    return gammas, lambdas


def remove_outer_indices(expr: str):
    new_expr = []
    for v in expr.split("->")[0].split(","):
        for c in alphabets:
            v = v.replace(c, "")
        new_expr.append(v)
    return ",".join(new_expr) + "->"
```

ここまでで準備ができたので、$\ket{\Psi} := U^\dagger H U \ket{\mathbf{0}}$ に対応する MPS を作る。

```python
num_qubits = 50
gammas, lambdas = zeros_mps(num_qubits)
for i in range(num_qubits):
    theta = (3/2)*math.pi/num_qubits * (i+1)
    apply_Ry(gammas, theta, i)
for i in range(num_qubits-1):
    apply_CX(gammas, lambdas, i, i+1)
apply_Z(gammas, 8)
apply_Z(gammas, 10)
for i in reversed(range(num_qubits-1)):
    apply_CX(gammas, lambdas, i, i+1)
for i in reversed(range(num_qubits)):
    theta = -(3/2)*math.pi/num_qubits * (i+1)
    apply_Ry(gammas, theta, i)

operands = [gammas[0][0, :]]
for lam, gamma in zip(lambdas[:-1], gammas[1:-1]):
    operands.append(lam)
    operands.append(gamma[:, 0, :])
operands.append(lambdas[-1])
operands.append(gammas[-1][:, 0])
```

注意として、$\Psi_{0 0 \cdots 0}$ しか求めないので予めインデックスを落としている。これに対応させて縮約のインデックス側も削っておく。

```python
expr = remove_outer_indices(make_expr(num_qubits))
```

後は縮約を計算して $\Psi_{0 0 \cdots 0}$ を求めると以下のようになる。

```python
expval = oe_contract(expr, *operands).real
print(expval)
```

> 0.2992070369841477

ここまでのシミュレーションの計算結果は、すべて `0.29920703698` まで一致している。

# まとめ

各種の方法でハミルトニアン $ZZ$ の期待値を計算してみた。MPS のもつれ量にもよるのだが、今回のケースは実はもつれ量がとても小さく、MPS 計算が捗る形になっている。よって期待値計算が一瞬で終わってしまった。

ハミルトニアンの期待値の計算は量子機械学習ではよく使われるものであるので、Barren Plateau の具合の確認を含め、大規模な回路での動きを見る上でこういった計算が役に立つかもしれない。
