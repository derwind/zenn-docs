---
title: "行列積状態について考える (9) — 100 量子ビットのもつれ状態"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "Qiskit", "TensorNetwork"]
published: false
---

# 目的

[行列積状態について考える (6) — Vidal の標準形](/derwind/articles/dwd-matrix-product06) と [行列積状態について考える (8) — Vidal の標準形 その 2](/derwind/articles/dwd-matrix-product08) で行列積状態 (MPS) の Vidal の標準形について書いた。今回はこれを用いて、100 量子ビットのもつれ状態を作成して、ちゃんとできたことを確認してみたい。

# 実装

必要なモジュールを import する。

```python
from __future__ import annotations
import numpy as np
from opt_einsum import contract

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.random import random_statevector
```

「Vidal の標準形」の実装は既に記載したが、細部に色々ミスがあったのでそれを手直ししつつ、2 量子ビットゲートの実装も行う。よく理解して実装したというよりは “動いた” という程度のものなのでまだミスがあるはずである。今回の内容的には少々ミスがあっても主張的がひっくり返るものではないと思うので気にしないことにする。

基本的には [The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477) と [Efficient classical simulation of slightly entangled quantum computations](https://arxiv.org/abs/quant-ph/0301063) に従った実装をしているつもりだが、嚙み合わせがよく分からなかった部分は、色々値を突っ込んで出力を見ながら、思った感じになるまで try & error して決めた。

## Vidal の標準形

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

上記が [行列積状態について考える (8) — Vidal の標準形 その 2](/derwind/articles/dwd-matrix-product08) の手直し・・・と言うか、こうしたら色々と動くという感じの実装である。

続いて 1 量子ビットゲートの実装は以下となる。

## 1 量子ビットゲート

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
PauliY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PauliZ = np.array([[1, 0], [0, -1]], dtype=float)
Hadamard = np.array([[1, 1], [1, -1]], dtype=float) / np.sqrt(2)


def apply_X(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliX, qubit)


def apply_Y(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliY, qubit)


def apply_Z(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, PauliZ, qubit)


def apply_H(gammas: list[np.ndarray], qubit: int) -> None:
    apply_one_qubit_gate(gammas, Hadamard, qubit)
```

更に 2 量子ビットゲートの実装は以下となる。

それっぽく `reshape` しているふりをしているが、実はよく分からなくて「こんな感じだと動く」内容を放り込んでいる状態である。

## 2 量子ビットゲート

```python
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

以上が実装で、これで今回やりたい程度の内容は動かせる。

# テスト

実装の妥当性を証明する方法が思いつかなかったので、ランダムな状態ベクトルを作って、Qiskit で MPS シミュレーションした場合の結果と一致すれば OK という評価をした。

```python
def reverse_sv(sv: Statevector) -> dict[str, complex]:
    return {k: v for k, v in sorted([(bin(i)[2:].zfill(sv.num_qubits)[::-1], coeff)
        for i, coeff in enumerate(sv.data)], key=lambda k_v: k_v[0])}


num_qubits = 5


def experiment(gate, indices):
    sv = random_statevector(2**num_qubits, seed=1234)
    qc = QuantumCircuit(num_qubits)
    i, j = indices
    if gate == "cx":
        qc.cx(i, j)
    elif gate == "cy":
        qc.cy(i, j)
    elif gate == "cz":
        qc.cz(i, j)

    evolved_sv = sv.evolve(qc)
    answer = np.array(list(reverse_sv(evolved_sv).values()))
    
    data = np.array(list(reverse_sv(sv).values()))
    gammas, lambdas = TT_SVD_Vidal(data)
    if gate == "cx":
        apply_CX(gammas, lambdas, i, j)
    elif gate == "cy":
        apply_CY(gammas, lambdas, i, j)
    elif gate == "cz":
        apply_CZ(gammas, lambdas, i, j)
    
    lambdas_ = lambdas[:]
    lambdas_.append(None)
    operands = []
    for i, (gamma, lam) in enumerate(zip(gammas, lambdas_)):
        operands.append(gamma)
        if lam is not None:
            operands.append(lam)
    
    expr = make_expr(sv.num_qubits)
    tensor = np.einsum(expr, *operands).flatten()
    global_phase_adjust = answer[0] / tensor[0]
    result = np.allclose(tensor * global_phase_adjust, answer)
    if not result:
        print(gate, indices)
    return result


indices_pairs = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]

total = 0
ok = 0
for gate in ["cx", "cy", "cz"]:
    for indices in indices_pairs:
        if experiment(gate, indices):
            ok += 1
        total += 1

print(f"OK: {ok}/{total}")
```

> OK: 24/24

ランダムな状態ベクトルのどこに `CX`, `CY`, `CZ` を適用しても Qiskit と一致するので、まぁ OK でしょうといったところである。

他にも色々テストは実装したが省略する。

# 100 量子ビットのもつれ状態

今回のメインである。`np.einsum` だと `operands` が多すぎるのか怒られたので、`opt_einsum` の `contract` を用いた。

$\ket{\Psi} = \ket{0}^{\otimes 100}$ に対応する MPS を作りたいのだが、愚直に作ると SVD を適用する以前にこの状態ベクトルがメモリに格納できないので、直接 MPS を作る関数 `zeros_mps` を作る。20 量子ビット分までのケースで、状態ベクトル + SVD で作った結果と比較してみた。

```python
def zeros_state(n: int):
    state = ket_ZERO
    for _ in range(n - 1):
        state = np.kron(state, ket_ZERO)
    return state

def zeros_mps(n: int):
    gammas = [np.array([[1.], [0.]])]
    lambdas = [np.array([1.])]
    for _ in range(1, n - 1):
        gammas.append(np.array([[[1.], [0.]]]))
        lambdas.append(np.array([1.]))
    gammas.append(np.array([[1., 0.]]))
    return gammas, lambdas

for n in range(2, 20+1):
    gammas, lambdas = TT_SVD_Vidal(zeros_state(n))
    gammas2, lambdas2 = zeros_mps(n)
    for i, (gamma1, gamma2) in enumerate(zip(gammas, gammas2)):
        assert gamma1.shape == gamma2.shape, (i, gamma1.shape, gamma2.shape)
        assert np.allclose(gamma1, gamma2)
    for i, (lam1, lam2) in enumerate(zip(lambdas, lambdas2)):
        assert lam1.shape == lam2.shape, (i, lam1.shape, lam2.shape)
        assert np.allclose(lam1, lam2)
```

特に問題が出なかったので正しく実装できたと信じる。

## 実験

かなりのインデックスを消費するので、インデックスの集合を用意しておく。

今回の実験では $\ket{\Psi} = \frac{1}{\sqrt{2}}(\ket{0}^{\otimes 100} - \ket{1}^{\otimes 100})$ を作って、波動関数 $\Psi_{00\cdots 0} = \frac{1}{\sqrt{2}}$ と $\Psi_{11\cdots 1} = -\frac{1}{\sqrt{2}}$ を求めてみることにした。

```python
%%time

indices = [chr(i) for i in range(ord("a"), ord("z")+1)] 
indices += [chr(i) for i in range(ord("A"), ord("Z")+1)]
indices += [chr(i) for i in range(ord("α"), ord("ω")+1)]
indices += [chr(i) for i in range(ord("Α"), ord("Ρ")+1)]
indices += [chr(i) for i in range(ord("Σ"), ord("Ω")+1)]

num_qubits = 100
gammas, lambdas = zeros_mps(num_qubits)

apply_H(gammas, 0)
for i in range(num_qubits - 1):
    apply_CX(gammas, lambdas, i, i + 1)
apply_Z(gammas, num_qubits - 1)
```

求めたい波動関数は

$$
\begin{align*}
\Psi_{00 \cdots 0} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} \Gamma^{[1] 0}_{\alpha_1} \lambda^{[1]}_{\alpha_1} \Gamma^{[2] 0}_{\alpha_1 \alpha_2} \lambda^{[2]}_{\alpha_2} \cdots \Gamma^{[d] 0}_{\alpha_{d-1}} 
\end{align*}
$$

と

$$
\begin{align*}
\Psi_{11 \cdots 1} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} \Gamma^{[1] 1}_{\alpha_1} \lambda^{[1]}_{\alpha_1} \Gamma^{[2] 1}_{\alpha_1 \alpha_2} \lambda^{[2]}_{\alpha_2} \cdots \Gamma^{[d] 1}_{\alpha_{d-1}} 
\end{align*}
$$

で求まるので、これを計算するための `operands` と `expr` を作って `opt_einsum` の `contract` に渡すことになる。

まずは縮約計算に用いる `operands` をそれぞれ用意する。

```python
idx = 0
ket_ZEROs_operands = [gammas[0][idx, :]]
for lam, gamma in zip(lambdas[:-1], gammas[1:-1]):
    ket_ZEROs_operands.append(lam)
    ket_ZEROs_operands.append(gamma[:, idx, :])
ket_ZEROs_operands.append(lambdas[-1])
ket_ZEROs_operands.append(gammas[-1][:, idx])

idx = 1
ket_ONEs_operands = [gammas[0][idx, :]]
for lam, gamma in zip(lambdas[:-1], gammas[1:-1]):
    ket_ONEs_operands.append(lam)
    ket_ONEs_operands.append(gamma[:, idx, :])
ket_ONEs_operands.append(lambdas[-1])
ket_ONEs_operands.append(gammas[-1][:, idx])
```

次に縮約計算用の `expr` を作る。

```python
expr = [indices[0]]  # for first Gamma
for i, (a, b) in enumerate(zip(indices[:-1], indices[1:])):
    if i + 2 >= num_qubits:
        break
    expr.append(a)  # for Lambda
    expr.append(f"{a}{b}")  # for Gamma
expr.append(indices[num_qubits - 2])  # for last Lambda
expr.append(indices[num_qubits - 2])  # for last Gamma
expr = ",".join(expr) + "->"
```

最後に実際に縮約計算を実行する。併せて各種情報も確認してしまう。

```python
print(f"The amount of entanglement: {max([len(lam) for lam in lambdas])}")
print(expr)
wave_func_for_zeros = contract(expr, *ket_ZEROs_operands)
wave_func_for_ones = contract(expr, *ket_ONEs_operands)
print("The wave func of |00..00>:", wave_func_for_zeros)
print("The wave func of |11..11>:", wave_func_for_ones)
print(np.isclose(abs(wave_func_for_zeros)**2 + abs(wave_func_for_ones)**2, 1))
```

> The amount of entanglement: 2
> a,a,ab,b,bc,c,cd,d,de,e,ef,f,fg,g,gh,h,hi,i,ij,j,jk,k,kl,l,lm,m,mn,n,no,o,op,p,pq,q,qr,r,rs,s,st,t,tu,u,uv,v,vw,w,wx,x,xy,y,yz,z,zA,A,AB,B,BC,C,CD,D,DE,E,EF,F,FG,G,GH,H,HI,I,IJ,J,JK,K,KL,L,LM,M,MN,N,NO,O,OP,P,PQ,Q,QR,R,RS,S,ST,T,TU,U,UV,V,VW,W,WX,X,XY,Y,YZ,Z,Zα,α,αβ,β,βγ,γ,γδ,δ,δε,ε,εζ,ζ,ζη,η,ηθ,θ,θι,ι,ικ,κ,κλ,λ,λμ,μ,μν,ν,νξ,ξ,ξο,ο,οπ,π,πρ,ρ,ρς,ς,ςσ,σ,στ,τ,τυ,υ,υφ,φ,φχ,χ,χψ,ψ,ψω,ω,ωΑ,Α,ΑΒ,Β,ΒΓ,Γ,ΓΔ,Δ,ΔΕ,Ε,ΕΖ,Ζ,ΖΗ,Η,ΗΘ,Θ,ΘΙ,Ι,ΙΚ,Κ,ΚΛ,Λ,ΛΜ,Μ,ΜΝ,Ν,ΝΞ,Ξ,ΞΟ,Ο,ΟΠ,Π,ΠΡ,Ρ,ΡΣ,Σ,ΣΤ,Τ,ΤΥ,Υ,ΥΦ,Φ,ΦΧ,Χ,Χ->
> The wave func of |00..00>: 0.7071067811865475
> The wave func of |11..11>: -0.7071067811865475
> True
> CPU times: user 421 ms, sys: 844 ms, total: 1.27 s
> Wall time: 67.9 ms

まず、もつれ量は GHZ 状態の場合には量子ビット数によらず 2 であり、これは Vidal の意味での _slightly entangled_ な状態である。よってテンソルネットワーク計算で効率的に計算ができることになる。

`expr` の中身も念のため表示してみたが、実に酷いものである。用意した文字をほとんど使い切っている。

波動関数は想定通りに $\frac{1}{\sqrt{2}}$ と $-\frac{1}{\sqrt{2}}$ が求まっており、2 乗和が 1 なので、他の波動関数は 0 であることが分かる。よって、$\ket{\Psi} = \frac{1}{\sqrt{2}}(\ket{0}^{\otimes 100} - \ket{1}^{\otimes 100})$ が正しく作られたことを確認できた。

そして、効率的に計算できたことの証拠として、CPU 上で実行したのに、量子ビットの準備からゲートの適用まですべてを含めて、僅かに 70ms 未満で計算できたことが表示から分かった。

# まとめ

「テンソルネットワーク計算なら 100 量子ビットのシミュレーションでも実行可能である」という類のことを NumPy の実装を通じて確認した。勿論これは「可能な時には可能」というだけであり、例えば全ての計算基底に対する波動関数を確認しようとすると、それは $2^n$ 回の縮約計算を伴うので (時間について) 指数関数的なリソースを要求する。よって事実上あまり可能ではなくなってくる。

また、[行列積状態について考える (7) — QAOA ともつれ量](/derwind/articles/dwd-matrix-product07) で見たように、QAOA の ansatz のような量子ビット数 $n$ に対して指数関数的に増大するもつれ量の回路の場合、テンソルネットワーク内の和をとる計算の回数が指数関数のオーダーで増えるので、これも不適切である (Vidal の意味での _slightly entangled_ な状態ではない)。

その他色々あるとは思うが、$n$ に対して各種の量が $\operatorname{poly}(n)$ で増大する程度の場合にはこのような MPS でのシミュレーションで効率的な計算ができるであろうという感じであろう。

# 参考文献

[S] [Ulrich Schollwoeck, The density-matrix renormalization group in the age of matrix product states, arXiv:1008.3477, 2010.](https://arxiv.org/abs/1008.3477)
[V] [Guifre Vidal, Efficient classical simulation of slightly entangled quantum computations, arXiv:quant-ph/0301063, 2003.](https://arxiv.org/abs/quant-ph/0301063)
