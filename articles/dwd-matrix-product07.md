---
title: "行列積状態について考える (7) — QAOA ともつれ量"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "Qiskit", "量子機械学習"]
published: false
---

# 目的

QAOA の実験を [Qiskit で遊んでみる (21) — QAOA でお絵描き](/derwind/articles/dwd-qiskit21) で行った。この時、大規模な QAOA のシミュレーションの可能性の模索のため 25 量子ビットの QAOA を試したのだが、GPU 上のテンソルネットワーク計算で最適化に T4 で 1 時間 35 分かかった。一般的にはテンソルネットワーク計算が向いているのは

- 浅い回路
- もつれの少ない回路

である。浅さはさて置きにしても、これほど時間がかかった理由として、**もつれも大きそうな気がする**のでこれについて確認したい。

この結果が真の時、QAOA の計算については、cuTensorNet を含めたテンソルネットワーク計算では (計算パスが多すぎて) 効率的なシミュレーションは望めないであろうということになる。

# もつれを定量的に調べるには？

ここは定番の [量子コンピュータと量子通信I](https://www.ohmsha.co.jp/book/9784274200076/) を紐解きたい。これは、p.151- の「2.5 Schmidt 分解と純粋化」が関係してくる。

$\ket{\psi}$ が複合システム AB の純粋状態であるとすると、A の正規直交基底 $\ket{i_A}$ と B の正規直交基底 $\ket{i_B}$ が存在して

$$
\begin{align*}
\ket{\psi} = \sum_i \lambda_i \ket{i_A} \ket{i_B}
\end{align*}
$$

と書けるという主張が「Schmidt 分解」である。ここで、$\lambda_i$ は正の数で $\sum_i \lambda_i^2 = 1$ を満たす[^1]。

[^1]: この証明はそんなに難しいものではなく、量子状態ベクトルの係数である「波動関数」を集めて作った行列を SVD するというものであり、[特異値分解のちょっと格好いい姿を眺めてみる](https://zenn.dev/derwind/articles/dwd-singular-value-decomposition) に書いたような内容が相当する。

特異値 $\lambda_i$ の部分について、これが $N$ 個あるとした場合、「Schmidt 数 (或は Schmidt ランク) が $N$ である」と呼び、これが**もつれの量**に相当するということらしい。つまり、$N > 1$ では何かしらもつれていることになる。

例として、2 量子ビットの複合システムを考える場合、$\ket{\psi} = \frac{1}{\sqrt{2}} \ket{0}(\ket{0} + \ket{1})$ の Schmidt 数は 1 であるが、$\ket{\psi} = \frac{1}{\sqrt{2}}(\ket{00} + \ket{11})$ の Schmidt 数は 2 である。

# Vidal 標準形

[行列積状態について考える (6) — Vidal の標準形](https://zenn.dev/derwind/articles/dwd-matrix-product06) では、MPS の一形態である Vidal 標準形について扱った。G. Vidal の論文 [Efficient classical simulation of slightly entangled quantum computations](https://arxiv.org/abs/quant-ph/0301063) を読むと、(2) 式で上記と同等の式

$$
\begin{align*}
\ket{\Psi} = \sum_{\alpha=1}^{\chi_A} \lambda_{\alpha} \ket{\Phi^{[A]}_{\alpha}} \ket{\Phi^{[B]}_{\alpha}}
\end{align*}
$$

が出て来る。文章を抜粋すると、

> The Schmidt rank $\chi_A$ is a natural measure of the entanglement between the qubits in $A$ and those in $B$. Accordingly, we quantify the entanglement of state $\ket{\Psi}$ by $\chi$,

$$
\begin{align*}
\chi \equiv \max_{A} \chi_A
\end{align*}
$$

> that is, by the maximal Schmidt rank over all possible bipartite splittings $A:B$ of the $n$ qubits.

とのことである。色々な箇所でシステムを分断してみて、一番 Schmidt 数が大きかったところをもって、定量化した「状態のもつれ量」と考えるということである。

さて、Vidal の標準形とは [行列積状態について考える (6) — Vidal の標準形](https://zenn.dev/derwind/articles/dwd-matrix-product06) で触れたように、

$$
\begin{align*}
\Psi_{i_1 i_2 \cdots i_d} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} \Gamma^{[1] i_1}_{\alpha_1} \lambda^{[1]}_{\alpha_1} \Gamma^{[2] i_2}_{\alpha_1 \alpha_2} \lambda^{[2]}_{\alpha_2} \cdots \Gamma^{[d] i_d}_{\alpha_{d-1}} 
\end{align*}
$$

のようなものであったが、これの言葉で書くと、状態 $\ket{\Psi}$ のもつれ量とは、**特異値行列 $\lambda^{[1]}, \cdots, \lambda^{[d-1]}$ のうち最大の行列のサイズ**のことである。

ここまで来たら細かいことは忘れてしまって、Python で実験をしてみよう。

# 準備実験

今回の実験に使うモジュールを import する。

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from tytan import symbols_list, Compile, sampler, Auto_array

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.quantum_info import AerStatevector as Statevector
```

Vidal の標準形を求める関数を定義する。[行列積状態について考える (6) — Vidal の標準形](https://zenn.dev/derwind/articles/dwd-matrix-product06) の流用であるが、ついでにテンソル計算用の添え字表現を計算する関数も追加しよう。

```python
def TT_SVD_Vidal(
    C: np.ndarray, num_qubits: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gammas = []
    lambdas = []

    if len(C.shape) == 1:
        C = C.reshape(1, -1)
    if num_qubits is None:
        num_qubits = int(np.log2(np.prod(C.shape)))

    for i in range(num_qubits - 1):
        # step 1
        if i == 0:
            remaining_matrix = C
        else:
            remaining_matrix = S.reshape(-1, 1) * Vh  # np.diag(S) @ Vh
        _, cols = remaining_matrix.shape
        reshaped_matrix = np.concatenate([
            remaining_matrix[:, :cols // 2], remaining_matrix[:, cols // 2:]
        ], axis=0)
        # step 2
        U, S, Vh = np.linalg.svd(reshaped_matrix, full_matrices=False)
        r = len(S) - len(S[S == 0])
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        # step 3
        rows, _ = U.shape
        left_gamma = np.array([U[:rows // 2, :].tolist(), U[rows // 2:, :].tolist()])
        if i != 0:
            left_gamma /= lambdas[-1].reshape(1, -1, 1)
        if i == 0:
            left_gamma = left_gamma.squeeze(axis=1)  # drop a1
        gammas.append(left_gamma)
        lambdas.append(S)
    # step 4
    _, cols = Vh.shape
    right_gamma = np.array([Vh[:, :cols // 2], Vh[:, cols // 2:]])
    right_gamma = right_gamma.squeeze(axis=2)  # drop a2
    gammas.append(right_gamma)

    return gammas, lambdas


def make_expr(n_qubits: int) -> str:
    outer_indices = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    inner_indices = [chr(i) for i in range(ord("a"), ord("z") + 1)]

    expr = []
    prev_inner = ""
    for i, (outer_i, inner_i) in enumerate(zip(outer_indices, inner_indices)):
        if i + 1 < n_qubits:
            expr.extend([f"{outer_i}{prev_inner}{inner_i}", inner_i])
            prev_inner = inner_i
        else:
            expr.extend([f"{outer_i}{prev_inner}"])
            break
    return ",".join(expr) + "->" + "".join(outer_indices[:n_qubits])
```

もつれを計算する関数は以下のようになる。`TT_SVD_Vidal` の戻り値の特異値行列のリストを入力にして求める。

```python
def calc_entanglement(lambdas: list[np.array]) -> int:
    return max([l.shape[0] for l in lambdas])
```

## もつれがない場合と Bell 状態

以下のようにそれぞれ、最大 Schmidt 数は 1 と 2 であり、想定通りである。

```python
qc = QuantumCircuit(2)
qc.h(1)
display(qc.draw("mpl", style="clifford"))

sv = Statevector(qc)

gammas, lambdas = TT_SVD_Vidal(sv.data, 2)
print("entanglement of the state:", calc_entanglement(lambdas))
```

![](/images/dwd-matrix-product07/001.png)

> entanglement of the state: 1

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
display(qc.draw("mpl", style="clifford"))

sv = Statevector(qc)

gammas, lambdas = TT_SVD_Vidal(sv.data, 2)
print("entanglement of the state:", calc_entanglement(lambdas))
```

![](/images/dwd-matrix-product07/002.png)

> entanglement of the state: 2

これを含め、一般に状態の最大 Schmidt 数が 1 であればもつれがなく、各量子ビットごとの状態のテンソル積で全体の状態が記述できることになる。

## GHZ 状態とかなりもつれていそうな場合の比較

もう少し複雑なケースを見てみたい。

```python
def make_ghz_circuit(n_qubits: int):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc

def make_very_entangled_circuit(n_qubits: int):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                qc.cx(i, j)
    return qc
```

```python
make_ghz_circuit(5).draw("mpl", style="clifford")
```

![](/images/dwd-matrix-product07/003.png)

```python
make_very_entangled_circuit(5).draw("mpl", style="clifford")
```

![](/images/dwd-matrix-product07/004.png)

これらの量子状態のもつれ量はどれくらいなのであろうか？

```python
for n in range(2, 10 + 1):
    ghz_circ = make_ghz_circuit(n)
    sv = Statevector(ghz_circ)
    gammas, lambdas = TT_SVD_Vidal(sv.data, n)
    ghz_entanglement = calc_entanglement(lambdas)
    circ = make_very_entangled_circuit(n)
    sv = Statevector(circ)
    gammas, lambdas = TT_SVD_Vidal(sv.data, n)
    very_entangled_entanglement = calc_entanglement(lambdas)
    print(f"[{n}] {ghz_entanglement} {very_entangled_entanglement}")
```

> [2] 2 2
> [3] 2 2
> [4] 2 4
> [5] 2 4
> [6] 2 8
> [7] 2 8
> [8] 2 16
> [9] 2 16
> [10] 2 32

GHZ 状態の回路は量子ビット数に関わらず常に 2 であるが、かなりもつれさせた回路は量子ビット数 $n$ に対して、$2^{\lfloor n/2 \rfloor}$ くらいのもつれ量になっていそうである[^2]。

[^2]: 理論的なチェックはしていない。

# QAOA ansatz の実験

## 最大カット問題

今回、QAOA を適用する問題として最大カット問題を扱いたい。[tytan_tutorial](https://github.com/tytansdk/tytan_tutorial) のコンテンツを拝借する。

繋ぐエッジはかなり適当だが、10 量子ビットのケースまで適当に拡張した。

```python
n_qubits = 5

edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
if n_qubits == 6:
    edges += [(0, 5), (3, 5)]
elif n_qubits == 7:
    edges += [(0, 5), (1, 6), (2, 6), (3, 5)]
elif n_qubits == 8:
    edges += [(0, 5), (1, 6), (2, 6), (3, 5), (0, 7), (2, 7)]
elif n_qubits == 9:
    edges += [(0, 5), (1, 6), (2, 6), (3, 5), (0, 7), (1, 8), (2, 7), (2, 8), (4, 8), (5, 7)]
elif n_qubits == 10:
    edges += [(0, 5), (1, 6), (2, 6), (3, 5), (0, 7), (1, 8), (2, 7), (2, 8), (3, 9), (4, 5), (4, 8), (5, 7), (6, 9), (8, 9)]

G = nx.Graph()
G.add_edges_from(edges)

nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G), with_labels=True, node_color="#EE5396")
```

![](/images/dwd-matrix-product07/005.png)

## イジングハミルトニアン

QAOA 用のイジングハミルトニアンを作るコードを一気に用意する。昔のコードを流用しているので少し不自然な部分もあるが、もつれ量の計測には支障はない。

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


q = symbols_list(n_qubits, 'q{}')

H = 0
for i, j in edges:
    H += (q[i] + q[j] - 1)**2

qubo, offset = Compile(H).get_qubo()

ising_dict, additional_offset = get_ising(qubo, n_qubits)
hamiltonian, coefficients = get_hamiltonian(ising_dict, n_qubits)

coefficients /= np.max(abs(coefficients))

qubit_op = SparsePauliOp(
    [ham[::-1] for ham in hamiltonian],
    coefficients,
)
```

## QAOA ansatz の作成

Qiskit の API を用いて ansatz を作成する。

```python
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

## もつれ量の計測

`n_qubits = 5` の場合を見てみよう。ansatz にランダムな値を割り当てて Vidal 標準形を求めて、もつれ量を計測する。

```python
rng = np.random.default_rng(seed=42)

params = rng.random(len(ansatz.parameters))
params_binded_ansatz = ansatz.assign_parameters(params)

sv = Statevector(params_binded_ansatz)
gammas, lambdas = TT_SVD_Vidal(sv.data, n_qubits)

print(f"[{n_qubits}] {calc_entanglement(lambdas)}")
```

> [5] 4

他のケースも見てみよう。

> [6] 8
> [7] 8
> [8] 16
> [9] 16
> [10] 32

で、前述の「かなりもつれさせた回路」と同様の状況になっていることが分かる。

## 元の状態の復元

元の状態の復元は以下のようにしてできる。

```python
lambdas_ = lambdas[:]
lambdas_.append(None)

operands = []
for g, l in zip(gammas, lambdas_):
    if l is not None:
        operands.extend([g, l])
    else:
        operands.extend([g])

tensor = np.einsum(make_expr(n_qubits), *operands).flatten()
print(np.allclose(tensor, sv.data))
```

但しどんどん重くなっていき、CPU での計算では 9 量子ビットで 1 分、**10 量子ビットで 1 時間くらい**かかったので、あまり嬉しい状況ではなかった。

## Vidal 標準形のテンソルの形状

最後に、10 量子ビットのケースでの Vidal 標準形のテンソルの形状だけ眺めておこう。

```python
lambdas_ = lambdas[:]
lambdas_.append(None)
for i, (gamma, lam) in enumerate(zip(gammas, lambdas_)):
    print(f"gamma_{i}.shape={gamma.shape}")
    if lam is not None:
        print(f"lambda_{i}.shape={lam.shape}")
```

> gamma_0.shape=(2, 2)
> lambda_0.shape=(2,)
> gamma_1.shape=(2, 2, 4)
> lambda_1.shape=(4,)
> gamma_2.shape=(2, 4, 8)
> lambda_2.shape=(8,)
> gamma_3.shape=(2, 8, 16)
> lambda_3.shape=(16,)
> gamma_4.shape=(2, 16, 32)
> lambda_4.shape=(32,)
> gamma_5.shape=(2, 32, 16)
> lambda_5.shape=(16,)
> gamma_6.shape=(2, 16, 8)
> lambda_6.shape=(8,)
> gamma_7.shape=(2, 8, 4)
> lambda_7.shape=(4,)
> gamma_8.shape=(2, 4, 2)
> lambda_8.shape=(2,)
> gamma_9.shape=(2, 2)

全体を半分ずつの部分システムに分けた場合のもつれ量が大きいらしい。

# まとめ

- なかなか量子状態のもつれ具合といっても、それを回路を見ただけで判断するのは困難だと思うが、今回 Schmidt 数という道具を用いて、定量的な見方について確認した。
- これは数値計算上はダイレクトに計算効率に関わってくる重要な値であり、QAOA の典型的な ansatz でどのような状況なのかを確認した。
- 結果、もつれ具合はかなり酷いものになりやすいことが感じられた。
- 理論的には確認していないが、量子ビット数 $n$ に対して、$2^{\lfloor n/2 \rfloor}$ くらいのもつれ量になるとすれば、[Qiskit で遊んでみる (21) — QAOA でお絵描き](/derwind/articles/dwd-qiskit21) で行った 25 量子ビットでの計算は、もつれ量 4096 にまで至りそうである。
- G. Vidal の論文では、もつれ量が小さいことの 1 つの資料は、もつれ量が $\operatorname{poly}(n)$ 程度の時を指しているので、今回は該当してない可能性が大きい。
- 理論的な細部は見ていないが、挙動的に「QAOA の期待値計算をする上で、テンソルネットワーク計算は効率的ではないだろう」と言えそうに思う。ということで冒頭の疑問についてはネガティブな見解を得た。

# 参考文献

[NC] [Michael A. Nielsen, Isaac L. Chuang, 量子コンピュータと量子通信I －量子力学とコンピュータ科学－, オーム社, 2004.](https://www.ohmsha.co.jp/book/9784274200076/)
[V] [Guifre Vidal, Efficient classical simulation of slightly entangled quantum computations, arXiv:quant-ph/0301063, 2003.](https://arxiv.org/abs/quant-ph/0301063)
