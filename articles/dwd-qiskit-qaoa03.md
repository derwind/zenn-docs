---
title: "QAOA を眺めてみる (3) ― HOBO と QAOA とグラフカラーリング問題"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "QUBO", "HOBO"]
published: false
---

# 目的

blueqat さんのブログ記事 [HOBOソルバーでグラフカラーリング問題を効率化](https://blueqat.com/yuichiro_minato2/ae758ca8-27fe-43e8-8bdc-2171dfc3c01e) を HOBO ソルバと QAOA で解いてみようというもの。

欲張って、理論とソルバの話も盛り込んだので、前置きがとても長い・・・。

# HOBO ソルバ

**理論**

- [arXiv:2407.16106 Tensor Network Based HOBO Solver](https://arxiv.org/abs/2407.16106)

**ソルバ**

- [arXiv:2407.19987 HOBOTAN: Efficient Higher Order Binary Optimization Solver with Tensor Networks and PyTorch](https://arxiv.org/abs/2407.19987)

# そもそも QUBO と HOBO とは？

QUBO (Quadratic Unconstrained Binary Optimization) という組み合わせ最適化の定式化に用いられる概念があって、以前に [cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) でも扱った。

QUBO というのは、$n$ 個のバイナリ変数 $\bm{x}^T = (x_1, x_2, \ldots, x_n)^T \in \{0, 1\}^n$ があるとして、適当な $n$ 次実対称行列 $Q \in \operatorname{Mat}(n; \R)$ を用いて **定数項を除いて**

$$
\begin{align*}
H (\bm{x}) = \bm{x}^T Q \bm{x}
\tag{1}
\end{align*}
$$

という形で記述できる。各 $x_i$ について適当に 0 か 1 を代入して $H (\bm{x})$ を最小化したいのだが、組み合わせ数が $2^n$ 通りになるので一般的には解くのが難しい。

QUBO は 2 次式であるが、これに対して、HOBO (Higher Order Binary Optimization) というものがあって、3 次以上の項が出て来るものを指す。HOBO は HUBO と書いてある文献もあるので略語には注意されたい。

従来、HOBO を解く場合には [HOBOからQUBOへの変換](https://qiita.com/nori_autumn/items/2713bb3dc48663cb680b) のようなバイナリ変数の $x^2 = x$ という性質を活用する形で、「補助量子ビット」というものを導入して 2 次式に落とすというテクニックが使われていた。

ところで、式 (1) は以下のようにも書ける。

$$
\begin{align*}
H (\bm{x}) = \sum_{i,j=1}^2 Q_{ij} x_i x_j
\tag{1'}
\end{align*}
$$

この考えを拡張すると、では 3 次式だったら適当な $T_{ijk} \in \R$ を用いて

$$
\begin{align*}
H (\bm{x}) = \sum_{i,j,k=1}^3 T_{ijk} x_i x_j x_k
\tag{2}
\end{align*}
$$

と書けるのでは？という考えに至る。まさに [arXiv:2407.16106 Tensor Network Based HOBO Solver](https://arxiv.org/abs/2407.16106) はこのことを主張していて、上記は PyTorch では

```python
H = torch.einsum("ijk,i,j,k->", T, x, x, x)
```

で実装できることが p.6 で述べられている。このような 3 次或はより高次の $T_{ijk}$ は「テンソル」[^1]と呼ばれるもので、式 (2) はテンソル計算[^2]ということになる。

[^1]: テンソルはスカラー、ベクトル、行列を拡張する概念で、これらを自然に内包する。順に 0 階のテンソル、1 階のテンソル、2 階のテンソルと呼ぶこともできる。

[^2]: テンソル計算、特にここで使っている `torch.einsum` は、[A. Einstein の縮約記法](https://ja.wikipedia.org/wiki/%E3%82%A2%E3%82%A4%E3%83%B3%E3%82%B7%E3%83%A5%E3%82%BF%E3%82%A4%E3%83%B3%E3%81%AE%E7%B8%AE%E7%B4%84%E8%A8%98%E6%B3%95)に由来する。$\nu=0$ の時間、$\nu=1,2,3$ の空間という時空の 4 変数について「共変テンソル」と「反変テンソル」の要素をかけ合わせて和をとる操作が相対性理論ではしばしば登場するが、大変煩雑なのでこのような記法が導入された。計算機の理論ではこの制約が大幅に緩和されたようで、普通の多次元配列同士の要素のかけ合わせが許される。気になる場合は、間に計量テンソルが挟まっていて、添え字の上げ下げが行われていると思っても良いかもしれない。

HOBO → QUBO へのリダクションを行わずにテンソル計算で実際に HOBO を解く試みが [arXiv:2407.19987 HOBOTAN: Efficient Higher Order Binary Optimization Solver with Tensor Networks and PyTorch](https://arxiv.org/abs/2407.19987) であり、ソルバの試験実装は現在 https://github.com/ShoyaYasuda/hobotan にある。論文には難しそうなことも書いてあるが、本質的なのは「テンソル $T_{ijk}$ をどうやって構築するか？」という部分である。PyPI にパッケージは上がっていないので、以下のようにしてインストールする必要がある。

```sh
pip install -U git+https://github.com/ShoyaYasuda/hobotan
```

前置きはこれくらいにして「グラフカラーリング問題」を試しに解いてみたい。

# グラフカラーリング問題

グラフカラーリング問題というのはグラフ（頂点、辺）において、辺で繋がった両端の頂点同士が異なる色になるように、指定の範囲の色で塗り分けるという問題である。

まず [HOBOソルバーでグラフカラーリング問題を効率化](https://blueqat.com/yuichiro_minato2/ae758ca8-27fe-43e8-8bdc-2171dfc3c01e) の HOBO ソルバでの解法を眺めたい。ここでは 5 頂点を 4 色で塗り分ける問題を扱っている。頂点の辺での接続は以下の通りである:

![](/images/dwd-qiskit-qaoa03/001.png =500x)

## HOBO で解く

必要なモジュールを import する:

```python
from __future__ import annotations

from hobotan import (
    symbols, symbols_list, symbols_nbit, sampler, Auto_array, Compile
)
```

今回は整数エンコーディングという手法を用いてある。つまり、量子ビット $q_0$ と $q_1$ を用いて 4 色 (0, 1, 2, 3) を表現している。例えば、色 2 は 2 進数で表現すると `10` であるので $q_1 = 1$, $q_0 = 0$ と符号化できるいった形である。

以下ではオリジナルのコードに加え、色の出力と OK かどうかの判定を加えた。

```python
%%time

q = symbols_list(10, 'q{}')

# A(0, 1), B(2, 3), C(4, 5), D(6, 7), E(8, 9)
H =  ((q[0] - q[2])**2 -1)**2 * ((q[1] - q[3])**2 -1)**2 #AB
H +=  ((q[0] - q[6])**2 -1)**2 * ((q[1] - q[7])**2 -1)**2 #AD
H +=  ((q[2] - q[6])**2 -1)**2 * ((q[3] - q[7])**2 -1)**2 #BD
H +=  ((q[2] - q[4])**2 -1)**2 * ((q[3] - q[5])**2 -1)**2 #BC
H +=  ((q[2] - q[8])**2 -1)**2 * ((q[3] - q[9])**2 -1)**2 #BE
H +=  ((q[4] - q[8])**2 -1)**2 * ((q[5] - q[9])**2 -1)**2 #CE
H +=  ((q[6] - q[8])**2 -1)**2 * ((q[7] - q[9])**2 -1)**2 #DE

hobo, offset = Compile(H).get_hobo()
print(f'offset\n{offset}')

solver = sampler.SASampler(seed=0)
result = solver.run(hobo, shots=100)

for r in result[:5]:
    print(r)
    arr, subs = Auto_array(r[0]).get_ndarray('q{}')
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9 = arr
    A = 2 * q1 + q0
    B = 2 * q3 + q2
    C = 2 * q5 + q4
    D = 2 * q7 + q6
    E = 2 * q9 + q8
    ok = (A != B) and (A != D) and (B != D) and (B != C) and \
         (B != E) and (C != E) and (D != E)
    print(arr, f"{A=} {B=} {C=} {D=} {E=} {ok=}")
```

これは以下のような出力になる。

> offset
> 7.0
> [{'q0': 0, 'q1': 0, 'q2': 0, 'q3': 1, 'q4': 0, 'q5': 0, 'q6': 1, 'q7': 0, 'q8': 1, 'q9': 1}, -7.0, 3]
> [0 0 0 1 0 0 1 0 1 1] A=0 B=2 C=0 D=1 E=3 ok=True
> [{'q0': 1, 'q1': 0, 'q2': 0, 'q3': 0, 'q4': 1, 'q5': 0, 'q6': 0, 'q7': 1, 'q8': 1, 'q9': 1}, -7.0, 4]
> [1 0 0 0 1 0 0 1 1 1] A=1 B=0 C=1 D=2 E=3 ok=True
> [{'q0': 1, 'q1': 0, 'q2': 0, 'q3': 0, 'q4': 1, 'q5': 0, 'q6': 1, 'q7': 1, 'q8': 0, 'q9': 1}, -7.0, 1]
> [1 0 0 0 1 0 1 1 0 1] A=1 B=0 C=1 D=3 E=2 ok=True
> [{'q0': 1, 'q1': 0, 'q2': 0, 'q3': 0, 'q4': 1, 'q5': 1, 'q6': 1, 'q7': 1, 'q8': 0, 'q9': 1}, -7.0, 1]
> [1 0 0 0 1 1 1 1 0 1] A=1 B=0 C=3 D=3 E=2 ok=True
> [{'q0': 1, 'q1': 0, 'q2': 0, 'q3': 1, 'q4': 1, 'q5': 0, 'q6': 0, 'q7': 0, 'q8': 1, 'q9': 1}, -7.0, 1]
> [1 0 0 1 1 0 0 0 1 1] A=1 B=2 C=1 D=0 E=3 ok=True
> CPU times: user 19.5 s, sys: 6.64 ms, total: 19.5 s
> Wall time: 19.5 s

変数 `H` を丹念に展開すると分かるが、量子ビットについて 4 次の項が現れており、HOBO 式になっている。ソルバに投入する前に QUBO へのリダクションをせずとも解けているのである。

## QAOA で解く

QAOA でも `Rzzz` ゲートなどを作ることで HOBO を直接扱えるが、今回は敢えて QUBO を扱ってみたい。この場合、扱いが少しややこしくなるため、まずは小さな問題を解きたい。**頂点を 4 つに減らし**、4 色での塗分けを考える。頂点の接続は以下とする。

![](/images/dwd-qiskit-qaoa03/002.png =500x)

今回、Colab 上で T4 を用いて計算を行いたい。やや計算が大きいものがあるので GPU シミュレーションを使いたいのだ。

必要なモジュールのインストール:

```sh
pip install -qU qiskit qiskit[visualization] qiskit-aer-gpu
```

念のため重要なパッケージのバージョンを表示しておく。

```sh
%%bash

pip list | egrep -e "(qiskit|hobotan)"
```

> hobotan                          0.0.8
> qiskit                           1.1.1
> qiskit-aer-gpu                   0.14.2

追加でモジュールを import する:

```python
import pprint
import re
import sys
import time

import numpy as np
import scipy as sp
import numpy.random as nr
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.quantum_info import AerStatevector
```

### QUBO での定式化

量子ビットとして 2 次元の $q_{v,i}$ を考える。$v$ は頂点の番号で、今回 0, 1, 2, 3 である。$i$ は色の番号で、こちらも今回は 0, 1, 2, 3 である。頂点 $v$ が色 $i$ で塗られる時に $q_{v,i} = 1$ となり、それ以外では $q_{v,i} = 0$ とする。

同じ頂点 $v$ においては色はただ 1 つ決まる必要があるのでワンホット制約

$$
\begin{align*}
\left(\sum_{i=0}^3 q_{v,i} - 1\right)^2
\end{align*}
$$

を設定する。コードでは `HA` が対応する。
辺 $(u, v)$ で接続された頂点同士が同じ色 $i$ で塗られることを禁止したいので、$q_{u,i} q_{v,i}$ にペナルティを設定したい。頂点の接続の集合 $E = {(u, v)}$ を考え以下のようなコスト

$$
\begin{align*}
\sum_{u, v \in E} \sum_{i=0}^3 q_{u,i} q_{v,i}
\end{align*}
$$

を設定する。コードでは `HB` が対応する。

```python
%%time

n_vertices = 4
n_colors = 4

# vertex (v=A, B, C, D), color (m=0, 1, 2, 3)
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

### QUBO からイジングハミルトニアンへの変換

[cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) でも触れたが、この QUBO 式を $z_i = 1 - 2q_i$ という変数変換により、$z_i$ を用いたイジングハミルトニアンに置き換える必要がある。

実装の詳細は割愛するが、hobotan の関数を流用しつつ以下のような変換関数を作成した:

```python
import symengine
from sympy import Rational
from hobotan.compile import replace_function


def get_hobo(H):

    #式を展開して同類項をまとめる
    expr = symengine.expand(H)

    #二乗項を一乗項に変換
    expr = replace_function(expr, lambda e: isinstance(e, symengine.Pow) and e.exp == 2, lambda e, *args: e)

    #最高字数を調べながらオフセットを記録
    #項に分解
    members = str(expr).split(' ')

    #各項をチェック
    offset = 0
    ho = 0
    for member in members:
        #数字単体ならオフセット
        try:
            offset += float(member) #エラーなければ数字
        except:
            pass
        #'*'で分解
        texts = member.split('*')
        #係数を取り除く
        try:
            texts[0] = re.sub(r'[()]', '', texts[0]) #'(5/2)'みたいなのも来る
            float(Rational(texts[0])) #分数も対応 #エラーなければ係数あり
            texts = texts[1:]
        except:
            pass

        if len(texts) > ho:
            ho = len(texts)
    # print(ho)

    #もう一度同類項をまとめる
    expr = symengine.expand(expr)

    coeff_dict = expr.as_coefficients_dict()

    hobo = {}
    for key, value in coeff_dict.items():
        if key.is_Number:
            continue
        tmp = str(key).split('*')
        hobo[tuple(sorted(tmp))] = float(value)

    return hobo


def hobo2ising(hobo: dict[tuple[str, ...], float]) -> dict[tuple[int, ...], float]:
    expr = 0
    for key, value in hobo.items():
        term = value
        for k in key:
            # s = 1 - 2x <==> x = (1 - s) / 2
            vertex, color = [int(v) for v in k[1:].split("_")]
            term = term * (1 - symengine.symbols(str(n_colors * vertex + color))) / 2
        expr += term
    expr = symengine.expand(expr)

    coeff_dict = expr.as_coefficients_dict()

    ising = {}
    for key, value in coeff_dict.items():
        if key.is_Number:
            continue
        tmp = str(key).split('*')
        new_key = tuple(sorted(int(k) for k in sorted(tmp)))
        ising[new_key] = float(value)

    return ising
```

これを用いて QUBO をイジングハミルトニアンに変換する。直後に書く理由で、今回は `HA` は定義したもののこれは用いずに `HB` だけ変換する。4 頂点 4 色なので、$4 \times 4 = 16$ 個の量子ビットへの対応となる。`(0, 4): 0.25` などは、$0.25 z_0 z_4$ に対応する。

```python
hobo = get_hobo(HB)
ising = hobo2ising(hobo)
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


ところで、通常は `H = HA + HB` が最小となるように最適化をするのだが、今回は、`HA` のワンホット制約が自動的に満たされるようにして `HB` だけの最適化に持ち込みたい。このためには XY-mixer というものが使える。c.f [arXiv:1904.09314 $XY$-mixers: analytical and numerical results for QAOA](https://arxiv.org/abs/1904.09314)

XY-mixer は大雑把には、2 量子ビットの場合だと $\frac{1}{\sqrt{2}} (\ket{01} + \ket{10})$ の状態を維持させるために用いられる。詳しい説明は [blueqatでXYミキサーを用いた制約付きQAOA](https://qiita.com/ryuNagai/items/1836601f4d3c5ec9e336) などにある。

3 量子ビットの場合には $\frac{1}{\sqrt{3}} (\ket{001} + \ket{010} + \ket{100})$、4 量子ビットの場合には $\frac{1}{2} (\ket{0001} + \ket{0010} + \ket{0100} + \ket{1000})$ を維持できるのであればワンホット制約は自動的に満たされるのである。このような状態を Dicke 状態と呼ぶが、これらの状態を準備する方法が [arXiv:1904.07358 Deterministic Preparation of Dicke States](https://arxiv.org/abs/1904.07358) で知られている。

### Dicke 状態の作成

詳細は論文に委ねるとして実装は以下のようになる:

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
