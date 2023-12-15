---
title: "行列積状態について考える (5) — ニューラルネットワークのモデル圧縮"
emoji: "⛓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "機械学習"]
published: true
---

# 目的

[行列積状態について考える (3)](/derwind/articles/dwd-matrix-product03) の続きとして、ニューラルネットワークのモデル圧縮について考えたい。[arXiv:1509:06569 Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569) に沿って考えたい。

# 全結合層と TT-層

ニューラルネットワークの全結合層は $W \in \operatorname{Mat}(m,n; \R)$ および $b \in \R^m$ に対して、$x \in \R^n$ を入力として

$$
\begin{align*}
y = Wx + b
\end{align*}
$$

により出力 $y \in \R^m$ を与えるようなものである。

以下、[arXiv:1509:06569](https://arxiv.org/abs/1509.06569) 3.1 TT-representations for vectors and matrices について見ていく。

## 行列

行列 $W$ 部分について見る。

仮に $m = m_1 m_2 m_3$, $n = n_4 n_5$ と因数分解できるとすると、$W$ を $\operatorname{reshape}(m_1, m_2, m_3, n_4, n_5)$ で整形すると 5 階のテンソルと見做すことができる。$5 = d$ として、更にこのテンソルを TT-分解することで、$W$ は $d$ 個の TT-分解の要素に対応する。

$1 \leq t \leq m$, $1 \leq \ell \leq n$ として、$W$ の要素 $W(t, \ell)$ は $\sigma_j := \sigma_j(t, \ell) = (\nu_j(t, \ell), \mu_j(t, \ell))$ として、テンソル $\mathcal{W}$ の要素 $\mathcal{W}(\sigma_1, \sigma_2, \sigma_3, \sigma_4, \sigma_5)$ と対応付く, i.e.,

$$
\begin{align*}
W(t, \ell) = \mathcal{W}(\sigma_1, \sigma_2, \sigma_3, \sigma_4, \sigma_5)
\end{align*}
$$

$\mathcal{W}$ を TT-分解 $G^1 G^2 G^3 G^4 G^5$ すると、更に

$$
\begin{align*}
W(t, \ell) &= \mathcal{W}(\sigma_1, \sigma_2, \sigma_3, \sigma_4, \sigma_5) \\
&= G^1_{\sigma_1} G^2_{\sigma_2} G^3_{\sigma_3} G^4_{\sigma_4} G^5_{\sigma_5} \\
&= \sum_{a_1=1}^{r_1} \sum_{a_2=1}^{r_2} \sum_{a_3=1}^{r_3} \sum_{a_4=1}^{r_4} G^1_{\sigma_1,a_1} G^2_{a_1,\sigma_2,a_2} G^3_{a_2,\sigma_3,a_3} G^4_{a_3,\sigma_4,a_4} G^5_{a_4,\sigma_4}
\end{align*}
$$

のように書ける。この辺が [arXiv:1509:06569](https://arxiv.org/abs/1509.06569) の (3) 式になる。この表現をしたものを論文では「TT-行列」と呼んでいる。

## ベクトル

ベクトル $b \in \R^m$ についても $m = m_1 m_2 m_3$ であることから 3 階のテンソルと見做すことができる。上で見た TT-行列のように、$b$ に対応するテンソル $\mathcal{B}$ を TT-分解したものに対応付けたものを、論文では「TT-ベクトル」と呼んでいる。

## TT-層

$$
\begin{align*}
y = Wx + b
\end{align*}
$$

をテンソルに変換して、**行列部分を更に TT-分解して TT-行列にする**と、

$$
\begin{align*}
&\mathcal{Y}(i_1,i_2,i_3,i_4,i_5) \\
=& \ \sum_{a_1,a_2,a_3,a_4,a_5} G^1_{i_1,a_1} G^2_{a_1,i_2,a_2} G^3_{a_2,i_3,a_3} G^4_{a_3,i_4,a_4} G^5_{a_4,i_4} \mathcal{X}(a_1,a_2,a_3,a_4,a_5) + \mathcal{B}(i_1,i_2,i_3,i_4,i_5)
\end{align*}
$$

を得る。このように変形した全結合層を論文では「TT-層」と呼んでいる。

Tensor-Train 中の各結合の次元を $r_1$, $r_2$, $r_3$, $r_4$ より小さくするとパラメータ数を削減できる、つまりニューラルネットワークの「プルーニング」「モデル圧縮」の一種を実現できる。

# 実装

理屈だけを見ていても分かるような分からないような気持ちにしかならないので Python で実装してみる。

[行列積状態について考える (3)](/derwind/articles/dwd-matrix-product03) で試した `TT_SVD` 関数を使う。Tensor-Train の最後の要素で `.T` をとっていたが、これも少し気持ち悪いので、今回はとらないようにする。また、結合次元の自動計算やチェック機能も追加して以下のようにする。

## 必要なモジュールの import

```python
from __future__ import annotations
from typing import Sequence
import numpy as np
```

## 少し改良した `TT_SVD`

```python
def TT_SVD(
    C: np.ndarray, r: Sequence[int] | None = None, check_r: bool = False
) -> list[np.ndarray]:
    """TT_SVD algorithm

    Args:
        C (np.ndarray): n-dimensional input tensor
        r (Sequence[int]): a list of bond dimensions.
                           If `r` is None, `r` will be automatically calculated
        check_r (bool): check if `r` is valid

    Returns:
        list[np.ndarray]: a list of core tensors of TT-decomposition
    """

    dims = C.shape
    n = len(dims)  # n-dimensional tensor

    if r is None or check_r:
        # Theorem 2.1
        r_ = []
        for sep in range(1, n):
            row_dim = np.prod(dims[:sep])
            col_dim = np.prod(dims[sep:])
            rank = np.linalg.matrix_rank(C.reshape(row_dim, col_dim))
            r_.append(rank)
        if r is None:
            r = r_

    if len(r) != n - 1:
        raise ValueError(f"{len(r)=} must be {n - 1}.")
    if check_r:
        for i, (r1, r2) in enumerate(zip(r, r_)):
            if r1 > r2:
                raise ValueError(f"{i}th dim {r1} must not be larger than {r2}.")

    # Algorithm 1
    tt_cores = []
    for i in range(n - 1):
        if i == 0:
            ri_1 = 1
        else:
            ri_1 = r[i - 1]
        ri = r[i]
        C = C.reshape(ri_1 * dims[i], np.prod(dims[i + 1:]))
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        # approximation
        U = U[:, :ri]
        S = S[:ri]
        Vh = Vh[:ri, :]
        tt_cores.append(U.reshape(ri_1, dims[i], ri))
        C = np.diag(S) @ Vh
    tt_cores.append(C)
    tt_cores[0] = tt_cores[0].reshape(dims[0], r[0])
    return tt_cores
```

## 全結合層とミニバッチの定義

機械学習では入力 $x$ は幾つかのデータを束ねたミニバッチであることが基本なので、そのようにする。

```python
rng = np.random.default_rng(12345)

batch_dim = 16

# a fully connected layer
w = rng.standard_normal((3*4, 5*6))
b = rng.standard_normal(3*4)

# mini batch
x = rng.standard_normal((batch_dim, 5*6))
```

最終的には TT-層を用いて以下の結果を得たい。

```python
answer = x @ w.T + b
print(f"num of params={np.prod(w.shape) + np.prod(b.shape)}")
```

> num of params=372

## 肩慣らし

徐々にテンソル計算にしていこう。

### 普通の行列計算を縮約計算で書く

```python
val1 = np.einsum("Nj,ij->Ni", x, w) + b  # use N as the index symbol for batch dim
print(f"{np.allclose(answer, val1)=}")
```

> np.allclose(answer, val1)=True

これは最も素直な行列の計算なので当然一致する。

### テンソルに reshape して計算する

少し雰囲気を出そう。

```python
W = w.reshape(3, 4, 5, 6)
X = x.reshape(batch_dim, 5, 6)
B = b.reshape(3, 4)

print(f"num of params={np.prod(W.shape) + np.prod(B.shape)}")

val2 = np.einsum("Nkl,ijkl->Nij", X, W) + B
val2 = val2.reshape(batch_dim, -1)
print(f"{np.allclose(answer, val2)=}")
```

> num of params=372
> np.allclose(answer, val2)=True

この時点ではパラメータ数も変化していないし、計算結果も元のものと一致している。

## 重み行列 `W` を TT-分解する

次に `W` を TT-分解する。

```python
tt_W = TT_SVD(W, [3, 12, 6])
print("tt_W:", [v.shape for v in tt_W])

print(f"num of params={np.sum([np.prod(v.shape) for v in tt_W]) + np.prod(B.shape)}")

W1 = np.einsum("ic,cjd,dke,el->ijkl", *tt_W)

print(f"{np.allclose(W, W1)=}")
```

> tt_W: [(3, 3), (3, 4, 12), (12, 5, 6), (6, 6)]
> num of params=561
> np.allclose(W, W1)=True

テンソル `W` と Tensor-Train `tt_W` は本質的には同じものであるが、パラメータ数が増えてしまった。厳密な値を再現できる状態だと結合部の分だけパラメータが増えてしまう。

続けて、TT-層としての計算を実行してみる。

```python
val3 = np.einsum("Nkl,ic,cjd,dke,el->Nij", X, *tt_W) + B
val3 = val3.reshape(batch_dim, -1)
print(f"{np.allclose(answer, val3)=}")
print(f"max diff={np.round(np.max(np.abs(answer - val3)), 5)}")
```

> np.allclose(answer, val3)=True
> max diff=0.0

元の `x @ w.T + b` の計算と値が一致した。

# 低ランク近似

このままだとパラメータ数が増えただけで計算結果も変わらないという、何も旨味がない状態である。ところが、Tensor-Train の結合部の次元を $r_1$, $r_2$, ... より小さくすることで近似計算をすると共にパラメータ削減ができる。これを見てみよう。

今回、重み行列を 4 階のテンソルにしてから TT-分解しているので、結合部の個数は 3 つで、元のテンソルを復元できる TT-ランクは `(3, 12, 6)` である。真ん中の TT-ランクを 12 から徐々に減らしてみよう。

```python
for dim in range(12, 5, -1):
    tt_W1 = TT_SVD(W, [3, dim, 6])
    print("tt_W1:", [v.shape for v in tt_W1])

    print(f"num of params={np.sum([np.prod(v.shape) for v in tt_W1]) + np.prod(B.shape)}")

    val4 = np.einsum("Nkl,ic,cjd,dke,el->Nij", X, *tt_W1) + B
    val4 = val4.reshape(batch_dim, -1)
    print(f"{np.allclose(answer, val4)=} for {dim=}")
    print(f"max diff={np.round(np.max(np.abs(answer - val4)), 5)} for {dim=}")
    print()
```

> tt_W1: [(3, 3), (3, 4, 12), (12, 5, 6), (6, 6)]
> num of params=561
> np.allclose(answer, val4)=True for dim=12
> max diff=0.0 for dim=12
> 
> tt_W1: [(3, 3), (3, 4, 11), (11, 5, 6), (6, 6)]
> num of params=519
> np.allclose(answer, val4)=False for dim=11
> max diff=2.46626 for dim=11
> 
> tt_W1: [(3, 3), (3, 4, 10), (10, 5, 6), (6, 6)]
> num of params=477
> np.allclose(answer, val4)=False for dim=10
> max diff=3.92577 for dim=10
> 
> tt_W1: [(3, 3), (3, 4, 9), (9, 5, 6), (6, 6)]
> num of params=435
> np.allclose(answer, val4)=False for dim=9
> max diff=3.57042 for dim=9
> 
> tt_W1: [(3, 3), (3, 4, 8), (8, 5, 6), (6, 6)]
> num of params=393
> np.allclose(answer, val4)=False for dim=8
> max diff=5.37306 for dim=8
> 
> tt_W1: [(3, 3), (3, 4, 7), (7, 5, 6), (6, 6)]
> num of params=351
> np.allclose(answer, val4)=False for dim=7
> max diff=6.59231 for dim=7
> 
> tt_W1: [(3, 3), (3, 4, 6), (6, 5, 6), (6, 6)]
> num of params=309
> np.allclose(answer, val4)=False for dim=6
> max diff=6.45532 for dim=6

下 2 つのケースでは元のパラメータ数 372 よりも小さくなっている。その代わりに代償として計算誤差も大きくなっている。どこまで計算精度を求めつつ、どこまでパラメータ数を削減したいかのトレードオフであろう。

# オマケ (量子状態の MPS 表現)

[行列積状態について考える (2)](/derwind/articles/dwd-matrix-product02) の頃には量子状態を扱っていたのに、気が付いたらニューラルネットワークになっていた。

折角なので、`TT_SVD` を量子状態にも適用して MPS 表現を見てみよう。

まずは準備をする。

```python
ket_ZERO = np.array([1, 0], dtype=float)
ket_ONE = np.array([0, 1], dtype=float)
```

## $\ket{000}$ で試す

まず、状態ベクトルを用意する。

```python
state_000 = state_000 = np.kron(np.kron(ket_ZERO, ket_ZERO), ket_ZERO)
state_000
```

> array([1., 0., 0., 0., 0., 0., 0., 0.])

次に、TT-分解をしてみる。

```python
mps_state_000 = TT_SVD(state_000.reshape(2, 2, 2))
print([v.shape for v in mps_state_000])
mps_state_000
```

> [(2, 1), (1, 2, 1), (1, 2)]
> [array([[1.],
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.]]),
>&nbsp;array([[[1.],
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0.]]]),
>&nbsp;array([[1., 0.]])]

多少見た目は違うが [行列積状態について考える (2)](/derwind/articles/dwd-matrix-product02) で得た結果に対応しているように思う。

以下の縮約計算で元の状態ベクトルを復元できる。

```python
np.einsum("ia,ajb,bk->ijk", *mps_state_000).flatten()
```

> array([1., 0., 0., 0., 0., 0., 0., 0.])

## $\frac{1}{\sqrt{2}}(\ket{000} + \ket{111})$ で試す

同様に状態ベクトルを用意する。

```python
state_111 = np.kron(np.kron(ket_ONE, ket_ONE), ket_ONE)
state_ghz = (state_000 + state_111) / np.sqrt(2)
state_ghz
```

> array([0.70710678, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.70710678])

次に、TT-分解をしてみる。

```python
mps_state_ghz = TT_SVD(state_ghz.reshape(2, 2, 2))
print([v.shape for v in mps_state_ghz])
mps_state_ghz
```

> [(2, 2), (2, 2, 2), (2, 2)]
> [array([[1., 0.],
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0., 1.]]),
>&nbsp;array([[[ 1.,  0.],
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ 0.,  0.]],
>&nbsp;
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[[ 0.,  0.],
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ 0., -1.]]]),
> &nbsp;array([[ 0.70710678,  0.        ],
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[ 0.        , -0.70710678]])]

こちらも [行列積状態について考える (2)](/derwind/articles/dwd-matrix-product02) で得た結果に対応しているように思う。

以下の縮約計算で元の状態ベクトルを復元できる。

```python
np.einsum("ia,ajb,bk->ijk", *mps_state_ghz).flatten()
```

> array([0.70710678, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.70710678])

# まとめ

[arXiv:1509:06569 Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569) に従う形で、ニューラルネットワークの全結合層を TT-分解を通じて「TT-層」に変換して縮約による順伝播計算をして、通常の線型代数の計算と一致することを確認した。

また、結合次元を下げることでモデル圧縮ができるが、代償として誤差が大きくなることを確認した。

この TT-分解を量子の状態ベクトルに適用すると、既に見た MPS 表現が得られることも確認した。

# 参考文献
[O] [Tensor-Train Decomposition, SIAM J. Sci. Comput., 33(5), 2295–2317. (23 pages), I. V. Oseledets](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition)
[S] [The density-matrix renormalization group in the age of matrix product states, arXiv:1008.3477, Ulrich Schollwoeck](https://arxiv.org/abs/1008.3477)
[NPOV] [Tensorizing Neural Networks, arXiv:1509.06569, Alexander Novikov, Dmitry Podoprikhin, Anton Osokin, Dmitry Vetrov](https://arxiv.org/abs/1509.06569)
