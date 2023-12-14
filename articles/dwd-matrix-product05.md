---
title: "行列積状態について考える (5)"
emoji: "⛓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python"]
published: false
---

# 目的

[行列積状態について考える (3)](/derwind/articles/dwd-matrix-product03) の続きとして、ニューラルネットワークのパラメータ圧縮について考えたい。[arXiv:1509:06569 Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569) に沿って考えたい。

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

Tensor-Train 中の各結合の次元を $r_1$, $r_2$, $r_3$, $r_4$, $r_5$ より小さくするとパラメータ数を削減できる、つまりニューラルネットワークの「枝刈り」の一種を実現できる。

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
