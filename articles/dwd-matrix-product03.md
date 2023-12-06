---
title: "行列積状態について考える (3)"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python"]
published: true
---

# 目的

行列積状態 (Matrix Product State; MPS) について以前に書いた [行列積状態について考える (2)](/derwind/articles/dwd-matrix-product02) の続きとして、もっと一般のテンソルでの Tensor-Train 分解 (量子化学の分野などでの呼称は MPS) を考えたい。

要するに [SIAM J. Sci. Comput., 33(5), 2295–2317. (23 pages) Tensor-Train Decomposition, I. V. Oseledets](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition) の内容を実装したいということになる。

# TT-分解 (Tensor-Train 分解)

一般に $A(i_1,i_2, \cdots, i_d) \ \ (1 \leq i_k \leq n_k)$ を要素に持つような $d$-階のテンソルが与えられた時に、その各要素が

$$
\begin{align*}
A(i_1,i_2, \cdots, i_d) = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} G_1(\alpha_0,i_1,\alpha_1) G_2(\alpha_1,i_2,\alpha_2) \cdots G_d(\alpha_{d-1},i_d,\alpha_d)
\end{align*}
$$

となるような 3 階のテンソルの列 $\{ G_1, G_2, \cdots, G_d \}$ を求めたいと言うものになる。各 $G_k(\alpha_{k-1},i_k,\alpha_k)$ は次元として $r_{k-1} \times n_k \times r_k$ を持つ 3 階のテンソルで、「コアテンソル」と呼ばれる。$r_k$ は隣接するコアテンソルとの間の次元であり「結合次元」と呼ばれる。但し、“境界条件” として $r_0 = r_d = 1$ を課す。

これは文献 [The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477) と記号を合わせるなら

$$
\begin{align*}
A_{i_1 i_2 \cdots i_d} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} G^{i_1}_{\alpha_1} G^{i_2}_{\alpha_1,\alpha_2} \cdots G^{i_d}_{\alpha_{d-1}}
\end{align*}
$$

と書いても良いかもしれない。こちらのほうはより「行列積」感が出そうな気がする。こちらの論文では、量子状態を計算基底について書き下した時に $\ket{0}^{\otimes n}$ から $\ket{1}^{\otimes n}$ までの $2^n$ 個の基底ベクトルの係数、即ち確率振幅を集めてきて作った数列 (或はベクトル) を `reshape` してテンソルとみなし、これを TT-分解するという内容になっている。

TT-分解の証明自体は Theorem 2.1 にて SD (Skeleton decomposition) という行列分解を用いて構成的になされる。また、これ自体は元のテンソルを完全に復元する厳密な分解であるが、近似アルゴリズムとしての TT-SVD アルゴリズムを誘導する。結合次元を絞り込むことで情報圧縮をしながら元のテンソルを近似することになる。

この TT-SVD アルゴリズム自体は論文の p.2301 に掲載されている。

# 実装例 (Qiita 記事より)

さて、実はこの TT-SVD アルゴリズムについての実装例が既に存在していて、それは [Rでtensor train decomposition してみた](https://qiita.com/Yh_Taguchi/items/0be41d2b1c0bd8ea0e6b) に R 言語で書かれている。

R 言語は全く知らないので、ここは AI パワーにすがるという形で GitHub Copilot の言語変換機能を試してみた。少々粗はあったのだが、多少の手直しをした結果の Python コードは以下のようになる:

```python
from __future__ import annotations

from typing import Sequence
import numpy as np


def TT(x: np.ndarray, r: Sequence[int]):
    D = x.shape
    n = len(D)

    LIST = [None] * n
    for i in range(n-1):
        if i == 0:
            l0 = 1
        else:
            l0 = r[i-1]
        l1 = r[i]
        x = np.reshape(x, (D[i] * l0, np.prod(D[i+1:n])))
        U, _, _ = np.linalg.svd(x, full_matrices=False)
        LIST[i] = np.reshape(U[:, :l1], (l0, D[i], l1))
        x = np.transpose(U[:, :l1]).dot(x)
    LIST[n-1] = np.transpose(x)
    LIST[0] = np.reshape(LIST[0], (D[0], r[0]))
    return LIST
```

雰囲気的に似ているかな？と思ったのが初見の印象であった。

続けて、同 Qiita の記事の実験を再現してみたい。R の多次元配列の読み方が分からなくて試行錯誤したが、結局は以下のように読み換えたら良かったようである。

```python
Z = np.array([
    [
        [1.262954, -1.1476570, -0.05710677, 0.9921604, 0.83204713],
        [1.272429, -0.4115108, -0.69095384, -0.2793463, -0.37670272],
        [-0.928567, 0.4356833, -0.23570656, -0.4527840, -0.05487747],
        [2.404653, 0.3773956, -0.64947165, -1.0655906, -0.17262350],
    ],
    [
        [-0.3262334, -0.2894616, 0.5036080, -0.4295131, -0.2273287],
        [0.4146414, 0.2522234, -1.2845994, 1.7579031, 2.4413646],
        [-0.2947204, -1.2375384, -0.5428883, -0.8320433, 0.2501413],
        [0.7635935, 0.1333364, 0.7267507, -1.5637821, -2.2239003],
    ],
    [
        [1.329799263, -0.2992151, 1.08576936, 1.2383041, 0.2661374],
        [-1.539950042, -0.8919211, 0.04672617, 0.5607461, -0.7953391],
        [-0.005767173, -0.2242679, -0.43331032, -1.1665705, 0.6182433],
        [-0.799009249, 0.8041895, 1.15191175, 1.1565370, -1.2636144],
    ],
])
```

因みにこの記事では結合次元を `c(2,3)` で与えているが、元のテンソルが小さいので、結構影響を受けてしまい、近似値のような近似値とも言えないようなテンソルが得られるようである。`c(3,4)` くらいがまぁまぁ近似値で、`c(3,5)` で元のテンソルが復元できる。

## 検証

```python
bond_dims = (3, 5)
tt_list = TT(Z, bond_dims)

Z1 = np.einsum("ia,ajb,kb->ijk", *tt_list)
np.allclose(Z, Z1)
```

> True

という感じで元のテンソルを復元できたことを検証できる。

次にややマイルドな近似として以下を実行すると

```python
TT(Z, (3, 4))
```

```
array([[[ 1.26321438, -1.14774143, -0.06648278,  0.99668222,  0.82531518],
        [ 1.25868211, -0.40705353, -0.19594873, -0.51807543, -0.02129007],
        [-0.92899622,  0.43582247, -0.2202511 , -0.4602378 , -0.04378048],
        [ 2.39952597,  0.37905798, -0.4648549 , -1.15462684, -0.04006905]],

       [[-0.31743688, -0.29231377,  0.18685824, -0.27675227, -0.45475438],
        [ 0.40846868,  0.25422483, -1.06232878,  1.6507073 ,  2.60095445],
        [-0.29416419, -1.23771874, -0.56291648, -0.82238419,  0.23576111],
        [ 0.76143661,  0.13403575,  0.80441711, -1.60123875, -2.16813598]],

       [[ 1.33996203, -0.30251026,  0.71982315,  1.41479121,  0.00338877],
        [-1.55201823, -0.88800813,  0.4812836 ,  0.35116944, -0.48332776],
        [ 0.0033049 , -0.22720941, -0.75998201, -1.00902456,  0.3836937 ],
        [-0.80351403,  0.80565012,  1.31412219,  1.07830678, -1.14714764]]])
```

くらいが得られるが、元のテンソルと比較するとそこそこ近いのが分かると思う。

コアテンソル同士の結合次元を絞っていくと、コアテンソルの次元は

> (3, 3) (3, 4, 5) (5, 5): 低圧縮
> (3, 3) (3, 4, 4) (5, 4): 中圧縮
> (3, 2) (2, 4, 3) (5, 3): 高圧縮

という感じになるので、こんな小さなテンソルの分解ではかえってパラメータ量が増えているので分かりにくいが、大きいテンソルでうまくやれば情報量が低→中→高で減らせそうな雰囲気が見えないわけでもない。

# 実装例 (論文そのまま)

さて、Qiita 記事の実装はよく見ると論文に書いてある式そのままではないように見える部分もあるので、愚直な実装も行ってみよう:

```python
def TT_SVD(C: np.ndarray, r: Sequence[int]) -> list[np.ndarray]:
    """TT_SVD algorithm

    Args:
        C (np.ndarray): n-dimensional input tensor
        r (Sequence[int]): a list of bond dimensions

    Returns:
        np.ndarray: a list of core tensors of TT-decomposition
    """

    dims = C.shape
    n = len(dims)  # n-dimensional tensor
    if len(r) != n - 1:
        raise ValueError(f"{len(r)=} must be {n - 1}.")

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
    tt_cores.append(C.T)
    tt_cores[0] = tt_cores[0].reshape(dims[0], r[0])
    return tt_cores
```

Qiita 記事に倣って結合次元を外から `r` で与えている以外は見た目的にも論文そのものになっているのではないだろうか？

なお、Qiita の実装もこの実装もそうだが、“境界条件” として $r_0 = r_d = 1$ を課していたので次元が 1 のインデックスがある。これについては `squeeze` したような感じで最初から落している。

さて、これは Qiita の記事由来のものと一致するのであろうか？

## 検証

```python
# strict
strict_bond_dims = (3, 5)

# medium
medium_bond_dims = (3, 4)

# loose
loose_bond_dims = (2, 3)

for bond_dims in (strict_bond_dims, medium_bond_dims, loose_bond_dims):
    tt_list = TT(Z, bond_dims)
    tt_list2 = TT_SVD(Z, bond_dims)
    for core1, core2 in zip(tt_list, tt_list2):
        print(np.allclose(core1, core2))
    print("-" * 10)
```

```
True
True
True
----------
True
True
True
----------
True
True
True
----------
```

ということで十分に一致した。

# まとめ

より一般のテンソルに対する Tensor-Train 分解について論文と Qiita の記事を通じて確認した。

論文を読むと、かなりうっ・・・となる内容であるが、どちらかと言うとそう感じる部分は NumPy で言う `reshape` の手続きを自然言語で細かく説明している部分で、コードに落して実装するとこれくらいのもののようである。

# 参考文献

[O] [Tensor-Train Decomposition, SIAM J. Sci. Comput., 33(5), 2295–2317. (23 pages), I. V. Oseledets](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition)
[S] [The density-matrix renormalization group in the age of matrix product states, arXiv:1008.3477, Ulrich Schollwoeck](https://arxiv.org/abs/1008.3477)
[T] [Rでtensor train decomposition してみた, Qiita, @Yh_Taguchi](https://qiita.com/Yh_Taguchi/items/0be41d2b1c0bd8ea0e6b)
