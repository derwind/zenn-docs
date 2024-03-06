---
title: "NumPy もどきを作る (1)"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "numpy"]
published: false
---

# 目的

去年作っていた NumPy もどきがある程度動くようになったので備忘録を残しておく。

- 何となく NumPy っぽく使えること
- 縮約計算 (`einsum`) が使えること
- データ構造の中心は多次元リスト (`int`, `float`, `complex`)

程度が目的で、NumPy を触りながら動きの雰囲気を合わせていった。細かいインターフェイスを知りたい場合は [NumPy reference](https://numpy.org/doc/stable/reference/index.html) なども参考にした。

リポジトリ: https://github.com/derwind/mynumpy

# 実装内容

実装は 2 段階のゴールからなる。

1. 縮約計算 (`einsum`) が使える
1. 基本的な MPS (行列積) 計算に使える

## v0.1

とりあえずそれっぽく使えるメソッド群を実装。

- `array`
- `zeros`
- `zeros_like`
- `ndarray`
    - `__str__`
    - `__repr__`
    - `__eq__`
    - `__ne__`
    - `__add__`
    - `__sub__`
    - `__mul__`
    - `__matmul__` (制限あり)
    - `__truediv__`
    - `__len__`
    - `ndim`
    - `shape`
    - `size`
    - `T`
    - `flatten`
    - `reshape`

そして、`broadcast` (テンソル + スカラーのみ) まで実装した。

## v0.2

最低限の縮約計算を実装。

- `einsum` (2 個のテンソルのみ)
    - ベクトル, 行列 → ベクトル (`"ij,j->i"`)
    - 行列, 行列 → 行列 (`"ij,jk->ik"`)
    - 高階テンソル, 高階テンソル → 高階テンソル

## v0.3

- `ones`
- `ones_like`
- `einsum` (2 個のテンソルのみ)
    - トレース (`"ii->"`)
    - 内積 (`"i,i->"`)
    - 転置  (`"ij->ji"`)

## v0.4

- `broadcast` 改善

## v0.5

とりあえず完成。

- `__getitem__` (ファンシーインデックス除く)
- `__setitem__` (ファンシーインデックス除く)

## v0.6

MPS 計算用に拡張開始。

- `einsum` (複数個のテンソル)
- `ndarray`
    - `dtype`
    - `real`
- 数学関数
    - `log`, `log2`, `log10`, `logn`, `sqrt`, `power`, `cos`, `sin`, `tan`, `arccos`, `arcsin`, `arctan`, `arctan2`, `cosh`, `sinh`, `tanh`, `arccosh`, `arcsinh`, `arctanh` (スカラーのみ)

## v0.7

回転ゲートのような複素数を伴うゲートを除いた MPS 計算用に線形代数演算を実装。

- `eye`
- `allclose`
- `ndarray`
    - `conj`
    - `copy`
    - `astype`
    - `tolist`
- `linalg`
    - `norm`
    - `matrix_rank` (実行列のみ)
    - `svd` (実行列のみ)

# コア実装

## 多次元リストの走査

基本的な計算のコア部分となるのは、多次元リストを再帰的に潜りながら処理を実行する部分である。よって以下のような処理を繰り返し用いている。

```python
Numbers = Union[int, float, complex]


def walk(data: list, func:  Callable[[Numbers], Numbers]):
    if is_number(data[0]):
        for i, d in enumerate(data):
            data[i] = func(d)
        return

    for subdata in data:
        walk(subdata, func)
```

## `einsum` 実装

- 2 個のテンソルに対する計算を気合で実装
    - 入れ物を用意
    - 和がわたるインデックスを求めて順に値をとって掛け合わせて可算して入れ物に格納
- 複数個のテンソルについては以下のように分解して 2 個ずつのテンソルの縮約計算の繰り返しで計算する
    - `"Aa,aBb,bC->ABC"` => `"Aa,aBb->ABb"` & `"ABb,bC->ABC"`

## `linalg` 実装

中心となるのは特異値分解 (SVD) の実装である。実行列に対する SVD の実装は片側 Jacobi 法が [Jacobi's method is more accurate than QR](https://www.netlib.org/lapack/lawnspdf/lawn15.pdf) Algorithm 4.1 に載っているのでこれを用いた。ここでは正方行列のケースが書かれているが、正方行列でないケースへの拡張を含め [convexbrainのけんきうメモ](https://convexbrain.github.io/studynotes/SVD) に解説がある。

$m \geq n$ に対する $G \in \operatorname{Mat}(m,n; \R)$ の SVD は

$$
\begin{align*}
U, \Sigma, V = \operatorname{svd}(G)
\end{align*}
$$

が求まり、$G = U \Sigma V^T$ で復元できる。ここで $\Sigma$ は非負の実対角行列である。

$m < n$ の場合には、$G^\prime = G^T$ を使うと、$G^\prime \in \operatorname{Mat}(n,m; \R)$ となるので

$$
\begin{align*}
U, \Sigma, V = \operatorname{svd}(G^\prime)
\end{align*}
$$

に対して $G^\prime = U \Sigma V^T$ で復元できる。よって、$G = V \Sigma U^T$ となる。つまり、SVD 計算での $U$ と $V$ の役割を入れ替えれば良いことになる。

# まとめ

最初は NumPy のエミュレートができるかは相当に懐疑的であったが、やってみて徐々に実装できるところから取り組んでみると、案外実装できることが分かった。まだ機能不足である感は否めないが、これくらいの機能があればある程度の計算は実行できる。
