---
title: "行列積状態について考える (13) — シンボリック計算"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "sympy", "TensorNetwork"]
published: false
---

# 目的

[行列積状態について考える (12) — テンソルネットワークで MNIST 分類](/derwind/articles/dwd-matrix-product12) の実験をやっていた時にあまりに辛すぎて手計算できなかったので、`sympy` でシンボリック計算をして検証していた。滅多にやらなくて忘れるので備忘録として残そうというもの。

# 実装

## 必要なモジュールを import

```python
from __future__ import annotations

import numpy as np
import sympy
from opt_einsum import contract
```

## シンボルでテンソルを構築する関数の定義

雰囲気で実装したので、正直よく覚えていない。数式で書くときにインデックスが 0 から始まると何となく気持ち悪いことがあるので、急遽フラグを追加してみた。

```python
def create_symbol_tensor(
    name_base: str, shape: int | tuple[int, ...], start_at_index_1: bool = False
) -> np.ndarray:
    def create_tensor_names(name_base, shape):
        tensor_names = []

        def loop(indices: tuple[int], depth: int = 0):
            for i in range(shape[depth]):
                if depth + 1 < len(shape):
                    idx = i + 1 if start_at_index_1 else i
                    loop((*indices, idx), depth + 1)
                    continue
                idx = i + 1 if start_at_index_1 else i
                suffix = "-".join(map(str, indices + (idx,)))
                name = name_base + "_{" + suffix + "}"
                tensor_names.append(name)
        loop((), 0)
        return tensor_names

    if isinstance(shape, int):
        shape = (shape,)
    tensor_names = create_tensor_names(name_base, shape)
    symbols = sympy.symbols(" ".join(tensor_names))
    for sym in symbols:
        sym.name = sym.name.replace("-", ",")
    return np.array(symbols).reshape(shape)
```

# 実験

## 行列の計算

$A = (a_{ij})_{1 \leq i \leq 4,1 \leq j \leq 5}$, $x = (x_j)_{1 \leq j \leq 5}$ の掛け算をやってみたい。つまり、$\sum_{j=1}^5 A_{ij} x_{j}$ をやってみたい。

```python
A = create_symbol_tensor("a", (4, 5), True)
x = create_symbol_tensor("x", 5, True)

y = contract("ij,j->i", A, x)

for i in range(len(y)):
    display(y[i])
```

> $a_{1,1} x_{1} + a_{1,2} x_{2} + a_{1,3} x_{3} + a_{1,4} x_{4} + a_{1,5} x_{5}$
> $a_{2,1} x_{1} + a_{2,2} x_{2} + a_{2,3} x_{3} + a_{2,4} x_{4} + a_{2,5} x_{5}$
> $a_{3,1} x_{1} + a_{3,2} x_{2} + a_{3,3} x_{3} + a_{3,4} x_{4} + a_{3,5} x_{5}$
> $a_{4,1} x_{1} + a_{4,2} x_{2} + a_{4,3} x_{3} + a_{4,4} x_{4} + a_{4,5} x_{5}$

良さそう。

## テンソルの計算

ここでわざわざ $\alpha$ や $\beta$ を使いたいというだけの理由で `opt_einsum.contract` を使っているが、`numpy.einsum` で普通に動く。

```python
shapes = [(2,1), (1,2,1), (1, 2)]
x = [create_symbol_tensor(f"x^{i+1}", (2,), True) for i in range(len(shapes))]
G = [create_symbol_tensor(f"G^{i+1}", shapes[i], True) for i in range(len(shapes))]

contracted_tensor = contract("A,B,C,Aα,αBβ,βC->", *x, *G)
display(contracted_tensor.item())
```

> $\left(G^1_{1,1} x^1_{1} + G^1_{2,1} x^1_{2}\right) \left(G^2_{1,1,1} x^2_{1} + G^2_{1,2,1} x^2_{2}\right) \left(G^3_{1,1} x^3_{1} + G^3_{1,2} x^3_{2}\right)$

```python
display(sympy.expand(contracted_tensor.item()))
```

> $G^1_{1,1} G^2_{1,1,1} G^3_{1,1} x^1_{1} x^2_{1} x^3_{1} + G^1_{1,1} G^2_{1,1,1} G^3_{1,2} x^1_{1} x^2_{1} x^3_{2} + G^1_{1,1} G^2_{1,2,1} G^3_{1,1} x^1_{1} x^2_{2} x^3_{1} + G^1_{1,1} G^2_{1,2,1} G^3_{1,2} x^1_{1} x^2_{2} x^3_{2} + G^1_{2,1} G^2_{1,1,1} G^3_{1,1} x^1_{2} x^2_{1} x^3_{1} + G^1_{2,1} G^2_{1,1,1} G^3_{1,2} x^1_{2} x^2_{1} x^3_{2} + G^1_{2,1} G^2_{1,2,1} G^3_{1,1} x^1_{2} x^2_{2} x^3_{1} + G^1_{2,1} G^2_{1,2,1} G^3_{1,2} x^1_{2} x^2_{2} x^3_{2}$

小さなテンソルの縮約ではあるが、結構な項の数になることが分かる。

普段は数式として

$$
\begin{align*}
\sum_{i,j,k,\alpha,\beta} G^1_{i \alpha} G^2_{\alpha j \beta} G^3_{\beta k} x^1_i x^2_j x^3_k
\end{align*}
$$

くらいしか書かないので、計算機での計算を意識することがほとんどないのだが、項数とかをちゃんと見積ったり値の範囲を見積らないと、うっかりオーバーフローが生じそうだなという気持ちになる。

# まとめ

手計算でやると心が折れるものの視覚的に確認したいと言う場合に、シンボリック計算が結構役に立つ。
