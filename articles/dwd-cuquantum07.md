---
title: "cuQuantum で遊んでみる (7) — 期待値計算再考"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "Python"]
published: true
---

# 目的

[cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) で `cuQuantum` を使った、QAOA 的なハミルトニアンの期待値計算を扱った。この計算は素直で分かりやすいのだが、一方で非効率なところもあるように思う。それは何度も `CircuitToEinsum` を呼び出し、何度も `contract` を呼ぶことである。まずは `contract` の呼び出しを減らせないかを考察してみたい。

# 問題設定

3 量子ビットの量子回路において、ハミルトニアン $H = Z_0 \otimes Z_1 + Z_1 \otimes Z_2$ の期待値を計算するものとする。

状態ベクトルとして $\ket{\psi(\theta)} = R_Y(\theta) \ket{0} \otimes \ket{1} \otimes \ket{0}$ をとり、$\braket{\psi(\theta) | H | \psi(\theta)}$ を計算したい。

$\ket{\psi_\theta} := R_Y(\theta)\ket{0} = \begin{pmatrix} \cos(\theta/2) \\ \sin(\theta/2) \end{pmatrix}$ とおく。この時、求める期待値は直接計算により以下であることがわかる:

$$
\begin{align*}
\braket{\psi | H | \psi} &= \braket{\psi | Z_0 \otimes Z_1 \otimes I_2 + I_0 \otimes Z_1 \otimes Z_2 | \psi} \\
&= \braket{\psi_\theta 1 | Z_0 Z_1 | \psi_\theta 1} + \braket{10 | Z_1 Z_2 | 10} \\
&= -(\cos^2 (\theta/2) - \sin^2 (\theta/2)) - 1 = -1 - \cos \theta
\end{align*}
$$

# Qiskit で確認する

初手は Qiskit で答えを確認するのが楽である。

必要なモジュールを import する。

```python
from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
```

そして期待値計算を行うが、パラメータは $0$, $\frac{\pi}{6}$, $\frac{\pi}{4}$, $\frac{\pi}{3}$, $\frac{\pi}{2}$ を試す。計算結果は丸めた値を掲載している。

```python
qc = QuantumCircuit(3)
qc.ry(Parameter("θ"), 0)
qc.x(1)

H = SparsePauliOp(["IZZ", "ZZI"])
estimator = Estimator()

angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

expvals = estimator.run(
    [qc]*len(angles),
    [H]*len(angles),
    angles.reshape(-1, 1)
).result().values

for angle, expval in zip(angles.tolist(), expvals):
    answer = -1 - np.cos(angle)
    print(f"angle={angle}: expval={expval} answer={answer}")
```

> angle=0.0: expval=-2.0 answer=-2.0
> angle=0.52: expval=-1.87 answer=-1.87
> angle=0.79: expval=-1.71 answer=-1.71
> angle=1.05: expval=-1.5 answer=-1.5
> angle=1.57: expval=-1.0 answer=-1.0

NumPy で直接計算した理論値と一致しているので良さそうである。

# テンソル計算

今回、テンソル計算を手で書こうと思う。つまり、API に頼らないので、心の準備として復習をしておく。練習問題として以下を解こう。

$$
\begin{align*}
\begin{pmatrix}
-1 \ \; -2
\end{pmatrix} \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} \begin{pmatrix}
1 \\
2
\end{pmatrix} &= \begin{pmatrix}
-1 \ \; -2
\end{pmatrix} \begin{pmatrix}
5 \\
11
\end{pmatrix} \\
&= -27
\end{align*}
$$

ベクトルや行列のそれぞれを 1 階と 2 階のテンソルとして以下のように書く:

- $x = (x_i)_{0 \leq i \leq 1} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$
- $A = (A_i^j)_{\substack{0 \leq i \leq 1 \\ 0 \leq j \leq 1}} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$
- $y = (y^j)_{0 \leq j \leq 1} = \begin{pmatrix} -1 \ \; -2 \end{pmatrix}$

すると、テンソル計算としては求めるものは

$$
\begin{align*}
\sum_{\substack{0 \leq i \leq 1 \\ 0 \leq j \leq 1}} x_i A_i^j y^j \ ,
\end{align*}
$$

或は Einstein の縮約記法を用いて単に $x_i A_i^j y^j$ と書ける。

これを用いると練習問題は以下のように解ける:

```python
import cupy as cp
from cuquantum import contract


x = cp.array([1., 2.])
A = cp.array([
    [1., 2.],
    [3., 4.]
])
y = cp.array([-1., -2.])

# 1 つずつ計算する:
Ax = contract("a,ba->b", x, A)
yAx = contract("b,b->", Ax, y)
# 或は以下のように一気に縮約計算しても良い:
yAx_ = contract("a,ba,b->", x, A, y)

print(Ax)
print(yAx)
print(yAx_)
```

> [ 5. 11.]
> -27.0
> -27.0

# 期待値計算に取り組む (cuTensorNet)

## 第一段階

$\braket{\psi | H | \psi} = \braket{\psi_\theta 1 | Z_0 Z_1 | \psi_\theta 1} + \braket{10 | Z_1 Z_2 | 10}$ であったので、別々に計算して和をとるという事が考えられる。これは [cuQuantum で遊んでみる (6) — 最大カット問題と QUBO と QAOA](/derwind/articles/dwd-cuquantum06) と同じであり、部分ハミルトニアンの数だけ `contract` を呼ぶので非効率な可能性がある。

```python
ZERO = cp.array([1., 0.])
ONE = cp.array([0., 1.])
I = cp.eye(2)
Z = cp.array([
    [1., 0.],
    [0., -1.]
])


# R_Y(θ)|0> を返す
def Ry0(theta) -> cp.ndarray:
    return cp.array([np.cos(theta/2), np.sin(theta/2)])


for angle in angles:
    expr1 = ",".join(["a,b", "da,eb", "d,e"]) + "->"
    operands1 = [
        Ry0(angle),
        ONE,
        Z,
        Z,
        Ry0(angle),
        ONE
    ]
    expr2 = ",".join(["b,c", "eb,fc", "e,f"]) + "->"
    operands2 = [
        ONE,
        ZERO,
        Z,
        Z,
        ONE,
        ZERO
    ]
    expval = contract(expr1, *operands1).real + contract(expr2, *operands2).real
    answer = -1 - np.cos(angle)
    print(f"angle={angle}: expval={expval} answer={answer}")
```

ところで、`expr1 = "a,b,da,eb,d,e->"` で `expr2 = "b,c,eb,fc,e,f->"` なので内容が異なる。このためテンソルをまとめるということが難しいのだが、この内容を一致させる方法がある。それは省略された恒等行列 `I` を補うことで達成される。

## 第二段階

恒等行列 `I` を補うことで、計算と逆計算でキャンセルされて何も起こらないネットワークパスが生まれるという多少の無駄が出るが、その代わりに `expr = "a,b,c,da,eb,fc,d,e,f->"` を共通化できるようになる。

```python
for angle in angles:
    expr = ",".join(["a,b,c", "da,eb,fc", "d,e,f"]) + "->"
    operands1 = [
        Ry0(angle),
        ONE,
        ZERO,
        Z,
        Z,
        I,
        Ry0(angle),
        ONE,
        ZERO
    ]
    operands2 = [
        Ry0(angle),
        ONE,
        ZERO,
        I,
        Z,
        Z,
        Ry0(angle),
        ONE,
        ZERO
    ]
    expval = contract(expr, *operands1).real + contract(expr, *operands2).real
    answer = -1 - np.cos(angle)
    print(f"angle={angle}: expval={expval} answer={answer}")
```

ここで、

```python
        Z,
        Z,
        I,
```

が $Z_0 \otimes Z_1 \otimes I_2$ に対応し、

```python
        I,
        Z,
        Z,
```

が、$I_0 \otimes Z_1 \otimes Z_2$ に対応している。

この段階ではまだ `contract` を呼ぶ回数は減らせていないが、次の段階で部分ハミルトニアンをすべて “積み上げて” 大きなテンソルにすることで、`contract` の呼び出し回数をたった 1 回に減らすことができる。

## 第三段階

ハミルトニアンをただ 1 つの大きなテンソルにする。このために第二段階でやったことを数式で書き下す。

### 第二段階を数式でレビューする

```python
        Ry0(angle),
        ONE,
        ZERO,
```

の部分をテンソル $A = (A_{abc})$ と書く。

```python
        Z,
        Z,
        I,
```

の部分をテンソル $R = (R_{abc}^{def})$ と書く。

```python
        I,
        Z,
        Z,
```

の部分をテンソル $S = (S_{abc}^{def})$ と書く。

```python
        Ry0(-angle),
        ONE,
        ZERO
```

の部分をテンソル $B = (B^{def})$ と書く。

このように書くと、第二段階は

$$
\begin{align*}
\sum_{a,b,c,d,e,f}\!\!\! A_{abc} R_{abc}^{def} B^{def} +\!\!\! \sum_{a,b,c,d,e,f}\!\!\! A_{abc} S_{abc}^{def} B^{def} =\!\!\! \sum_{a,b,c,d,e,f}\!\!\! A_{abc} (R_{abc}^{def} + S_{abc}^{def}) B^{def}
\tag{1}
\end{align*}
$$

を計算していることになる。ここで $R_{abc}^{def} + S_{abc}^{def}$ をなんとか 1 つのテンソル表記ができると非常に嬉しい。そしてそれは可能である。

このためにインデックス $g$ を導入し、テンソル $T = ({}^{g}T_{abc}^{def})$ を考える。特に、${}^{0}T_{abc}^{def} = R_{abc}^{def}$, ${}^{1}T_{abc}^{def} = S_{abc}^{def}$ とする。この時、

$$
\begin{align*}
R_{abc}^{def} + S_{abc}^{def} = \sum_g {}^{g}T_{abc}^{def}
\tag{2}
\end{align*}
$$

となることに注意して欲しい。

式 (1) と (2) を併せることで、求めたい期待値は以下のテンソル計算で書けることが分かった。

$$
\begin{align*}
\sum_{a,b,c,d,e,f,g}\!\!\! A_{abc} {}^{g}T_{abc}^{def} B^{def}
\end{align*}
$$

### 実装

部分ハミルトニアンの “積み上げ” は横方向に行うので多少注意が必要だが、以下のようになる。

```python
for angle in angles:
    expr = ",".join(["a,b,c", "gda,geb,gfc", "d,e,f"]) + "->"
    operands = [
        Ry0(angle),
        ONE,
        ZERO,
        cp.array([Z, I]),
        cp.array([Z, Z]),
        cp.array([I, Z]),
        Ry0(angle),
        ONE,
        ZERO
    ]
    expval = contract(expr, *operands).real
    answer = -1 - np.cos(angle)
    print(f"angle={angle}: expval={expval} answer={answer}")
```

# まとめ

実際に計算を高速化させることができるかは別問題として、`contract` の呼び出し回数をたった 1 回にまで落とす計算について触れた。

実際の QAOA の計算に適用するには、ハミルトニアン部分の処理をうまくやらないとならないため実装が多少手間なのだが、試してみたいと思う。

# Appendix (ハミルトニアンが係数を持つ場合)

ハミルトニアンは $H = Z_0 \otimes Z_1 + Z_1 \otimes Z_2$ ではなく、係数がかかって $H^\prime = 2 Z_0 \otimes Z_1 -3 Z_1 \otimes Z_2$ のようなものかもしれない。このような場合は少し扱いにくそうに見えるかもしれないが、実は簡単な改造で対応できる。

現在 `expr = "a,b,c,gda,geb,gfc,d,e,f->"` であるが、`expr = "a,b,c,gda,geb,gfc,d,e,f->g"` のように変更し、インデックス `g` については縮約しないようにすれば良い。こうすると計算結果が部分ハミルトニアンに対する部分期待値を要素に持つような 1 階のテンソルになるので、係数を掛けながら和をとれば良い。これは要素ごとのアダマール積をとって和をとることに相当するが、`CuPy` や `NumPy` はそういった計算は得意である。
