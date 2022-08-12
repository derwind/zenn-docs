---
title: "Bloch ベクトルを眺めてみる (2)"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["poem", "Python"]
published: true
---

# 目的

[前回](/derwind/articles/dwd-bloch-vector)は $R_X$ ゲートの場合のみ計算したので、残りの $R_Y$, $R_Z$ ゲートの場合を確認したい。と言っても、もう手計算はつらいので、`sympy` を活用する。

# 準備

以下のように計算に必要なシンボル類を定義する。

```python
import sympy

theta = sympy.Symbol('θ', real=True)
phi = sympy.Symbol('φ', real=True)
lam = sympy.Symbol('λ', real=True)

I = sympy.Matrix([
    [1, 0],
    [0, 1]
])
X = sympy.Matrix([
    [0, 1],
    [1, 0]
])
Y = sympy.Matrix([
    [0, -1.j],
    [1.j, 0]
])
Z = sympy.Matrix([
    [1, 0],
    [0, -1]
])
def Rx(t):
    return sympy.cos(t/2)*I - 1.j*sympy.sin(t/2)*X
def Ry(t):
    return sympy.cos(t/2)*I - 1.j*sympy.sin(t/2)*Y
def Rz(t):
    return sympy.cos(t/2)*I - 1.j*sympy.sin(t/2)*Z

def density_matrix(state):
    return state * sympy.conjugate(state.T)

Zero = sympy.Matrix([[1], [0]])
One = sympy.Matrix([[0], [1]])
```

# 前回のおさらい

`sympy` が今回の計算の役に立つかを確認するために、[前回](/derwind/articles/dwd-bloch-vector)の計算をさせてみよう。

まず、基本となる状態ベクトルを定義する:

```python
state = sympy.cos(theta/2)*Zero + sympy.exp(1.j*phi)*sympy.sin(theta/2)*One
display(state)
```

> $\left[\begin{matrix}\cos{\left(\frac{θ}{2} \right)}\\e^{1.0 i φ} \sin{\left(\frac{θ}{2} \right)}\end{matrix}\right]$

```python
rho = density_matrix(state)
display(rho)
```

> $\left[\begin{matrix}\cos^{2}{\left(\frac{θ}{2} \right)} & e^{- 1.0 i φ} \sin{\left(\frac{θ}{2} \right)} \cos{\left(\frac{θ}{2} \right)}\\e^{1.0 i φ} \sin{\left(\frac{θ}{2} \right)} \cos{\left(\frac{θ}{2} \right)} & \sin^{2}{\left(\frac{θ}{2} \right)}\end{matrix}\right]$

ここまではかなりそれっぽい結果になる。

```python
rho_out = density_matrix(Rx(lam)@rho)

x = sympy.simplify((X@rho_out).trace())
y = sympy.simplify((Y@rho_out).trace())
z = sympy.simplify((Z@rho_out).trace())

display(x)
display(y)
display(z)
```

> $1.0 \sin{\left(θ \right)} \cos{\left(1.0 φ \right)}$
> $- 0.5 i e^{1.0 i φ} \sin{\left(θ \right)} \cos{\left(λ \right)} - 1.0 \sin{\left(λ \right)} \cos{\left(θ \right)} + 0.5 i e^{- 1.0 i φ} \sin{\left(θ \right)} \cos{\left(λ \right)}$
> $- 0.5 i e^{1.0 i φ} \sin{\left(θ \right)} \sin{\left(λ \right)} + 1.0 \cos{\left(θ \right)} \cos{\left(λ \right)} + 0.5 i e^{- 1.0 i φ} \sin{\left(θ \right)} \sin{\left(λ \right)}$

となった。`sympy.simplify` にも限界はあるだろうし、ここから先は手計算で更に整理しよう。

$$
\begin{align*}
x^\prime &= \sin \theta \cos \phi \\
y^\prime &= - \frac{1}{2} i e^{i \phi} \sin \theta \cos \lambda - \sin \lambda \cos \theta + \frac{1}{2} i e^{-i \phi} \sin \theta \cos \lambda \\
&= \cos \lambda \sin \theta \sin \phi - \sin \lambda \cos \theta \\
z^\prime &= - \frac{1}{2} i e^{i \phi} \sin \theta \sin \lambda + \cos \theta \cos \lambda + \frac{1}{2} i e^{-i \phi} \sin \theta \sin \lambda \\
&= \sin \lambda \sin \theta \sin \phi + \cos \lambda \cos \theta
\tag{1}
\end{align*}
$$

となる。(1) 式は前回の計算結果と一致している。かなり簡単に計算できることが分かった。

感触を確かめたので、$R_Y$, $R_Z$ ゲートのケースも確認してみよう。

以下で使うので、状態ベクトル $\ket{\psi} = \cos \frac{\theta}{2} \ket{0} + e^{i \phi} \sin \frac{\theta}{2} \ket{1}$ に対応する Bloch ベクトル $\vec{r} = \frac{1}{2} (I + xX + yY + zZ)$ の係数を再掲しよう:

$$
\begin{align*}
x &= \sin \theta \cos \phi \\
y &= \sin \theta \sin \phi \\
z &= \cos \theta
\end{align*}
$$

# $R_Y$ の場合

```python
rho_out = density_matrix(Ry(lam)@rho)

x = sympy.simplify((X@rho_out).trace())
y = sympy.simplify((Y@rho_out).trace())
z = sympy.simplify((Z@rho_out).trace())

display(x)
display(y)
display(z)
```

> $0.5 e^{1.0 i φ} \sin{\left(θ \right)} \cos{\left(λ \right)} + 1.0 \sin{\left(λ \right)} \cos{\left(θ \right)} + 0.5 e^{- 1.0 i φ} \sin{\left(θ \right)} \cos{\left(λ \right)}$
> $1.0 \sin{\left(θ \right)} \sin{\left(1.0 φ \right)}$
> $- 0.5 e^{1.0 i φ} \sin{\left(θ \right)} \sin{\left(λ \right)} + 1.0 \cos{\left(θ \right)} \cos{\left(λ \right)} - 0.5 e^{- 1.0 i φ} \sin{\left(θ \right)} \sin{\left(λ \right)}$

$$
\begin{align*}
x^\prime &= \frac{1}{2} e^{i \phi} \sin \theta \cos \lambda + \sin \lambda \cos \theta + \frac{1}{2} e^{-i \phi} \sin \theta \cos \lambda \\
&= \sin \lambda \cos \theta + \cos \lambda \sin \theta \cos \phi \\
y^\prime &= \sin \theta \sin \phi \\
z^\prime &= - \frac{1}{2} e^{i \phi} \sin \theta \sin \lambda + \cos \theta \cos \lambda - \frac{1}{2} e^{-i \phi} \sin \theta \sin \lambda \\
&= \cos \lambda \cos \theta - \sin \lambda \sin \theta \cos \phi
\tag{2}
\end{align*}
$$

となる。よって、$y^\prime = y$ であり、

$$
\begin{align*}
\begin{bmatrix}
z^\prime \\ x^\prime
\end{bmatrix} = \begin{bmatrix}
\cos \lambda & - \sin \lambda \\
\sin \lambda & \cos \lambda
\end{bmatrix} \begin{bmatrix}
z \\ x
\end{bmatrix}
\end{align*}
$$

となっていることが分かる。つまり、ZX 平面内で角度 $\lambda$ の回転が適用された形である。

# $R_Z$ の場合

```python
rho_out = density_matrix(Rz(lam)@rho)

x = sympy.simplify((X@rho_out).trace())
y = sympy.simplify((Y@rho_out).trace())
z = sympy.simplify((Z@rho_out).trace())

display(x)
display(y)
display(z)
```

> $1.0 \left(i e^{2.0 i φ} \sin{\left(λ \right)} + e^{2.0 i φ} \cos{\left(λ \right)} - i \sin{\left(λ \right)} + \cos{\left(λ \right)}\right) e^{- 1.0 i φ} \sin{\left(\frac{θ}{2} \right)} \cos{\left(\frac{θ}{2} \right)}$
> $1.0 i \left(- i e^{2.0 i φ} \sin{\left(λ \right)} - e^{2.0 i φ} \cos{\left(λ \right)} - i \sin{\left(λ \right)} + \cos{\left(λ \right)}\right) e^{- 1.0 i φ} \sin{\left(\frac{θ}{2} \right)} \cos{\left(\frac{θ}{2} \right)}$
> $1.0 \cos{\left(θ \right)}$

$$
\begin{align*}
x^\prime &= (i e^{2 i \phi} \sin \lambda + e^{2 i \phi} \cos \lambda -i \sin \lambda + \cos \lambda) e^{-i \phi} \sin \frac{\theta}{2} \cos \frac{\theta}{2} \\
&= (-2 \sin \lambda \sin \phi + 2 \cos \lambda \cos \phi) \sin \frac{\theta}{2} \cos \frac{\theta}{2} \\
& = \cos \lambda \sin \theta \cos \phi - \sin \lambda \sin \theta \sin \phi \\
y^\prime &= i (- i e^{2 i \phi}\sin \lambda - e^{2 i \phi} \cos \lambda -i \sin \lambda + \cos \lambda) e^{-i \phi} \sin \frac{\theta}{2} \cos \frac{\theta}{2} \\
&= (2 \sin \lambda \cos \phi + 2 \cos \lambda \sin \phi) \sin \frac{\theta}{2} \cos \frac{\theta}{2} \\
&= \sin \lambda \sin \theta \cos \phi + \cos \lambda \sin \theta \sin \phi \\
z^\prime &= \cos \theta
\tag{3}
\end{align*}
$$

となる。よって、$z^\prime = z$ であり、

$$
\begin{align*}
\begin{bmatrix}
x^\prime \\ y^\prime
\end{bmatrix} = \begin{bmatrix}
\cos \lambda & - \sin \lambda \\
\sin \lambda & \cos \lambda
\end{bmatrix} \begin{bmatrix}
x \\ y
\end{bmatrix}
\end{align*}
$$

となっていることが分かる。つまり、XY 平面内で角度 $\lambda$ の回転が適用された形である。

# まとめ

$\R^3$ における作用としては、$R_X$ は YZ 平面内で、$R_Y$ は ZX 平面内で、$R_Z$ は XY 平面内での回転を Bloch ベクトルに適用することが分かった。

`sympy` の力強さも確認できたし、これからは回転ゲートを適用する時に脳内で Bloch 球上の動きを的確に追跡できるようになった。
