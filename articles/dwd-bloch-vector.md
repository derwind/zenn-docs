---
title: "Bloch ベクトルを眺めてみる"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: false
---

# 目的

Bloch 球の上に状態ベクトルをマッピングした Bloch ベクトルという概念がある。よく見かけるのは計算基底 $\ket{0}$, $\ket{1}$ や X 基底 $\ket{+}$, $\ket{-}$ をマッピングしたものだが、回転ゲートを適用した場合に、Bloch ベクトルとしてはどのような影響を受けるのかを見てみたい。
回転ゲート自体は Bloch ベクトルではなく、対応する状態ベクトル $\in \mathbb{C}^2\!/\!\!\sim\ \simeq \mathbb{C}P^1$ に対する演算なので、結果はそれほど自明ではない。

# Bloch ベクトル

$$
\begin{align*}
I = \begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix},\ X = \begin{bmatrix}
0. & 1 \\
1 & 0
\end{bmatrix},\ Y = \begin{bmatrix}
0 & -i \\
i & 0
\end{bmatrix},\ Z = \begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
\end{align*}
$$

とする。また、$\vec{\sigma} = (X\ Y\ Z)^T \in \mathrm{Mat}(2, \mathbb{C})^3$ と置く。

状態ベクトル $\ket{\psi} \in \mathbb{C} P^1$ を Bloch 球上のベクトル $\vec{r} = (x\ y\ z)^T \in [-1, 1]^3$ に対応させることができ、この $\vec{r}$ を「Bloch ベクトル」と呼ぶ。

より具体的には、純粋状態 $\ket{\psi}$ の密度行列を

$$
\begin{align*}
\rho = \ket{\psi} \bra{\psi}
\tag{1}
\end{align*}
$$
とする。この時、

$$
\begin{align*}
x &= \braket{X} = \operatorname{tr}(X \rho), \\
y &= \braket{Y} = \operatorname{tr}(Y \rho), \\
z &= \braket{Z} = \operatorname{tr}(Z \rho)
\tag{2}
\end{align*}
$$

とすると、$\rho = \frac{1}{2}(I + \vec{r}\cdot\vec{\sigma}) = \frac{1}{2}(I + xX + yY + zZ)$ という関係を満たす。詳細は文献 [NC1] 演習 2.72 や [NC3] 8.4.2 量子プロセストモグラフィーに譲る。

## Bloch ベクトルのサンプル

以下に、状態ベクトル $\ket{0}$ と、X 軸に関して $\frac{\pi}{4}$ 回転させたものを Bloch 球上に Bloch ベクトルとして描画する。

|$\ket{0}$|$R_X(\frac{\pi}{4}) \ket{0}$|
|:--:|:--:|
|![](/images/dwd-bloch-vector/001.png =300x)|![](/images/dwd-bloch-vector/002.png =300x)|

雰囲気としては、Bloch 球を普通のユークリッド空間 $\R^3$ の中で見ることにして、YZ 平面内で $\frac{\pi}{4}$ 回転させたような位置に来ているのでは？と予想される。実際それはそうなっている。ところが「一旦 $R_Y$ ゲートで YZ 平面から Bloch ベクトルを浮かせた状態で $R_X(\frac{\pi}{4})$ を適用した場合はどうなるのだろうか？」といったことは自明ではない。

# 実際に計算で確認する

今回は、$R_X(\lambda)$ の適用だけを確認する。[^1]

[^1]: 計算した結果としてかなりつらかったので、他のゲートの確認をする気力がない。

純粋状態、つまり Bloch 球上に対応する状態ベクトルは、グローバル位相を除いて $\ket{\psi} = \cos \frac{\theta}{2} \ket{0} + e^{i \phi} \sin \frac{\theta}{2} \ket{1}$ と書けるのであった。これに対応する Bloch ベクトルは (1) 式と (2) 式より、

$$
\begin{align*}
x &= \sin \theta \cos \phi \\
y &= \sin \theta \sin \phi \\
z &= \cos \theta
\tag{3}
\end{align*}
$$

である。要するに、よく見かける $\R^3$ の単位球上の点の極座標表示である。

また、$R_X(\lambda)$ ゲートは、

$$
\begin{align*}
R_X(\lambda) = \cos \frac{\lambda}{2} I -i \sin \frac{\lambda}{2} X = \begin{bmatrix}
\cos \frac{\lambda}{2} & -i \sin \frac{\lambda}{2} \\
-i \sin \frac{\lambda}{2} & \cos \frac{\lambda}{2}
\end{bmatrix}
\end{align*}
$$

と書けた。よって、任意の純粋状態の状態ベクトルを X 軸に沿って回転した結果は

$$
\begin{align*}
\ket{\psi^\prime} = R_X(\lambda) \ket{\psi} = \begin{bmatrix}
\cos \frac{\lambda}{2} \cos \frac{\theta}{2} -i e^{i \phi} \sin \frac{\lambda}{2} \sin \frac{\theta}{2} \\
-i \sin \frac{\lambda}{2} \cos \frac{\theta}{2} +  e^{i \phi} \cos \frac{\lambda}{2} \sin \frac{\theta}{2}
\end{bmatrix} = \begin{bmatrix} a \\ b \end{bmatrix}
\end{align*}
$$

となるが、これを Bloch ベクトル $\vec{r^\prime} = (x^\prime\ y^\prime\ z^\prime)^T$ として表現したい。このために使える道具は再び (1) 式と (2) 式である。

## $x^\prime$ の計算

$x^\prime = \operatorname{tr}(X \ket{\psi^\prime} \bra{\psi^\prime}) = \operatorname{tr}(X \begin{bmatrix} a a^* & a b^* \\ b a^* & b b^* \end{bmatrix}) = b a^* + a b^* = 2 \operatorname{Re}(a b^*)$ を計算する。

$$
\begin{align*}
x^\prime &= 2 \operatorname{Re} \left( (\cos \frac{\lambda}{2} \cos \frac{\theta}{2} -i e^{i \phi} \sin \frac{\lambda}{2} \sin \frac{\theta}{2}) (i \sin \frac{\lambda}{2} \cos \frac{\theta}{2} + e^{-i \phi} \cos \frac{\lambda}{2} \sin \frac{\theta}{2}) \right) \\
&= 2 (\cos \frac{\lambda}{2} \cos \frac{\theta}{2} \cos \phi \cos \frac{\lambda}{2} \sin \frac{\theta}{2} + \cos \phi \sin \frac{\lambda}{2} \sin \frac{\theta}{2} \sin \frac{\lambda}{2} \cos \frac{\theta}{2}) \\
&= 2 \cos \phi \sin \frac{\theta}{2} \cos \frac{\lambda}{2} = \sin \theta \cos \phi
\end{align*}
$$

となる。(3) 式と比較すると、$x^\prime = x$ であることが分かる。つまり、X 方向への移動は発生しないことになる。

## $y^\prime$ と $z^\prime$ の計算

この 2 つの計算はまとめて行うほうが結果の考察がやりやすいのでセットで行う。[^2]

[^2]: 計算の詳細は地味で長いので大幅に割愛する。

$y^\prime = \operatorname{tr}(Y \ket{\psi^\prime} \bra{\psi^\prime}) = \operatorname{tr}(Y \begin{bmatrix} a a^* & a b^* \\ b a^* & b b^* \end{bmatrix}) = i (a b^* - (a b^*)^*) = - 2 \operatorname{Im}(a b^*)$ を計算する。

$$
\begin{align*}
y^\prime &= -2 \operatorname{Im} \left( (\cos \frac{\lambda}{2} \cos \frac{\theta}{2} -i e^{i \phi} \sin \frac{\lambda}{2} \sin \frac{\theta}{2}) (i \sin \frac{\lambda}{2} \cos \frac{\theta}{2} + e^{- i \phi} \cos \frac{\lambda}{2} \sin \frac{\theta}{2}) \right) \\
&= -2(\sin \frac{\lambda}{2} \cos \frac{\lambda}{2} \cos \theta - \sin \phi \cos \lambda \sin \frac{\theta}{2} \cos \frac{\theta}{2}) \\
&= \cos \lambda \sin \theta \sin \phi - \sin \lambda \cos \theta
\end{align*}
$$

$z^\prime = \operatorname{tr}(Z \ket{\psi^\prime} \bra{\psi^\prime}) = \operatorname{tr}(Z \begin{bmatrix} a a^* & a b^* \\ b a^* & b b^* \end{bmatrix}) = a a^* - b b^*$ を計算する。

$$
\begin{align*}
z^\prime &= \cos^2 \frac{\lambda}{2} \cos \theta + \sin \phi (2 \sin \frac{\lambda}{2} \cos \frac{\lambda}{2}) (2 \sin \frac{\theta}{2} \cos \frac{\theta}{2}) - \sin^2 \frac{\lambda}{2} \cos \theta \\
&= \sin \lambda \sin \theta \sin \phi + \cos \lambda \cos \theta
\end{align*}
$$

となる。よって、

$$
\begin{align*}
\begin{bmatrix}
y^\prime \\ z^\prime
\end{bmatrix} = \begin{bmatrix}
\cos \lambda & - \sin \lambda \\
\sin \lambda & \cos \lambda
\end{bmatrix} \begin{bmatrix}
\sin \theta \sin \phi \\
\cos \theta
\end{bmatrix}
\end{align*}
$$

となっている。つまり YZ 平面内で角度 $\lambda$ の回転が適用された形である。

# Python でもやってみる

やっつけで書いたものなので、ダメなケースもあるかもしれないが、ざっくり動かすことはできると思う。

```python
import numpy as np

I = np.array([
    [1., 0.],
    [0., 1.]
])
X = np.array([
    [0., 1.],
    [1., 0.]
])
Y = np.array([
    [0., -1.j],
    [1.j, 0.]
])
Z = np.array([
    [1., 0.],
    [0., -1.]
])
def Rx(theta):
    return np.cos(theta/2) * I - 1.j * np.sin(theta/2) * X
def Ry(theta):
    return np.cos(theta/2) * I - 1.j * np.sin(theta/2) * Y
def Rz(theta):
    return np.cos(theta/2) * I - 1.j * np.sin(theta/2) * Z

def density_matrix(state):
    if len(state.shape) == 1:
        state = state.reshape(-1, 1)
    return state * np.conjugate(state.T)

def state2bloch(state):
    rho = density_matrix(state)
    x, y, z = np.trace(X @ rho), np.trace(Y @ rho), np.trace(Z @ rho)
    return np.real(np.array([x, y, z]))

def bloch2state(vec, epsilon = 1e-10):
    x, y, z = np.real(vec)
    cos = np.sqrt((1 + z)/2)
    if z > 1 - epsilon: # theta = 0
        return np.array([1., 0.])
    elif z < -1 + epsilon: # theta = pi
        return np.array([0., 1.])
    else:
        sin = np.sqrt(x**2 + y**2) / (2*cos)
        if x < 0:
            sin = -sin
        if abs(x) < epsilon: # phi = pi/2, 3pi/2
            if y >= 0:
                phi = np.pi/2
            else:
                phi = 3 * np.pi/2
        else:
            phi = np.arctan(y/x)
        return np.array([cos, np.exp(1.j*phi)*sin])
```

## 実験

### 自作 APIs を使う

冒頭のほうで書いた「一旦 $R_Y$ ゲートで YZ 平面から Bloch ベクトルを浮かせた状態で $R_X(\frac{\pi}{4})$ を適用した場合はどうなるのだろうか？」を実験してみる。Y 軸に関して $\frac{\pi}{4}$ 回転させてから X 軸に関して $\frac{\pi}{4}$ 回転させる。

```python
init_state = np.array([1., 0.])
final_state = Rx(np.pi/4) @ Ry(np.pi/4) @ init_state
final_vec = state2bloch(final_state)
print(final_vec)
display(plot_bloch_vector(final_vec))
```

![](/images/dwd-bloch-vector/003.png =300x)

### NumPy のみで計算する

既に計算したように、回転演算はユークリッド空間の中の直感的な回転として実行しても良いので、以下のように書ける:

```python
def rot(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

init_vec = np.array([0., 0., 1.])

vec = init_vec[:]
vec[[2,0]] = rot(np.pi/4)@vec[[2,0]] # rotate only 2:z, 0:x around Y-axis
vec[[1,2]] = rot(np.pi/4)@vec[[1,2]] # rotate only 1:y, 2:z around X-axis
print(vec)
display(plot_bloch_vector(vec))
```

表示される絵は上記と同じなので割愛する。

## Qiskit での計算と比較

念のために、量子回路を実行して得た状態ベクトルから `plot_bloch_multivector` で結果を描画してみよう。

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

qc = QuantumCircuit(1)
qc.ry(np.pi/4, 0)
qc.rx(np.pi/4, 0)
state = Statevector.from_instruction(qc)
display(plot_bloch_multivector(state))
```

![](/images/dwd-bloch-vector/004.png)

可視化の API が違うので細部に少し差はあるが、見た感じは似ているのではないだろうか。

# まとめ

状態ベクトルに回転ゲートを適用するとなんとなく Bloch 球上を移動することは分かり、また、$\frac{\pi}{2}$ や $\pi$ の回転の時は、分かりやすい位置に移動するのでそれほど疑問もなかった[^3]。ところが、もっと中途半端な角度であるとか、回転の軸を複数組み合わせる時にどうなるかといったことについてはまったく自信がなかった。これについて今回は $R_X$ ゲートのみであるが、“直感的な意味での” 回転が Bloch 球上の Bloch ベクトルに対して作用することが分かった。

[^3]: というよりは気にしないでおくことができた。

本来は $R_Y$ および $R_Z$ についても計算すべきであるが、$R_X$ だけで大分大変だったのでまだ計算していない。`SymPy` あたりで計算の手を抜けないだろうかと考えている。

実験に使用したコード類は GitHub 上に [bloch_vector.ipynb](https://github.com/derwind/qiskit_applications/blob/2a7d28be9b68ec51bfd1a8557eaadf22fe69fd0f/bloch_vector/bloch_vector.ipynb) として置いている。

# 参考文献

[NC1] M. A. Nielsen, I. L. Chuang. [量子コンピュータと量子通信I](https://shop.ohmsha.co.jp/shop/shopdetail.html?brandcode=000000006441), オーム社, 2004
[NC3] M. A. Nielsen, I. L. Chuang. [量子コンピュータと量子通信III](https://shop.ohmsha.co.jp/shop/shopdetail.html?brandcode=000000006439), オーム社, 2004
