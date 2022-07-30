---
title: "Qiskit で遊んでみる (5) — QGSS2022 より"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: true
---

# 目的

[Qiskit Global Summer School 2022: Quantum Simulations](https://qiskit.org/events/summer-school/) に参加した。その中で興味を持った題材について忘れないうちにまとめてみたい。

具体的には [Quantum computers as universal quantum simulators: state-of-art and perspectives](https://arxiv.org/abs/1907.03505) を参考にする形でハミルトニアン $H = X \otimes X$ の時間発展についてまとめる。

# おさらい

次回の記事の伏線の形で、ハミルトニアン $H = X$ の時間発展について考えてみる。これは $U(t) = \exp(-itH)$ であるが、$R_X(\theta) = \exp(-i \frac{\theta}{2} X)$ を思い出すと、$U(t) = R_X(2t)$ であることが分かる。つまり、$X$ の時間発展は $X$ 軸の周りの回転ゲートである。

# では $X \otimes X$ の時間発展は？

そんなものが量子回路で実装できるのか？と思ってしまう部分もあったのだが、実は実装できてしまう。結論としては以下の回路である:s

```python
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter

t = Parameter('t')
qc = QuantumCircuit(2)

qc.ry(np.pi/2,[0,1])
qc.cx(0,1)
qc.rz(2 * t, 1)
qc.cx(0,1)
qc.ry(-np.pi/2,[0,1])

qc.draw()
```

![](/images/dwd-qiskit05/001.png)

ちょっとこの実装は面倒臭いなという場合には、以下のようにしても良さそうである。

```python
from qiskit.opflow import X

H = X^X
U = (t * H).exp_i()
```

さて、この両者は同じなのだろうか？簡単な実験をしてみよう

## ハミルトニアンの時間発展を比較する

関数名はいまいちかもしれないが、以下のように `opflow` 版と `QuantumCircuit` 版を用意する。

```python
def XX_opflow(t):
    theta = Parameter('theta')
    H = X^X
    U = (theta * H_XX).exp_i()
    U = U.assign_parameters({theta: t})
    return U.to_matrix()

def XX_circuit(t):
    theta = Parameter('theta')
    qc = QuantumCircuit(2)

    qc.ry(np.pi/2,[0,1])
    qc.cx(1,0)
    qc.rz(2 * theta, 0)
    qc.cx(1,0)
    qc.ry(-np.pi/2,[0,1])

    qc = qc.assign_parameters({theta: t})

    sim = Aer.get_backend('aer_simulator')
    qc.save_unitary()
    result = sim.run(qc).result()
    return result.get_unitary()
```

そして、$0.01\pi$ 刻みで $0$ から $\pi$ までの時刻の時間発展を比較すると

```python
times = np.arange(0, np.pi, .01*np.pi)
result = [np.allclose(XX_opflow(t), XX_unitary(t)) for t in times]
print(np.all(result))
```

> True

となる。離散値での確認ではあるが、同じであると考えて良いだろう。

# 実際に回路のユニタリ行列を計算する

折角なので数式でも比較してみよう。以下、直接計算であり、あまり美しくはない。

$$
\begin{align*}
X \otimes X = \begin{bmatrix}
O & X \\
X & O
\end{bmatrix}
\end{align*}
$$

であり、$(X \otimes X)^2 = I$ であることから、

$$
\begin{align*}
\exp(-i t X \otimes X) &= \cos t I - i \sin t (X \otimes X) \\
&= \begin{bmatrix}
\cos t I & -i \sin t X \\
-i \sin t X & \cos t I
\end{bmatrix}
\tag{1}
\end{align*}
$$

が分かる。冒頭の量子回路のユニタリ行列がこれに一致することを確認する。

$$
\begin{align*}
R_Y(\pm \frac{\pi}{2}) = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & \mp 1 \\
\pm 1 & 1
\end{bmatrix}
\end{align*}
$$

であるので、

$$
\begin{align*}
R_Y(\frac{\pi}{2}) \otimes R_Y(\frac{\pi}{2}) &= \frac{1}{\sqrt{2}} \begin{bmatrix}
R_Y(\frac{\pi}{2}) & - R_Y(\frac{\pi}{2}) \\
R_Y(\frac{\pi}{2}) & R_Y(\frac{\pi}{2})
\end{bmatrix} \\
R_Y(- \frac{\pi}{2}) \otimes R_Y(- \frac{\pi}{2}) &= \frac{1}{\sqrt{2}} \begin{bmatrix}
R_Y(- \frac{\pi}{2}) & R_Y(- \frac{\pi}{2}) \\
- R_Y(- \frac{\pi}{2}) & R_Y(- \frac{\pi}{2})
\end{bmatrix}
\end{align*}
$$

である。また、

$$
\begin{align*}
CX &= \begin{bmatrix}
I & O \\
O & X
\end{bmatrix} \\
I \otimes R_Z(2t) &= e^{-i t} \begin{bmatrix}
P(2t) & O \\
O & P(2t)
\end{bmatrix}
\end{align*}
$$

である。

ここから計算の見てくれが煩雑になるので、略記として $P = P(2t)$, $R = R_Y(\frac{\pi}{2})$, $R^\dagger = R_Y(-\frac{\pi}{2})$ を使うことにする。すると回路のユニタリ行列は、

$$
\begin{align*}
&\ \left(R_Y(- \frac{\pi}{2}) \otimes R_Y(- \frac{\pi}{2})\right) CX \left(I \otimes R_Z(2t)\right) CX \left(R_Y(\frac{\pi}{2}) \otimes R_Y(\frac{\pi}{2})\right)  \\
=&\ \frac{e^{-i t}}{2} \begin{bmatrix}
R^\dagger & R^\dagger \\
- R^\dagger & R^\dagger
\end{bmatrix}
\begin{bmatrix}
I & O \\
O & X
\end{bmatrix}
\begin{bmatrix}
P & O \\
O & P
\end{bmatrix}
\begin{bmatrix}
I & O \\
O & X
\end{bmatrix}
\begin{bmatrix}
R & - R \\
R & R
\end{bmatrix} \\
= &\ \frac{e^{-i t}}{2} \begin{bmatrix}
R^\dagger (P + XPX) R & - R^\dagger (P - XPX) R \\
- R^\dagger (P - XPX) R & R^\dagger (P + XPX) R
\end{bmatrix}
\tag{2}
\end{align*}
$$

となる。簡単な計算で $P + XPX = (1 + e^{-2it})I$, $P - XPX = (1 - e^{-2it})Z$ が分かるので、$R^\dagger Z R = -X$ に注意すると、(2) は

$$
\begin{align*}
&\ \frac{e^{-i t}}{2} \begin{bmatrix}
R^\dagger (P + XPX) R & - R^\dagger (P - XPX) R \\
- R^\dagger (P - XPX) R & R^\dagger (P + XPX) R
\end{bmatrix} \\
=&\ \frac{e^{-i t}}{2}
\begin{bmatrix}
R^\dagger (1 + e^{2it}) R & - R^\dagger (1 - e^{2it})Z R \\
- R^\dagger (1 - e^{2it})Z R & R^\dagger (1 + e^{2it}) R
\end{bmatrix} \\
=&\ \begin{bmatrix}
\frac{e^{it} + e^{-it}}{2} I & - \frac{e^{it} - e^{-it}}{2} X \\
- \frac{e^{it} - e^{-it}}{2} X & \frac{e^{it} + e^{-it}}{2} I
\end{bmatrix} = \begin{bmatrix}
\cos t I & -i \sin t X \\
-i \sin t X & \cos t I
\end{bmatrix}
\end{align*}
$$

となり、(1) と一致することが示された。

# まとめ

ハミルトニアン $X \otimes X$ の時間発展を記述する量子回路について確認ができた。リンクした論文の pp.6-7 を見ることで、同様に $Y \otimes Y$ および $Z \otimes Z$ の時間発展も量子回路で実装できることが分かる。
