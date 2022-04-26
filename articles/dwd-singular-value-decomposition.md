---
title: "特異値分解のちょっと格好いい姿を眺めてみる"
emoji: "📈"
type: "tech"
topics: ["math", "ポエム"]
published: false
---

# 目的

統計学や機械学習において特異値分解というものに遭遇することがある。これについて正方行列の時には関数解析の方面で「コンパクト作用素の canonical form」と呼ばれているものと実質同じものになっているので、これを少し見てみたい。

# 特異値分解

$A \in \mathrm{Mat}(m,n; \mathbb{C})$ に対して、適当なユニタリ行列 $U \in U(m)$ と $V \in U(n)$ および、非負の数を対角成分に持つ対角行列 $\Sigma \in \mathrm{Mat}(m,n;\R)$がとれて

$$
\begin{align*}
A = U \Sigma V^{*}
\tag{1}
\end{align*}
$$

が成立するというものである。ここで $V^*$ は $V$ の共役行列である。
これについて、簡単のため、$m=n=2$ の場合を見てみたい。

## ざっくり説明

説明については、手持ちの本に手頃なものがないので文献 [RS1] に従う。

$A^* A$ はエルミート行列になるので、スペクトル分解より適当な非負の数 $s_1^2$ と $s_2^2$ とそれぞれの固有空間への射影行列 $P_1$ と $P_2$ を用いて

$$
\begin{align*}
A^* A = s_1^2 P_1 + s_2^2 P_2
\end{align*}
$$

と書ける。ここで、簡単のため、$s_1^2 \neq s_2^2 > 0$ の場合のみを考える。
この場合、一般論よりそれぞれの固有空間は直交するので、単位ベクトル $\psi_1 \in P_1 \mathbb{C}^2$ と $\psi_2 \in P_2 \mathbb{C}^2$ をとるとこれらは直交する。よって $V = (\psi_1\quad \psi_2) \in \mathrm{Mat}(2,2;\mathbb{C})$ とおくと、これはユニタリ行列になる。

次に、$\phi_i = \frac{1}{s_i} A \psi_i,\ i=1,2$ と置いて、$U = (\phi_1\quad \phi_2) \in \mathrm{Mat}(2,2;\mathbb{C})$ と置くと、直接計算により $U$ もユニタリ行列になることがわかる。

ここまでで準備した $U$ と $V$ を使って以下の計算をすると、

$$
\begin{align*}
U \begin{pmatrix}
s_1 & 0 \\
0 & s_2
\end{pmatrix} V^* = \left(\frac{1}{s_1} A \psi_1\quad \frac{1}{s_2} A \psi_2\right) \begin{pmatrix}
s_1 \bar{\psi}_1^T \\
s_2 \bar{\psi}_2^T
\end{pmatrix} = A
\end{align*}
$$

となる。最後の等号はそれほど自明ではないが、任意の $\psi \in \mathbb{C}^2$ をとる時、

$$
\begin{align*}
\left(\frac{1}{s_1} A \psi_1\quad \frac{1}{s_2} A \psi_2\right) \begin{pmatrix}
s_1 \bar{\psi}_1^T \\
s_2 \bar{\psi}_2^T
\end{pmatrix} \psi = A((\bar{\psi}_1^T \psi_1)\psi_1 + (\bar{\psi}_2^T \psi_2)\psi_2) = A \psi
\tag{2}
\end{align*}
$$

と計算されることから分かる。ここで、$\psi = (\bar{\psi}_1^T \psi_1)\psi_1 + (\bar{\psi}_2^T \psi_2)\psi_2$ という $\psi$ の正規直交基底による展開を用いた。

最後に、$\Sigma = \begin{pmatrix} s_1 & 0 \\ 0 & s_2 \end{pmatrix}$ と置くことで、(1) を得る。

これを文献 [RS1] で言う “コンパクト作用素の canonical form” の形で書くと

$$
\begin{align*}
A \psi = s_1 \braket{\psi_1,\psi} \phi_1 + s_2 \braket{\psi_2,\psi} \phi_2
\end{align*}
$$

というちょっと格好いい級数展開の形式になる。ここで、$\braket{\cdot,\cdot}$ は $\mathbb{C}^2$ のエルミート内積である。この canonical form の右辺は (2) の左辺を $\phi_i$ の定義を思い出しつつ変形すれば得られる。

# 具体例

できるだけ簡単な計算で眺めたいので、2 乗して単位行列になるような Pauli 行列を改造してサンプルを作る[^1]。

[^1]: 量子ゲートでいう $Y$ ゲートを使う。

$$
\begin{align*}
A = \begin{pmatrix}
0 & -\frac{i}{3} \\
\frac{i}{2} & 0
\end{pmatrix}
\end{align*}
$$

を考える。直接計算して $A^* A = A = \begin{pmatrix} 1/4 & 0 \\ 0 & 1/9 \end{pmatrix}$ を得る。この時、上記で言うような $\psi_1, \psi_2$ はそれぞれ $\begin{pmatrix} -1 \\ 0 \end{pmatrix}, \begin{pmatrix} 0 \\ -1 \end{pmatrix}$ にとれる[^2]。後は、$\phi_1$ と $\phi_2$ をそれぞれ計算すると、以下が $A$ の特異値分解であることが分かる。

[^2]: 後の都合でここでは符号をマイナスにとっている。

$$
\begin{align*}
A = \begin{pmatrix}
0 & i \\
-i & 0
\end{pmatrix}\begin{pmatrix}
\frac{1}{2} & 0 \\
0 & \frac{1}{3}
\end{pmatrix}\begin{pmatrix}
-1 & 0 \\
0 & -1
\end{pmatrix}
\end{align*}
$$

# NumPy で検証してみる

以下のような実装で、理論上の計算と同じ結果が得られていることが分かる。

```python
>>> import numpy as np
>>> A = np.array([[0, -(1/3)*1.j], [(1/2)*1.j, 0]])
>>> u, s, vh = np.linalg.svd(A)
>>> u
array([[0.+0.j, 0.+1.j],
       [0.-1.j, 0.+0.j]])
>>> s
array([0.5       , 0.33333333])
>>> vh
array([[-1.+0.j, -0.+0.j],
       [-0.+0.j, -1.+0.j]])
```

# まとめ

特異値分解は普段はライブラリで計算してしまうので、それほど気に留める存在でもないかもしれない。ところが、関数解析の目線で見てあげると意外とイケメンな素顔もあるので、そこに焦点を当ててみようかと思ってざっと記事を書いてみた。

## 余談

Wikipedia の [Singular value](https://en.wikipedia.org/wiki/Singular_value) を紐解くと

> In mathematics, in particular functional analysis, the singular values, or s-numbers of a compact operator T : X → Y acting between Hilbert spaces X and Y, are the square roots of non-negative eigenvalues of the self-adjoint operator T\*T (where T\* denotes the adjoint of T).

といったように、特異値は分野によっては `s-number` と呼ばれる。手元に適当な本がないのでうろ覚えだが、Schatten クラスと呼ばれるコンパクト作用素のイデアルがこの s-number を用いて定義され、ルベーグ積分における $L^p$ クラスのような位置付けになっていたような記憶がある。

# 文献

[RS1] M. Reed and B. Simon. Methods of Modern Mathematical Physics, I. Functional Analysis, pp.203-204, Academic Press, New York, 1981
