---
title: "行列単位のテンソル積分解"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Qiskit", "ポエム"]
published: false
---

# 目的

Qiskit textbook の [Proving Universality](https://qiskit.org/textbook/ch-gates/proving-universality.html) を読んでいた時に数学的に非自明な内容があったので、ニールセン&チャンとかで調べるのも面倒くさいし[^1]、線形代数のおさらいなので自分で証明することにした。

[^1]: 仮に書いてあっても丁寧な証明など期待できない。

# 用語や記法

## 行列単位 (Matrix unit)

Wikipedia の[行列単位](https://ja.wikipedia.org/wiki/%E8%A1%8C%E5%88%97%E5%8D%98%E4%BD%8D) を少し書き換えたものとして、この文書のスコープ内で $(a,b)$-成分のみ 1 で他は 0 であるような $n$ 次の行列単位 $E_n^{ab}$ を考える。

$$
\begin{align*}
E_n^{ab} = \begin{pmatrix}
0 & \cdots & 0 & \cdots & 0 \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
0 & \cdots & 1 & \cdots & 0 \\
\vdots & \ddots & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & \cdots & 0
\end{pmatrix}
\end{align*}
$$

## テンソル積

行列 $A$ と $B$ のテンソル積はニールセン&チャンのような量子計算の教科書の通りに以下のようなものとする。

$$
\begin{align*}
\begin{pmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{m1} & \cdots & a_{mn}
\end{pmatrix} \otimes B = \begin{pmatrix}
a_{11}B & \cdots & a_{1n}B \\
\vdots & \ddots & \vdots \\
a_{m1}B & \cdots & a_{mn}B
\end{pmatrix}
\end{align*}
$$

$\ket{0} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $\ket{1} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$ とし、$i_1\cdots i_n \in \{0,1\}$ に対し、$\ket{i_1 \cdots i_n} = \ket{i_1} \otimes \cdots \otimes \ket{i_n}$ とする。

### 記号 $d(i_1,\cdots,i_n)$

$i_1\cdots i_n$ という 2 進数がある時に、

$$
\begin{align*}
d(i_1,\cdots,i_n) = \sum_{k=0}^{n-1} 2^k i_{n-k}
\end{align*}
$$

で対応する 10 進数を定めたい。要するに、0b1010 に対しては $d(1,0,1,0) = 10$ という解釈である。

# 本記事の主張

$i_1,\cdots,i_n,j_1,\cdots,j_n \in \{0,1\}$ の時、

$$
\begin{align*}
\ket{i_1}\bra{j_1} \otimes \cdots \otimes \ket{i_n}\bra{j_n} = \ket{i_1\cdots i_n}\bra{j_1\cdots j_n}
\tag{1}
\end{align*}
$$

を示したい。

数学的な言い回しをすると（つまりケットブラで作れる）“行列単位” は $\ket{0}\bra{0}$, $\ket{0}\bra{1}$, $\ket{1}\bra{0}$, $\ket{1}\bra{1}$ のテンソル積に分解できることを見たい。
後述の補題 2 より任意の行列単位は一意に “主張の右辺” の形に書けるので、小さな 2x2 の行列単位のテンソル積に分解できることになる。

Qiskit textbook の表現で書くと

> any matrix can be expressed in terms of tensor products of Pauli matrices

である。[^2]

[^2]: この式変形まで含めると狙いがぼやけるので、今回は数学的な事実の証明にとどめる。

# 補題 1

行列単位同士のテンソル積は

$$
\begin{align*}
E_n^{ab} \otimes E_m^{cd} = E_{nm}^{(a-1)m+c,(b-1)m+d}
\end{align*}
$$

となる。証明はテンソル積の定義に基づく直接計算による。

# 補題 2

直接計算で $i,j \in \{0,1\}$ に対して $\ket{i}\bra{j} = E_2^{i+1,j+1} = E_2^{d(i)+1,d(j)+1}$ がわかる。

また、$\ket{i_1\cdots i_n} = (0 \cdots 1 \cdots 0)^T$ のように縦ベクトル表示をした時に 1 になる箇所が $d(i_1,\cdots,i_n)+1$ 番目であることに注意すれば、直接計算で一般に以下も分かる。

$$
\begin{align*}
\ket{i_1\cdots i_n}\bra{j_1\cdots j_n} = E_{2^n}^{d(i_1,\cdots,i_n)+1,d(j_1,\cdots,j_n)+1}
\end{align*}
$$

# 主張の証明

直接計算で、$i,j,k,\ell \in \{0,1\}$
に対して以下が示せる。

$$
\begin{align*}
\ket{i}\bra{j} \otimes \ket{k}\bra{\ell} &= E_2^{i+1,j+1} \otimes E_2^{k+1,\ell+1} \\
&= E_4^{2i+k+1,2j+\ell+1} \\
& = E_4^{d(i,k)+1,d(j,\ell)+1} \\
& = \ket{ik}\bra{j\ell}
\tag{1}
\end{align*}
$$

これを拡張して主張の式

$$
\begin{align*}
\ket{i_1}\bra{j_1} \otimes \cdots \otimes \ket{i_n}\bra{j_n} = \ket{i_1\cdots i_n}\bra{j_1\cdots j_n}
\end{align*}
\tag{2}
$$

を示したい。(2) が $n$ で成立しているとして、$n+1$ の時を考える:

$$
\begin{align*}
&\ \ket{i_1}\bra{j_1} \otimes \cdots \otimes \ket{i_n}\bra{j_n} \otimes \ket{i_{n+1}}\bra{j_{n+1}} \\
=&\ \ket{i_1\cdots i_n}\bra{j_1\cdots j_n} \otimes \ket{i_{n+1}}\bra{j_{n+1}} \\
\stackrel{\text{補題 2}}{=}&\ E_{2^n}^{d(i_1,\cdots,i_n)+1,d(j_1,\cdots,j_n)+1} \otimes E_2^{d(i_{n+1})+1,d(j_{n+1})+1} \\
\stackrel{\text{補題 1}}{=}&\ E_{2^n\cdot 2}^{2d(i_1,\cdots,i_n)+d(i_{n+1}),2d(j_1,\cdots,j_n)+d(j_{n+1})} \\
=&\ E_{2^{n+1}}^{d(i_1,\cdots,i_{n+1}),d(j_1,\cdots,j_{n+1})} \stackrel{\text{補題 2}}{=} \ket{i_1\cdots i_{n+1}}\bra{j_1\cdots j_{n+1}}
\tag{3}
\end{align*}
$$

故に、(1) と (3) より帰納法によって (2) が一般の $n$ に対して成立することが分かった。

# まとめ

計算しながら思いつきで導入した記号を使って証明したのでごちゃごちゃしたが、もう少しうまい定理とか使えばもっとコンパクトに証明できるかもしれない。
計算はごちゃごちゃしているが、$n=3$ まで手計算すれば本質が見えるので、消しゴムで 3 を消して $n$ に置き換えて、“$\cdots$” を加えたら証明の大部分は完了する。
