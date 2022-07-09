---
title: "QAOA を眺めてみる (1)"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: true
---

# 目的

QAOA (Quantum Approximate Optimization Algorithm) について手頃な教科書が見当たらなかったので、[Qiskit textbook](https://learn.qiskit.org/course/ch-applications/solving-combinatorial-optimization-problems-using-qaoa) を眺めることにした。すると、あまり自明とは言えない数式が出てきたので、そもそも読み違えていないかを確認する上でも証明しておこうというのが今回のスコープ。[^1]

[^1]: 但し、内容的には示唆に富むとか教育的であるといったことは一切なく、なんとか頑張って計算して同一性を示しただけである。

# QAOA とは？

ある与えられたコスト関数を最小化するような組み合わせを見つける問題:

$$
\begin{align*}
\argmax
\end{align*}_{x \in S} C(x)
$$

に対する量子近似最適化アルゴリズムということになるであろう。

基本的には textbook を順に読んでいけば良いが、難しい式があったので汚い証明をしておく。

# 記号

- $n$: 与えられた自然数。
- $[n] = \{1, \cdots, n\}$
- $Q \subset [n]$ とは例えば $Q = \varnothing$ や $Q = \{1\}$ や $Q = \{1, 2, 3, 5, 8, \cdots\}$ など。
- 整数 $0 \leq x \leq 2^{n-1}$ を 2 進展開 $x_1 x_2 \cdots x_n$ と同一視。
    - $\ket{x} = \ket{x_1 x_2 \cdots x_n} = \ket{x_1} \otimes \ket{x_2} \otimes \cdots \otimes \ket{x_n}$

# 問題の数式（命題）

バイナリ組み合わせ最適化問題におけるコスト関数を $C$ とし、$x \in \{0, 1\}^n$ に対し、これを

$$
\begin{align*}
\def\barQ{\overline{Q}}
C(x) = \sum_{(Q,\barQ) \subset [n]} w_{Q,\barQ} \prod_{i \in Q} x_i \prod_{j \in \barQ} (1-x_j)
\tag{1}
\end{align*}
$$

とおく。次に “ハミルトニアン” を

$$
\begin{align*}
H = \sum_{x \in \{0, 1\}^n} C(x) \ket{x} \bra{x}
\tag{2}
\end{align*}
$$

とおく。この時、$H$ は (1) 式で $x_i \to \frac{1-Z_i}{2}$ と置き換えた

$$
\begin{align*}
\def\barQ{\overline{Q}}
\sum_{(Q,\barQ) \subset [n]} w_{Q,\barQ} \frac{1}{2^{|Q| + |\barQ|}} \prod_{i \in Q} (1-Z_i) \prod_{j \in \barQ} (1+Z_j)
\tag{3}
\end{align*}
$$

に等しくなる。という主張が書かれている。ここで、$Z_i$ は $I^{\otimes n}$ の $i$ 番目が $Z$ になったテンソル積 $I \otimes \cdots \otimes Z \otimes \cdots \otimes I$ である。

ところで、

$$
\begin{align*}
\frac{1-Z_i}{2} &= I \otimes \cdots \otimes \frac{1-Z}{2} \otimes \cdots \otimes I = I \otimes \cdots \otimes \ket{1}\bra{1} \otimes \cdots \otimes I \\
\frac{1-Z_i}{2} &= I \otimes \cdots \otimes \frac{1+Z}{2} \otimes \cdots \otimes I = I \otimes \cdots \otimes \ket{0}\bra{0} \otimes \cdots \otimes I
\end{align*}
$$

に注意すると、(3) 式は

$$
\begin{align*}
\def\barQ{\overline{Q}}
\sum_{(Q,\barQ) \subset [n]} w_{Q,\barQ} \prod_{i \in Q} I \otimes \cdots \otimes \overbrace{\ket{1}\bra{1}}^i \otimes \cdots \otimes I \prod_{j \in \barQ} I \otimes \cdots \otimes \overbrace{\ket{0}\bra{0}}^j \otimes \cdots \otimes I
\tag{3'}
\end{align*}
$$

とも書けることに注意したい。

# 示すこと

(2) 式と (3) 式が等しいこと。

# 証明

## 問題の書き換え

(1) を (2) に代入して整理すると

$$
\begin{align*}
\def\barQ{\overline{Q}}
H = \sum_{(Q,\barQ) \subset [n]} w_{Q,\barQ} \left[ \sum_{x \in \{0, 1\}^n} \prod_{i \in Q} x_i \prod_{j \in \barQ} (1-x_j) \ket{x}\bra{x} \right]
\tag{4}
\end{align*}
$$

となる。よって (3') 式と比較すると、示すべきことは任意の $(Q,\overline{Q}) \subset [n]$ に対する以下の (5) と (6) の同値性である。[^2]

[^2]: どちらも対角行列なので、対角成分の一致を見れば良い。

$$
\begin{align*}
\def\barQ{\overline{Q}}
\prod_{i \in Q} I \otimes \cdots \otimes \overbrace{\ket{1}\bra{1}}^i \otimes \cdots \otimes I \prod_{j \in \barQ} I \otimes \cdots \otimes \overbrace{\ket{0}\bra{0}}^j \otimes \cdots \otimes I
\tag{5}
\end{align*}
$$

$$
\begin{align*}
\def\barQ{\overline{Q}}
\sum_{x \in \{0, 1\}^n} \prod_{i \in Q} x_i \prod_{j \in \barQ} (1-x_j) \ket{x}\bra{x}
\tag{6}
\end{align*}
$$

以下、$Q$ と $\overline{Q}$ は互いに素、つまり $Q \cap \overline{Q} = \varnothing$ とする。仮にそうでない場合、$k \in Q \cap \overline{Q}$ が存在して、(5) 式からは $I \otimes \cdots \otimes \overbrace{\ket{1}\bra{1} \ket{0}\bra{0}}^k \otimes \cdots \otimes I$ が、(6) 式からは $x_k (1-x_k)$ という項が見出せる。前者は自明に $0$ であり、後者はいかなる $x \in \{0, 1\}^n$ に対しても $0$ である。

## (6) 式を見る ― 対角成分に注目する
(6) をよく見ると、$x \in \{0, 1\}^n$ に対して $\ket{x}\bra{x}$ は違いに直交する射影演算子になっており、$\sum_{x \in \{0, 1\}^n} \ket{x}\bra{x} = I$ であるのでスペクトル分解の形になっている。

$y \in \{0, 1\}^n$ を任意にとって固定する。この時、“固有ベクトル” $\ket{y}$ に対する固有値は

$$
\begin{align*}
\def\barQ{\overline{Q}}
\prod_{i \in Q} x_i \prod_{j \in \barQ} (1-x_j) \bigg|_{x=y}
\tag{7}
\end{align*}
$$

である。よって、(5) も (6) も共に対角行列であることに注意し、(5) 式より、

$$
\begin{align*}
\def\barQ{\overline{Q}}
\prod_{i \in Q} I \otimes \cdots \otimes \overbrace{\ket{1}\bra{1}}^i \otimes \cdots \otimes I \prod_{j \in \barQ} I \otimes \cdots \otimes \overbrace{\ket{0}\bra{0}}^j \otimes \cdots \otimes I \ket{y}
\tag{8}
\end{align*}
$$

が (7) に等しいことを見れば良い。

## (8) 式を見る

$\ket{y} = \ket{y_1} \otimes \cdots \otimes \ket{y_n}$ であるが、$Y_0 := \{i;\ \ket{y_i} = \ket{0} \}$, $Y_1 := \{j;\ \ket{y_j} = \ket{1} \}$ と置くと、$[n] = Y_0 \sqcup Y_1$ (disjoint union) である。

(7) 式をよくみると、

$$
\begin{align*}
= \begin{cases}
1,\quad Q \subset Y_1 \;\;\;\text{かつ} \;\;\; \overline{Q} \subset Y_0 \\
0,\quad \text{それ以外}
\end{cases}
\tag{9}
\end{align*}
$$

であることが分かる。

(8) 式を少し丁寧に見ると、$y_k = 0$ の時、$k \in Q$ なら $0$、$k \not\in Q$ なら $1$ が分かる。逆に、$y_k = 1$ の時、$k \in \overline{Q}$ なら $0$、$k \not\in \overline{Q}$ なら $1$ が分かる。言い換えると、

$$
\begin{align*}
= \begin{cases}
1,\quad Q \cap Y_0 = \varnothing \;\;\;\text{かつ} \;\;\; \overline{Q} \cap Y_1 = \varnothing \\
0,\quad \text{それ以外}
\end{cases}
\tag{10}
\end{align*}
$$

である。これは容易に分かるが (9) に等しい。同じ条件で $0$ or $1$ が決まるので、(7) 式と (8) 式の値は等しいことになる。

$\{\ket{y}\}_{y \in \{0, 1\}^n}$ は $\mathbb{C}^{2^n}$ を張るので、これらの線形結合を考えることで (5) 式と (6) 式が等しいことが示された。以上より命題の主張は示された。

# まとめ

実は最初からこの式変形で示したわけではなくて、$n=1$ と $n=2$ で $(Q,\overline{Q}) \subset [n]$ のすべてのケースについて成立することを手計算するところから始めた。

長い間うまい式変形が思いつかず苦戦したが、同じく Qiskit textbook の [Proving Universality](https://learn.qiskit.org/course/ch-gates/proving-universality) を見ているうちに「テンソル積を計算した後の巨大な行列の性質ではなく、積表示の個々の単一量子ゲートの各量子ビットへの作用にもっと注目すべきでは・・・」と思ったのが決め手になった。
