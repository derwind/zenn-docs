---
title: "QAOA を眺めてみる (2)"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: false
---

# 目的

[前回](/articles/dwd-qiskit-qaoa01)、Qiskit textbook の [Solving combinatorial optimization problems using QAOA](https://learn.qiskit.org/course/ch-applications/solving-combinatorial-optimization-problems-using-qaoa) の中の難しいなと感じる式を証明してみたが、その上で読み直してもすぐにはよく分からないので、適当に行間を埋めてみたい。

但し、textbook を読んでも原論文 [A Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028) を読んでも[^1]、何故問題のハミルトニアンとミキサーを交互に混ぜるのか？などは見えておらず、あくまで雰囲気だけの記事である。

[^1]: そもそもまだちゃんと読んでいないくて斜め読みなのだが・・・。

# 記号

- $x = (x_1,x_2,x_3,x_4) \in \{0, 1\}^4$, $\ket{x} = \ket{x_1 x_2 x_3 x_4}$
- $\ket{s} = H^{\otimes 4} \ket{0}^{\otimes 4} = \sum_{x \in \{0,1\}^4} \ket{x} = \ket{0000} + \ket{1000} + \ket{0100} + \cdots + \ket{1111}$
- $I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$
- $Z_1 = Z \otimes I \otimes I \otimes I$
- $Z_2 = I \otimes Z \otimes I \otimes I$
- $Z_3 = I \otimes I \otimes Z \otimes I$
- $Z_4 = I \otimes I \otimes I \otimes Z$
- $p \geq 1$ なる自然数
- $\beta = (\beta_1,\cdots,\beta_p) \in \R^p$
- $\gamma = (\gamma_1,\cdots,\gamma_p) \in \R^p$

# MAXCUT 問題を追いかける

MAXCUT 問題は、無向グラフの頂点を 2 色に塗り分ける時、辺（枝）で繋がっている隣接頂点同士が異なる色の場合に辺をカットするという試行を考えた時に、最大のカット数は幾つになるか？を求める問題である。

## コスト関数を作る

教材ではいきなり問題のハミルトニアンが出てきて眺めているうちに問題が解けてしまいよく分からないことになるので、最初から Appendix を見つつハミルトニアンを作っていく。[^2]

[^2]: どの本に書いてあったか忘れたが、最適化したい対象のことをハミルトニアンと呼ぶことがあるらしく、今回のケースでも特に何かしらの物理系に関連づいているわけではなさそうだ。

この教材の例題では $n=4$ で、コスト関数 $C(x)$ は具体的には

$$
\begin{align*}
C(x) &= x_1 (1-x_2) + x_2 (1-x_1) + x_2 (1-x_3) + x_3 (1-x_2) \\
&+ x_3 (1-x_4) + x_4 (1-x_3) + x_4 (1-x_1) + x_1 (1-x_4)
\tag{1}
\end{align*}
$$

となる。ここで、0 の頂点を $x_1$ に、1 の頂点を $x_2$ に、2 の頂点を $x_3$ に、3 の頂点を $x_4$ に割り当てている。
この関数は $x_i (1-x_j) + x_j (1-x_i)$ の形の項のペアを 4 つ連結したものであり、$x_i = x_j$ で 0 となり、$x_i \neq x_j$ で 1 となる。[^3]

[^3]: 今回の例題は、順に赤、青、赤、青で塗ることですべての辺（枝）について 1 をとれるので、コスト関数は最適時に 4 をとることができる。

$x=x_1 x_2 x_3 x_4$ の順番で書くことにし、具体的に $C(x)$ を計算すると、$C(0000) = 0, C(1000) = 2, C(0100) = 2, C(1100) = 2, C(0010) = 2, C(1010) = 4, C(0110) = 2, C(1110) = 2, C(0001) = 2, C(1001) = 2, C(0101) = 4, C(1101) = 2, C(0011) = 2, C(1011) = 2, C(0111) = 2, C(1111) = 0$ である。この時点で雰囲気的に明らかなのだが、コストが最大になっている $1010$ と $0101$ が問題の解であって、これを量子アルゴリズムで求めましょうということになる。

## ハミルトニアンを作る

教材に従い、スペクトル分解のような形のハミルトニアンを作ると以下のようになる。

$$
\begin{align*}
H &= \sum_{x \in \{0,1\}^4} C(x) \ket{x} \bra{x} \\
&= 2 \ket{1000}\bra{1000} + 2 \ket{0100}\bra{0100} + 2 \ket{1100}\bra{1100} \\
&+ 2 \ket{0010}\bra{0010} + 4 \ket{1010}\bra{1010} + 2 \ket{0110}\bra{0110} + 2 \ket{1110}\bra{1110} \\
&+ 2 \ket{0001}\bra{0001} + 2 \ket{1001}\bra{1001} + 4 \ket{0101}\bra{0101} + 2 \ket{1101}\bra{1101} \\
&+ 2 \ket{0011}\bra{0011} + 2 \ket{1011}\bra{1011} + 2 \ket{0111}\bra{0111}
\tag{2}
\end{align*}
$$

## ハミルトニアンの期待値と MAXCUT 問題の解

ここで、実は求めたい状態があって、それは MAXCUT 問題の 2 つの解に対応する状態

$$
\begin{align*}
\ket{\psi_\text{opt}} = \frac{1}{\sqrt{2}}(\ket{1010} + \ket{0101})
\end{align*}
$$

である。この状態に対するハミルトニアンの期待値は $\braket{\psi_\text{opt}|H|\psi_\text{opt}} = 4$ でコスト関数の最大値をとる形で最大となる。

では、この $\ket{\psi_\text{opt}}$ はどこからくるのか？というと、あるパラメータ付きのユニタリ演算子、従って PQC（パラメータ付き量子回路）$U(\beta,\gamma)$ を使って、$\ket{\psi_\text{opt}} \leftarrow \ket{\psi (\beta, \gamma)} := U(\beta,\gamma) \ket{s}$ という形で得られることになっている。

つまり、$\braket{\psi (\beta, \gamma)|H|\psi (\beta, \gamma)}$ が最大値（ここでは 4）をとるようにパラメータ $\beta$ と $\gamma$ を最適化し、その時に $\beta_\text{opt}$ と $\gamma_\text{opt}$ になったとすると、$\ket{\psi_\text{opt}} \approx \ket{\psi (\beta_\text{opt}, \gamma_\text{opt})}$ が得られたと考えることにしますというストーリである。

MAXCUT 問題の最終的な解は $\ket{\psi (\beta_\text{opt}, \gamma_\text{opt})}$ を計算基底について測定することで、概ね $\ket{1010}$ と $\ket{0101}$ が 1/2 ずつくらいの確率で得られるであろうことから $x_1=1, x_2=0, x_3=1, x_4=0$ 或は $x_1=0, x_2=1, x_3=0, x_4=1$（つまり頂点を交互に赤のグループと青のグループに割り振る）が解として得られる。

ストーリーが分かったところで、(2) 式のハミルトニアンを量子回路として実装する必要がある。これはどうすれば良いのであろうか？ここで[前回](/articles/dwd-qiskit-qaoa01)の記事の出番である。

# 量子回路の実装

## 問題のハミルトニアン

[前回](/articles/dwd-qiskit-qaoa01)の記事の結果を使って (1) を書き直すと、

$$
\begin{align*}
H &= \left(\frac{1-Z_1}{2}\right) \left(\frac{1+Z_2}{2}\right) + \left(\frac{1-Z_2}{2}\right) \left(\frac{1+Z_1}{2}\right) + \cdots \\
&= \frac{1-Z_1 Z_2}{2} + \frac{1-Z_2 Z_3}{2} + \frac{1-Z_3 Z_4}{2} + \frac{1-Z_4 Z_1}{2} \\
&= 2I - \frac{1}{2}(Z_1 Z_2 + Z_2 Z_3 + Z_3 Z_4 + Z_4 Z_1)
\end{align*}
$$

となる。これが (2) 式と等しいというのが[前回](/articles/dwd-qiskit-qaoa01)確認した内容であった。

定数項 $2I$ は期待値をとる時に常に定数 $2$ の寄与であるので最適化の考察上は除外して問題ない。よって、textbook に倣い

$$
\begin{align*}
H_P &= Z_1 Z_2 + Z_2 Z_3 + Z_3 Z_4 + Z_4 Z_1 \\
&= (Z \!\otimes\! Z \!\otimes\! I \!\otimes\! I) + (I \!\otimes\! Z \!\otimes\! Z \!\otimes\! I) + (I \!\otimes\! I \!\otimes\! Z \!\otimes\! Z) + (Z \!\otimes\! I \!\otimes\! I \!\otimes\! Z)
\tag{3}
\end{align*}
$$

の部分だけを考える[^4]。負符号も取り去ったので、この “新しい” ハミルトニアンに対しては期待値を**最小化**する形での最適化を行うことになる。

[^4]: textbook ではこの時点では係数 $1/2$ がかかっているが、後の説明やコードの実装では $1/2$ を捨てているようなので、ここでも捨てておく。最適化処理に影響はない。

## ミキシングハミルトニアン

“ミキサー” と呼んでいる文献もある。ここはまったく理解できていないのでそのまま受け入れる。

$$
\begin{align*}
H_B = (X \!\otimes\! I \!\otimes\! I \!\otimes\! I) + (I \!\otimes\! X \!\otimes\! I \!\otimes\! I) + (I \!\otimes\! I \!\otimes\! X \!\otimes\! I) + (I \!\otimes\! I \!\otimes\! I \!\otimes\! X)
\end{align*}
$$

## パラメータ付き量子回路 $U(\beta, \gamma)$

$\ket{\psi_\text{opt}} \leftarrow \ket{\psi (\beta, \gamma)} := U(\beta,\gamma) \ket{s}$ という状態を作る回路が必要であるが、ここも理解できていないのでそのまま受け入れる。

パラメータ $\beta = (\beta_1,\cdots,\beta_p) \in \R^p$ と $\gamma = (\gamma_1,\cdots,\gamma_p) \in \R^p$ に対して

$$
\begin{align*}
U(\beta, \gamma) = \underbrace{\left( e^{-i \beta_p H_B} e^{-i \gamma_p H_P} \right) \left( e^{-i \beta_{p-1} H_B} e^{-i \gamma_{p-1} H_P} \right) \cdots \left( e^{-i \beta_1 H_B} e^{-i \gamma_1 H_P} \right)}_{p}
\end{align*}
$$

という $2p$ 個のユニタリ行列の積を考える。$e^{-i \gamma_{i} H_P}$ は $Rzz$ ゲートで、$e^{-i \beta_{i} H_B}$ は $Rx$ ゲートで実装が可能である。

ここまでで必要な回路の実装の道筋が見えた。後は回路のパーツを組み合わせて、ハミルトニアン $H_P$ の期待値を COBYLA オプティマイザを使って最小化し、パラメータの最適値 $\beta_\text{opt}$ と $\gamma_\text{opt}$ を求めることで、MAXCUT 問題の解を得ることができる。

最後に期待値の計算の部分を眺めて終わりにしよう。

# 期待値の計算

$$
\begin{align*}
\ket{\psi(\beta, \gamma)} = U(\beta, \gamma)\ket{s} = \alpha_{0000} \ket{0000} + \alpha_{1000} \ket{1000} + \cdots + \alpha_{1111} \ket{1111}
\end{align*}
$$

となったとする。この時 $H_P$ に対する期待値を計算したい。

いきなりは難しいので、練習問題として $\ket{\psi} = \ket{1100}$ について期待値を見てみよう:

$$
\begin{align*}
&\ \braket{\psi|H_P|\psi} \\
=&\ \braket{1100 | (Z \!\otimes\! Z \!\otimes\! I \!\otimes\! I) + (I \!\otimes\! Z \!\otimes\! Z \!\otimes\! I) + (I \!\otimes\! I \!\otimes\! Z \!\otimes\! Z) + (Z \!\otimes\! I \!\otimes\! I \!\otimes\! Z) | 1100} \\
=&\ (-1)(-1) + (-1) + (1) + (-1) = 0
\end{align*}
$$

である。もう 1 つのサンプルとして $\ket{\psi} = \ket{1010}$ でも試してみよう:

$$
\begin{align*}
&\ \braket{\psi|H_P|\psi} \\
=&\ \braket{1010 | (Z \!\otimes\! Z \!\otimes\! I \!\otimes\! I) + (I \!\otimes\! Z \!\otimes\! Z \!\otimes\! I) + (I \!\otimes\! I \!\otimes\! Z \!\otimes\! Z) + (Z \!\otimes\! I \!\otimes\! I \!\otimes\! Z) | 1010} \\
=&\ (-1) + (-1) + (-1) + (-1) = -4
\end{align*}
$$

である。他のケースを見ても分かるが、$1010$ のようなビット列を考えた時に、先頭と末尾を繋げてリング状にした場合に、$0$ と $1$ が切り替わるところの個数だけ $-1$ し、$00$ や $11$ のように同じビットが連なるところでは $+1$ する形で求まる[^5]。

[^5]: 実装は $+1$ ではなく $+0$ しているが最小値を求める上では問題ない。

よって、この “部分的な期待値” を計算する関数を $f$ とでもすると、

$$
\begin{align*}
f(0000) = 4,\ f(1000) = 0, \cdots,\ f(1100) = 0, \cdots,\ f(1010) = -4, \cdots
\end{align*}
$$

ようになっており、全体の期待値は

$$
\begin{align*}
\braket{\psi(\beta, \gamma) | H_P |\psi(\beta, \gamma)} = |\alpha_{0000}|^2 f(0000) + |\alpha_{1000}^2| f(1000) + \cdots + |\alpha_{1111}|^2 f(1111)
\end{align*}
$$

で求まる。これが textbook の関数 `compute_expectation` の実装である。

# まとめ

大分長くなってしまったが、Qiskit textbook の [Solving combinatorial optimization problems using QAOA](https://learn.qiskit.org/course/ch-applications/solving-combinatorial-optimization-problems-using-qaoa) の前半部分の例題「MAXCUT」について全体の流れを「よく分からないところはそういうものとして割り切って受け入れる」形で眺めてみた。

読み返しつつまとめてみてもまだまだスッキリとはしないので、また何度も読み返して考察する必要があるように感じる。他のケースでの問題設定での解を求めてみたりもしたい[^6]。

[^6]: たぶん実機では試さない。理屈上はいけるのだが、古典-量子のハイブリッドモデルなので、パラメータ更新を古典コンピュータで試して次に量子コンピュータにタスクを投げる時に毎回もの凄く待たされることになる。恐らく全体の計算が収束するまでにかなりの時間を要すると考えられる・・・。
