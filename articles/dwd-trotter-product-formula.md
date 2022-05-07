---
title: "Trotter の積公式"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Qiskit", "ポエム"]
published: true
---

# 目的

よく量子コンピュータの文献で [リー・トロッター積公式](https://ja.wikipedia.org/wiki/%E3%83%AA%E3%83%BC%E3%83%BB%E3%83%88%E3%83%AD%E3%83%83%E3%82%BF%E3%83%BC%E7%A9%8D%E5%85%AC%E5%BC%8F) が出てくるので、紙と鉛筆で殴り書きで検証して遊んで放置していた。Qiskit textbook [Proving Universality](https://qiskit.org/textbook/ch-gates/proving-universality.html) を読んでいると、しれっと公式が使われていたので、行き場を無くしていた放置メモを記事にすることにした。

# 概要

ヒルベルト空間上の一般には非有界な自己共役作用素 $A$, $B$ について、$A+B$ が 定義域 $\mathcal{D}(A+B) := \mathcal{D}(A) \cap \mathcal{D}(B)$ 上で本質的に自己共役である場合に、**$A$ と $B$ が非可換であっても**

$$
\begin{align*}
\mathrm{s}\!-\!\lim_{n\to\infty} (e^{it \frac{A}{n}} e^{it \frac{B}{n}})^n = e^{i t (A+B)}
\end{align*}
$$

が成立する・・・というのが、Trotter の積公式であるが・・・これは流石に大袈裟すぎるので、文献 [RS1] に従う形で **Lie の積公式**を見る。量子コンピュータのコンテキストではこれで十分に思われる。見るといっても証明も長くはないので、眺めつつ文献では端折られている計算の細部を補いましょうという程度のことである・・・。

# Lie の積公式

$A$ と $B$ を $d$ 次正方行列とする。この時、以下が成立する。

$$
\begin{align*}
\exp(A + B) = \lim_{n\to\infty}\left[\exp\left(\frac{A}{n}\right) \exp\left(\frac{B}{n}\right)\right]^n
\end{align*}
$$

量子コンピュータの本とかの場合だと、例えば $d=2$ として Pauli 行列 $X$, $Z$ をとってきて、$A=iaX$, $B=ibZ$ と置いて利用すればよく、十分に大きな $n$ の時に両辺の $n$ 乗根をとって、近似的に

$$
\begin{align*}
e^{iaX/n} e^{ibZ/n} \simeq e^{i(aX + bZ)/n}
\end{align*}
$$

とできます、という話になる。

# 証明

$S_n = \exp\left(\frac{A+B}{n}\right)$, $T_n = \exp\left(\frac{A}{n}\right) \exp\left(\frac{B}{n}\right)$ と置く。
直接計算で以下が示される。

$$
\begin{align*}
\sum_{m=0}^0 S_1^m (S_1 - T_1) T_1^{-m} &= S_1 - T_1 = S_1^1 - T_1^1, \\
\sum_{m=0}^1 S_2^m (S_2 - T_2) T_2^{1-m} &= (S_2 - T_2)T_2 + S_2(S_2 - T_2) = S_2^2 - T_2^2, \\
\sum_{m=0}^2 S_3^m (S_3 - T_3) T_3^{2-m} &= (S_3 - T_3)T_3^2 + S_3(S_3 - T_3)T_3 + S_3^2(S_3 - T_3) \\
&= S_3^3 - T_3^3, \\
\end{align*}
$$

より一般に、

$$
\begin{align*}
\sum_{m=0}^{n-1} S_n^m (S_n - T_n) T_n^{n-1-m} = S_n^n - T_n^n
\tag{1}
\end{align*}
$$

が期待される。実際、左辺を直接展開すると、

$$
\begin{align*}
&\ \sum_{m=0}^{n-1} S_n^m (S_n - T_n) T_n^{n-1-m} \\
=&\ (S_n \!-\! T_n)T_n^{n-1} \!+\! S_n(S_n \!-\! T_n)T_n^{n-2} \!+\! S_n^2(S_n \!-\! T_n)T_n^{n-3} \!+\! \cdots \!+\! S_n^{n-1}(S_n \!-\! T_n) \\
=&\ (S_nT_n^{n-1} \!-\! T_n^{n}) \!+\! (S_n^2T_n^{n-2} \!-\! S_n T_n^{n-1}) \!+\! (S_n^3T_n^{n-3} \!-\! S_n^2T_n^{n-2}) \!+\! \cdots \!+\! (S_n^{n} \!-\! S_n^{n-1}T_n)
\end{align*}
$$

となり、$i$ 番目の括弧の前の項と $i+1$ 番目の括弧の後ろの項 がキャンセルし合い、残るのは $- T_n^{n} + S_n^{n}$ である。よって (1) が一般の $n$ の対して成立することが分かった。

$S_n^n - T_n^n$ を評価したい。このために、まずは $X = \frac{A}{n}$, $Y = \frac{B}{n}$ とおいて $S_n - T_n$ の評価を行う。

$$
\begin{align*}
&\ \sum_{m=0}^\infty \frac{1}{m!} (X+Y)^m - \left(\sum_{m=0}^\infty \frac{1}{m!} X^m\right) \left(\sum_{m=0}^\infty \frac{1}{m!} Y^m\right) \\
\leq&\ \left\{\! I \!+\! (X\!+\!Y) \!+\! \frac{1}{2} (X\!+\!Y)^2 \!+\! \cdots \!\right\} - \left(\! I \!+\! X \!+\! \frac{1}{2} X^2 \!+\! \cdots \!\right) \left(\! I \!+\! Y \!+\! \frac{1}{2} Y^2 \!+\! \cdots \!\right) \\
\leq&\ \frac{1}{2}(YX - XY) + (\text{3 次以上の項})
\tag{2}
\end{align*}
$$

ここで、“3 次以上の項” を評価する。この部分は上記の Taylor 展開の 3 次以上の項なので、

$$
\begin{align*}
&\ \|(\text{3 次以上の項})\| \\
\leq&\ \left\{\! \frac{1}{3!} \|\!X\!\!+\!\!Y\!\|^3 \!\!+\!\! \frac{1}{4!} \|\!X\!\!+\!\!Y\!\|^4 \!\!+\! \cdots \!\right\} \!+\! \left(\! \|\!X\!\| \!\!+\!\! \frac{1}{2} \|\!X\!\|^2 \!\!+\! \cdots \!\right) \!\! \left(\! \|\!Y\!\| \!\!+\!\! \frac{1}{2} \|\!Y\!\|^2 \!\!+\! \cdots \!\right) \!-\! \| XY \| \\
\leq&\ \left\{\! \frac{1}{3!} \frac{\|\!A\!\!+\!\!B\!\|^3}{n^3} \!\!+\!\! \frac{1}{4!} \frac{\|\!A\!\!+\!\!B\!\|^4}{n^4} \!\!+\! \cdots \!\right\} \!+\! \left(\! \frac{\|\!A\!\|}{n} \!\!+\!\! \frac{1}{2} \frac{\|\!A\!\|^2}{n^2} \!\!+\! \cdots \!\right) \!\! \left(\! \frac{\|\!B\!\|}{n} \!\!+\!\! \frac{1}{2} \frac{\|\!B\!\|^2}{n^2} \!\!+\! \cdots \!\right) \!-\! \frac{\| AB \|}{n^2} \\
\leq&\ \frac{1}{n^3} \!\! \left[ \! \left\{\! \frac{1}{3!} \|\!A\!\!+\!\!B\!\|^3 \!\!+\!\! \frac{1}{4!} \|\!A\!\!+\!\!B\!\|^4 \!\!+\! \cdots \!\right\} \!+\! \left(\! \|\!A\!\| \!\!+\!\! \frac{1}{2} \|\!A\!\|^2 \!\!+\! \cdots \!\right) \!\! \left(\! \|\!B\!\| \!\!+\!\! \frac{1}{2} \|\!B\!\|^2 \!\!+\! \cdots \!\right) \!-\! \| AB \| \! \right] \\
\leq&\ \frac{1}{n^3} \left\{\exp( \|A+B\|) + \exp \|A\| \exp \|B\| \right\}
\end{align*}
$$

と評価できる。これを (2) と併せると、

$$
\begin{align*}
&\ \left\|\sum_{m=0}^\infty \frac{1}{m!} (X+Y)^m - \left(\sum_{m=0}^\infty \frac{1}{m!} X^m\right) \left(\sum_{m=0}^\infty \frac{1}{m!} Y^m\right) \right\| \\
\leq&\ \frac{1}{n^2} \cdot \frac{1}{2} \|BA-AB\| + \frac{1}{n^3} \left\{\exp( \|X+Y\|) + \exp \|X\| \exp \|Y\| \right\} = \mathcal{O}(1/n^2)
\end{align*}
$$

つまり、

$$
\begin{align*}
\| S_n - T_n \| = \mathcal{O}(1/n^2)
\tag{3}
\end{align*}
$$

という評価式を得る。

また

$$
\begin{align*}
\|S_n^m\| \leq \|S_n\|^m \leq \|S_n\|^n \leq \exp(\|A+B\|) \leq C
\tag{4}
\end{align*}
$$

$$
\begin{align*}
\|T_n^{n-1-m}\| \leq \|T_n\|^{n-1-m} \leq \|T_n\|^n \leq \exp(\|A\|) \exp(\|B\|) \leq C
\tag{5}
\end{align*}
$$

という評価が得られるので、(1) と (3) と併せて

$$
\begin{align*}
\| S_n^n - T_n^n \| \leq \sum_{m=0}^{n-1} C^2 \cdot \mathcal{O}(1/n^2) = \mathcal{O}(1/n) \to 0 \quad\text{as}\quad n \to \infty
\end{align*}
$$

$S_n^n = \exp(A+B)$ で有限確定であるので主張を得た。

# まとめ

あまり広くない横幅の中に無理矢理数式を詰め込んだので汚くなったが、求めたい式を適当に分解して、個別に $n$ についてどれくらいのオーダーで大きくなるのか小さくなるのかを評価するという比較的単純な計算で証明することができた。

$A$ と $B$ が可換である場合、$\exp(A+B) = \exp(A) \exp(B)$ が成立するので特に何も驚くことはないのだが、非可換であっても同様の結果が成立するので興味深い。比較的古典的な定理だと思われるが、こういったものが量子コンピュータのゲートの近似に使われるのは面白い。[^1]

[^1]: ゲートの近似という点では Solovay-Kitaev の定理も有名と思われる。こちらはニールセン&チャンによると 1995 年と比較的近年示されたものであるらしい。Lie の積公式が示されたのはいつのことかわからないが、Trotter の積公式は 1959 年となかなか古い。

# 文献

[RS1] M. Reed and B. Simon. Methods of Modern Mathematical Physics, I. Functional Analysis, pp.295, Academic Press, New York, 1981
