---
title: "最適化について考える (1) — Lagrange の未定乗数法"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "ポエム"]
published: true
---

# 目的

量子機械学習の勉強をしているうちに、古典機械学習と言えど軽く本で概要を読んだ後は多くの場合単に API を叩いているだけで、中の動きについてそれほど理解しているとは言えないなと思った。今回、SVM (サポートベクトルマシン) について調べたが、その一歩手前の内容として、基本的な「Lagrange の未定乗数法 (Lagrange multiplier method)」についてごく簡単なケースについて考えてみたい。

# 今回の範囲

2 変数関数についての Lagrange の未定乗数法を簡単に眺める。条件はよく教科書に書いてある内容からは少し書き換える。

# Lagrange の未定乗数法

$\R^2$ の適当な領域で定義された実数値の可微分な陰関数 $f$ と $\varphi$ があって、

- 点 $\mathbf{a} = (a_x^0, a_\xi^0)$ のある近傍で $\varphi(x, \xi) = 0$ は $\xi = \psi(x)$ と陽に解け、
- かつ $\frac{d \psi}{d x} (a_x^0)$ は有限確定で逆数を持つ

ものとする。この時、束縛条件 $\varphi(x, \xi) = 0$ のもと $f$ が $\mathbf{a}$ で極値をとる場合、適当な定数 $\lambda \in \R$ が存在して、

$$
\begin{align*}
L(x, \xi; \lambda) = f(x, \xi) + \lambda \varphi(x, \xi)
\tag{1}
\end{align*}
$$

と置くと、極値を与える点 $\mathbf{a}$  において

$$
\begin{align*}
\frac{\partial L}{\partial x}(a_x^0, a_\xi^0; \lambda) &= 0, \\
\frac{\partial L}{\partial \xi}(a_x^0, a_\xi^0; \lambda) &= 0
\end{align*}
$$

が成立する。

## この定理は何が嬉しいか？落とし穴は？

主張の対偶をとると、「どのような $\lambda$ をとっても $L$ の $x$ と $\xi$ の微分が $\mathbf{a}$ で同時に消えないなら、$f$ は $\mathbf{a}$ で極値をとらない」となる。このため、**微分が消える点は極値を与える点の候補**となる。この事実は、最適化問題

$$
\begin{align*}
\min_{x, \xi} f(x, \xi) \quad\text{subject to}\quad \varphi(x, \xi) = 0
\end{align*}
$$

などを解きたい場合の強力な武器になることを示している。

但し、極大値か極小値かは分からないし、最大値・最小値を考えたい場合、極値だけでなく、束縛条件による境界での値も確認しないとならない。

## 証明

仮定より $\mathbf{x} = (x, \xi)^T$ が $\mathbf{a}$ に十分に近い時、$\xi = \psi(x)$ とできる。これを使って、

$$
\begin{align*}
\tilde{f}(x) := f(x, \psi(x))
\end{align*}
$$

という関数を定義する。これを $x$ について $\mathbf{a}$ において微分すると $f$ がこの点で極値をとることから $\frac{\partial f}{\partial x}$ と $\frac{\partial f}{\partial \xi}$ と同時に消えるので

$$
\begin{align*}
\frac{d \tilde{f}}{d x}(\mathbf{a}) = \frac{\partial f}{\partial x}(a_x^0, a_\xi^0) + \frac{\partial f}{\partial \xi}(a_x^0, a_\xi^0) \frac{d \psi}{d x}(a_x^0) = 0
\tag{2}
\end{align*}
$$

を得る。また、$\varphi(x, \psi(x)) = 0$ を $x$ について $\mathbf{a}$ において微分すると

$$
\begin{align*}
0 = \frac{\partial \varphi}{\partial x}(a_x^0, a_\xi^0) + \frac{\partial \varphi}{\partial \xi}(a_x^0, a_\xi^0) \frac{d \psi}{d x}(a_x^0)
\tag{3}
\end{align*}
$$

を得る。つまり、$\frac{d \psi}{d x}(a_x^0) = - \frac{\partial \varphi}{\partial x}(a_x^0, a_\xi^0) / \frac{\partial \varphi}{\partial \xi}(a_x^0, a_\xi^0)$ となる。(3) 式を (2) 式に代入すると、

$$
\begin{align*}
\frac{\partial f}{\partial x}(a_x^0, a_\xi^0) - \frac{\frac{\partial \varphi}{\partial x}(a_x^0, a_\xi^0)}{\frac{\partial \varphi}{\partial \xi}(a_x^0, a_\xi^0)} \frac{\partial f}{\partial \xi}(a_x^0, a_\xi^0) = 0
\tag{4}
\end{align*}
$$

を得る。$\lambda := - \frac{\partial f}{\partial \xi}(a_x^0, a_\xi^0) / \frac{\partial \varphi}{\partial \xi}(a_x^0, a_\xi^0)$ と置くと、それ自身を変形して

$$
\begin{align*}
\frac{\partial f}{\partial \xi}(a_x^0, a_\xi^0) + \lambda \frac{\partial \varphi}{\partial \xi}(a_x^0, a_\xi^0) = \frac{\partial}{\partial \xi}(f + \lambda \varphi)(a_x^0, a_\xi^0) = 0
\end{align*}
$$

を得ると同時に、(4) に代入することで

$$
\begin{align*}
\frac{\partial f}{\partial x}(a_x^0, a_\xi^0) + \lambda \frac{\partial \varphi}{\partial x}(a_x^0, a_\xi^0) = \frac{\partial}{\partial x}(f + \lambda \varphi)(a_x^0, a_\xi^0) = 0
\end{align*}
$$

となる。$\blacksquare$

## 注意

この定理で一般的に書かれている前提条件は、陰関数 $\varphi(x, \xi) = 0$ が都合の良い形で $x$ または $\xi$ について陽に解けるための十分条件である。今回は試しにそのことを明確に書いてみた。

# 例題

$x^2 + y^2 = 1$ の束縛条件のもとでの、$f(x, y) = x - y$ の最大値を求める。

$L(x, y, \lambda) = x - y + \lambda (x^2 + y^2 - 1)$ と置く。$\frac{\partial L}{\partial x} = 1 + 2 \lambda x = 0$, $\frac{\partial L}{\partial y} = -1 + 2 \lambda y = 0$ としてみると、$(x,y) = (- \frac{1}{2 \lambda}, \frac{1}{2 \lambda})$ が極値を与える点の候補になる。束縛条件を満たすためには、$\left( - \frac{1}{2 \lambda}\right)^2 + \left( \frac{1}{2 \lambda}\right)^2 = 1$ である必要があるので、$\lambda = \pm \frac{1}{\sqrt{2}}$ でなければならない。つまり、$(x,y) = (- \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$ か $(x,y) = (\frac{1}{\sqrt{2}}, - \frac{1}{\sqrt{2}})$ が極値を与える点の候補であり、また最大値か最小値を与える点の候補になる。

実際には、$(x,y) = (- \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}})$ が最大値 $\sqrt{2}$ を与える。細かい計算が面倒臭いので、以下のグラフを眺めて納得することにしよう。

![](/images/dwd-optimization01/001.png)

# まとめ

駆け足で Lagrange の未定乗数法について眺めた。沢山の変数がある場合にはかなり煩雑な証明になるが、実際には、陰関数を解いて、各与えられた式に放り込んで偏微分して、式同士を組み合わせて整理するだけである。

# 文献

[1] 水野克彦, 新講解析学, pp.141-142, 学術図書出版社, 1982
