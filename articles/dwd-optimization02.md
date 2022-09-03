---
title: "最適化について考える (2) — SVM"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "ポエム"]
published: false
---

# 目的

[前回](/derwind/articles/dwd-optimization01)に引き続きという形で今回は SVM（サポートベクターマシン）を扱う。

SVM は何となく本で読んで何となくサポートベクトルと決定境界とのマージンを最大化するように最適化されるといったことを読んでそれなりに納得し、API を叩くことになると思う。結局何が行われているのかさっぱり分からないままになるので、実際に手を動かして計算をしてみたいと思う。

# 今回の範囲

3 点からなるデータセットにおいて、決定境界を求める。主問題を設定し、そこから双対問題を求める。これを具体的に眺めることでどういったものであるかを少しでも掴んでいく。

# データセット

3 パターン考えたが以下のものが教育的観点で良さそうに感じたのでこれを扱う。他のケースでも結局は同じ答えになるのだが、手計算で確認したところ、計算の煩雑な部分が都度変わり、あまり本質的ではないのに記事が長くなるので思い切って問題設定を 1 種類に絞った。

![](/images/dwd-optimization02/001.png)

にように、データセット $\{ (\xi_1, t_1),(\xi_2, t_2),(\xi_3, t_3)\}$ を用意する。$t_i \in \{-1, +1\}$ である。

# 主問題

暫く PRML の通りに書き下していく。

$$
\begin{align*}
y(\mathbf{x}) &= \mathbf{w}^T \mathbf{x} + b = \braket{\mathbf{w}, \mathbf{x}} + b = w_1 x_1 + w_2 x_2 + b
\end{align*}
$$

の形で決定境界を求めたい。$\operatorname{sgn}(y(\mathbf{x})) > 0$ と $\operatorname{sgn}(y(\mathbf{x})) < 0$ によって 2 つのクラスを分類する。

まだ、決定境界を定めるパラメータ値が求まっていないので、境界の代わりに “超平面” と呼ぶことにする[^1]。

[^1]: 今回は普通の平面である。

超平面への $\xi_i$ からの距離は $|y(\xi_i)| / |\mathbf{w}|$ で与えられる。ラベル $t_i$ の絶対値を 1 にとっていることから、$|y(\xi_i)| = t_i y(\xi_i)$ が成立することに注意する。すると、$\xi_i$ からの超平面への距離は

$$
\begin{align*}
\frac{t_i y(\xi_i)}{|\mathbf{w}|} = \frac{t_i (\mathbf{w}^T \xi_i + b)}{|\mathbf{w}|}
\end{align*}
$$

となる。サポートベクターはこれが最小となるものであるので、サポートベクターの長さは

$$
\begin{align*}
\min_i \frac{t_i (\mathbf{w}^T \xi_i + b)}{|\mathbf{w}|} = \frac{1}{|\mathbf{w}|} \min_i [t_i (\mathbf{w}^T \xi_i + b)]
\end{align*}
$$

となる。この値、つまりサポートベクターと超平面への距離 — マージン — を最大化したいということなので、Max-Min 問題

$$
\begin{align*}
\mathbf{w}_\text{optim},b_\text{optim} = \argmax_{\mathbf{w},b} \frac{1}{|\mathbf{w}|} \min_i [t_i (\mathbf{w}^T \xi_i + b)]
\end{align*}
$$

を解く事で決定境界を定めるパラメータが求まる。

# 文献

[PRML] C. M. Bishop, [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), pp.325-331, Springer, 2006
