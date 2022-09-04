---
title: "最適化について考える (2) — SVM"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "ポエム"]
published: true
---

# 目的

[前回](/derwind/articles/dwd-optimization01)に引き続きという形で今回は SVM（サポートベクターマシン）を扱う。

SVM は何となく本で読んで何となくサポートベクトルと決定境界とのマージンを最大化するように最適化されるといったことを読んでそれなりに納得し、API を叩くことになると思う。結局何が行われているのかさっぱり分からないままになるので、実際に手を動かして計算をしてみたいと思う。

# 今回の範囲

3 点からなるデータセットにおいて、決定境界を求める。主問題を設定し、そこから双対問題を求める。これを具体的に眺めることでどういったものであるかを少しでも掴んでいく。

# データセット

3 パターン考えたが以下のものが教育的観点および計算の楽さの観点で良さそうに感じたのでこれを扱う。他のケースでも結局は似たような感じにはなるのだが、手計算で確認したところ計算の煩雑な部分が都度変わり、あまり本質的ではないのに記事が長くなるので思い切って問題設定を 1 種類に絞った。

下図

![](/images/dwd-optimization02/001.png)

のように、データセット $\{ (\xi_1, t_1),(\xi_2, t_2),(\xi_3, t_3)\}$ を用意する。$t_i \in \{-1, +1\}$ である。

# 主問題

暫く PRML の通りに書き下していく。

$$
\begin{align*}
y(\mathbf{x}) &= \mathbf{w}^T \mathbf{x} + b = \braket{\mathbf{w}, \mathbf{x}} + b = w^1 x^1 + w^2 x^2 + b
\tag{1}
\end{align*}
$$

の形で決定境界を求めたい。$\operatorname{sgn}(y(\mathbf{x})) > 0$ と $\operatorname{sgn}(y(\mathbf{x})) < 0$ によって 2 つのクラスを分類することになる。

まだ、決定境界を定めるパラメータ値が求まっていないので、境界の代わりに “超平面” と呼ぶことにする[^1]。

[^1]: 今回は普通の平面である。

超平面への $\xi_i$ からの距離は $|y(\xi_i)| / |\mathbf{w}|$ で与えられる[^2]。ラベル $t_i$ の絶対値を 1 にとっていることから、$|y(\xi_i)| = t_i y(\xi_i)$ が成立することに注意する。すると、$\xi_i$ からの超平面への距離は

[^2]: 超平面の法線方向のベクトルは $\mathbf{w}$ であるので、$\xi_i + \mathbf{w} t$ が超平面に乗る $t$ を求めれば良い。この時、$|\mathbf{w} t| = |\mathbf{w}| |t|$ が計算したい距離になる。$0 = y(\xi_i + \mathbf{w} t) = \mathbf{w}^T \xi_i + \mathbf{w}^T \mathbf{w} t + b = y(\xi_i) + |\mathbf{w}|^2 t$ なので、$|\mathbf{w}| |t| = |y(\xi_i)| / |\mathbf{w}|$ が超平面への距離となる。

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
\mathbf{w}^*,b^* = \argmax_{\mathbf{w},b} \frac{1}{|\mathbf{w}|} \min_i [t_i (\mathbf{w}^T \xi_i + b)]
\end{align*}
$$

を解く事で決定境界を定めるパラメータが求まる。

$\mathbf{w} \to \kappa \mathbf{w}$, $b \to \kappa b$ のスケーリングで $|y(\xi_i)| / |\mathbf{w}|$ は不変なので、$\kappa$ を調整することで、「超平面に最も近い $\xi_i$ については $|y(\xi_i)| = t_i (\mathbf{w}^T \xi_i + b) = 1$ と仮定」することができる。この時、すべてのデータについて $t_i (\mathbf{w}^T \xi_i + b) \geq 1$ となる。

この状況下で残っている課題は $\frac{1}{|\mathbf{w}|}$ の最大化であるが、同値な条件として $\frac{1}{2} |\mathbf{w}|^2$ の最小化が考えられる。ここまでをまとめると、**主問題**としての凸最適化である 2 次計画問題

$$
\begin{align*}
\min \frac{1}{2} |\mathbf{w}|^2 \quad\text{subject to}\quad t_i (\mathbf{w}^T \xi_i + b) \geq 1
\tag{2}
\end{align*}
$$

を得る。

# 双対問題

今回は主問題のままでは解くのが難しいので双対問題と呼ばれる同値な問題を求めてそちらを解くことにする。文献 [カーネル法] によると、凸最適化問題を解く際には双対問題を考えると有用なことが多いらしい。

ここでは PRML の記法そのままではなく、Lagrange の未定乗数法でよく見かける $\lambda = \{ \lambda_i\} \subset \R_{\geq 0}$ を Lagrange multiplier として用いる。Lagrange の未定乗数法で行ったように、最適化したい式と束縛条件に Lagrange multiplier を掛けた式の和として次の Lagrange 関数を考える。

$$
\begin{align*}
L(\mathbf{w}, b, \lambda) = \frac{1}{2} |\mathbf{w}|^2 - \sum_i \lambda_i \{ t_i (\mathbf{w}^T \xi_i + b) - 1 \}
\tag{3}
\end{align*}
$$

極値を与える点を求めるために $L$ の微分をとって 0 と置く; $\frac{\partial L}{\partial \mathbf{w}} = 0$ と $\frac{\partial L}{\partial b} = 0$ から

$$
\begin{align*}
\mathbf{w} &= \sum_i \lambda_i t_i \xi_i \\
0 &= \sum_i \lambda_i t_i
\tag{4}
\end{align*}
$$

を得る。これらを (3) 式に代入すると **$b$ の係数は (4) の第 2 式によって消える**ので、

$$
\begin{align*}
&\ \frac{1}{2} \sum_{j, k} (\lambda_j t_j \xi_j^T) (\lambda_k t_k \xi_k) - \sum_i \left[ \lambda_i t_i \{ (\sum_k \lambda_k t_k \xi_k^T) \xi_i + b \} - \lambda_i \right] \\
=&\ \sum_i \lambda_i - \frac{1}{2} \sum_{j, k} (\lambda_j t_j \xi_j^T) (\lambda_k t_k \xi_k)
\end{align*}
$$

を得る。添字を整理して最後の式を $\tilde{L}(\lambda)$ と置く:

$$
\begin{align*}
\tilde{L}(\lambda) = \sum_i \lambda_i - \frac{1}{2} \sum_{i, j} \lambda_i \lambda_j t_i t_j \xi_i^T \xi_j
\end{align*}
$$

この $\tilde{L}(\lambda)$ を用いた次の問題を**双対問題**と呼ぶ:

$$
\begin{align*}
\max \tilde{L}(\lambda) \quad\text{subject to}\quad \lambda_i \geq 0,\ i = 1,2,\cdots
\tag{5}
\end{align*}
$$

・・・

## ここで思いっきり省略する

何故急に $\tilde{L}(\lambda)$ を最大化する話が出てくるのか分りにくいのだが、**強双対性**という概念を用いるためにそういうことになるらしい、というのが現時点での理解である。つまり:

文献 [カーネル法] pp.212-215 辺りを読むことで、主問題が凸最適化であり、Slater 条件を満たす[^3]ことから強双対性が成立し、主問題の最適値 $p^*$ と双対問題の最適値 $d^*$ は一致し、またある $\lambda^* (= \argmax_\lambda \tilde{L}(\lambda))$ に対して $\tilde{L}(\lambda^*)$ と等しいことが分かる。

[^3]: 無茶苦茶なデータセットであれば満たさないかもしれないが、今回考えている 3 点データセットでは純分に分離性があり条件を満たす。

要するに、**今回のケースでは双対問題を解いて $\lambda^* = \{ \lambda^*_i \}$ を求めれば良い。** このようにして $\lambda^* = \{ \lambda^*_i \}$ が求まれば、(4) の第 1 式に代入して $\mathbf{w}$ を定め、その $\mathbf{w}$ を (1) 式に代入すると、決定境界は

$$
y(\mathbf{x}) = (\sum_i \lambda^*_i t_i \xi_i^T) \mathbf{x} + b = \sum_i \lambda^*_i t_i \braket{\xi_i, \mathbf{x}} + b
\tag{6}
$$

で定まることになる。

双対問題を導く過程で $b$ は消えていたので、$\lambda^*$ からは決定境界の法線ベクトルしか求まらない。$b$ の値は「超平面に最も近い $\xi_i$ については $|y(\xi_i)| = t_i (\mathbf{w}^T \xi_i + b) = 1$ と仮定」していたことから定めることになる。

# 例題

冒頭の 3 点データセットに対して決定境界を求める。

(4) 式を具体的に書くと:

$$
\begin{align*}
\mathbf{w} &= \begin{pmatrix}
2 \lambda_1 - 3 \lambda_2 \\
\lambda_1 + \lambda_2 + \lambda_3
\end{pmatrix}, \\
0 &= \lambda_1 - \lambda_2 + \lambda_3
\end{align*}
$$

また、各データ同士の内積を求めておくと:

$\braket{\xi_1, \xi_1} = 5$, $\braket{\xi_2, \xi_2} = 10$, $\braket{\xi_3, \xi_3} = 1$, $\braket{\xi_1, \xi_2} = 5$, $\braket{\xi_2, \xi_3} = -1$, $\braket{\xi_3, \xi_1} = 1$

である。よって、

$$
\begin{align*}
\tilde{L}(\lambda) &= \lambda_1 + \lambda_2 + \lambda_3 - \frac{1}{2}(-10 \lambda_1 \lambda_2 + 2 \lambda_2 \lambda_3 + 2 \lambda_3 \lambda_1 + 5 \lambda_1^2 + 10 \lambda_2^2 + \lambda_3^2) \\
&= \cdots \\
&= - \frac{13}{2} \left( \lambda_2 -\frac{2}{13} (1 + 3 \lambda_1) \right)^2 - \frac{8}{13} (\lambda_1 - \frac{3}{4})^2 + \frac{1}{2} \leq \frac{1}{2}
\end{align*}
$$

を得る。これを最大化する。考えられる最大値は $\frac{1}{2}$ である。

まず右辺第 2 項について $\lambda_1 = \frac{3}{4}$ が候補である。この時、右辺第 1 項を 0 と置くと、$\lambda_2 = \frac{1}{2}$ が求まる。ところが、$\lambda_3 = \lambda_2 - \lambda_1 = \frac{1}{2} - \frac{3}{4} = - \frac{1}{4} < 0$ となり**制約条件に反する**。よって、単純には最適解が求まらないことが分かる。

結局は、計算は少々面倒臭いが網掛けの部分で解を求める形になり、$- \frac{13}{2} \left( x -\frac{2}{13} (1 + 3 x) \right)^2 - \frac{8}{13} (x - \frac{3}{4})^2$ を最大化する形で $x = \frac{2}{5}$、**つまり、$\lambda_1 = \lambda_2 = \frac{2}{5}$, $\lambda_3 = 0$ が最適解**であることが分かる。

![](/images/dwd-optimization02/002.png)

この時、決定境界は (6) 式より

$$
\begin{align*}
y(\mathbf{x}) &= \frac{2}{5} \braket{\mathbf{x}, \xi_1} - \frac{2}{5} \braket{\mathbf{x}, \xi_2} + b \\
&= \frac{2}{5} (2 x^1 + x^2) - \frac{2}{5} (3 x^1 - x^2) + b \\
&= - \frac{2}{5} x^1 + \frac{4}{5} x^2 + b
\end{align*}
$$

となる。「超平面に最も近い $\xi_i$ については $|y(\xi_i)| = t_i (\mathbf{w}^T \xi_i + b) = 1$ と仮定」から $b$ を定めると、$b = 1$ となる。よって、最終的な決定境界は

$$
\begin{align*}
y(\mathbf{x}) &= \frac{2}{5} \braket{\mathbf{x}, \xi_1} - \frac{2}{5} \braket{\mathbf{x}, \xi_2} + b \\
&= \frac{2}{5} (2 x^1 + x^2) - \frac{2}{5} (3 x^1 - x^2) + b \\
&= - \frac{2}{5} x^1 + \frac{4}{5} x^2 + 1
\end{align*}
$$

である。$y(\mathbf{x}) = 0$ を冒頭の図に赤線で示している。

# Python でも実装してみる

折角なので、Python での実装も眺めてみる。scikit-learn を用いる。

```python
from sklearn.svm import LinearSVC

X = np.array([
    [2, 1],
    [3, -1],
    [0, 1],
])
y = np.array([1, -1, 1])

clf = LinearSVC(C=.1)
result = clf.fit(X, y)
```

のような形で良いと思われる。決定境界の可視化がよく分からなかったので、[[Python]サポートベクトルマシン(SVM)の理論と実装を徹底解説してみた](https://qiita.com/renesisu727/items/964005bd29aa680ad82d) を参考にさせていただいた。

```python
import mglearn

mglearn.plots.plot_2d_separator(result, X, fill=False)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')

plt.show()
```

ちょっと内部で使っている式が違いそうな気もして以下のような決定境界が描かれたが、本質的には今回計算した内容と大差はないと思う。そう信じたい[^4]。

![](/images/dwd-optimization02/003.png)

# まとめ

できるだけ簡単なデータセットに対して手計算で SVM の決定境界を求めてみた。こんな小さなデータセットに対しても計算はとても大変であり、とてもではないが手計算で現実的な問題を解けないことが分かる。よって、通常は今まで通り API で片付けることになるが、背景にはこういう考えがあることを知っておくのは悪くないと思う。

[^4]: 手計算なので、勿論計算を間違えている可能性は大いにある。

# 文献

- [PRML] C. M. Bishop, [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), pp.325-331, Springer, 2006
- [カーネル法] 福水健次, カーネル法入門, 朝倉書店, 2010
