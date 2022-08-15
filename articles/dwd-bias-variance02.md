---
title: "バイアス-バリアンスについて考える (2)"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "ポエム"]
published: false
---

# 目的

[前回の記事](/derwind/dwd-bias-variance01)では、推定問題について神様の視点でモデルの評価を行なった。今回はもっと現実的に手に入るデータセットからモデルを作る場合の考察を行う。

# 今回の範囲

PRML 3.2. The Bias-Variance Decomposition (pp.147-149) を眺める。この内容は推定問題について現実的な視点で考えた時の話で、手に入る範囲のデータでモデルを作った時にどういった事が言えるか？を評価するものである。

# 課題

あるラベル付きのデータセットが与えられた時に、データからラベルを推定する最良のモデルを構築したい。この時、バイアスやバリアンスの視点でモデルを取り巻く状況を数理的にとらえたい。前提として、**データセットは限定的であり、手に入る範囲のもので何とかするしかない**ものとする。

# 前回のおさらい

[前回の記事](/derwind/dwd-bias-variance01) から (5) 式を再掲する:

$$
\begin{align*}
\mathbb{E}[L] = \int (y(\mathrm{x}) - h(\mathrm{x}))^2 p(\mathrm{x}) d\mathrm{x} + \int \operatorname{var}[t|\mathrm{x}] p(\mathrm{x}) d\mathrm{x}
\tag{1}
\end{align*}
$$

ここで $h(\mathrm{x}) = \mathbb{E}[t|\mathrm{x}]$ である。

前回見たように、右辺第 2 項は**データセットのノイズ**であるのでどうにもできない。このため、今回は右辺第 1 項について掘り下げる。

# 有限のデータセットで頑張る

残念ながら我々は有限のデータセットを幾つか用意するのが限界である。仮に $\mathcal{D}_1 = \{ (\mathrm{x_1^1}, t_1^1), \cdots, (\mathrm{x_1^N}, t_1^N) \}, \cdots, \mathcal{D}_M = \{ (\mathrm{x_M^1}, t_M^1), \cdots, (\mathrm{x_M^N}, t_M^N) \}$ という $N$ 個のデータとラベルからなるような $M$ 個のデータセットが得られているとする[^1]。我々には知るよしもないが、神様は知っている確率分布 $p(\mathrm{x},t)$ にこれらのデータセットは従っているものとする。
$\mathcal{D} = \{ \mathcal{D}_1, \cdots, \mathcal{D}_M \}$ と置く。

[^1]: 例えば、日本、アメリカ、フランス、ドイツの家賃事情といったところである。いわゆる「ミニバッチ」的なものを考えても良さそうに思う。

データセット $\mathcal{D}_i$ に注目しよう。$\mathcal{D}_i$ に基づいて、ラベルを推定するモデル $y(\mathrm{x};\mathcal{D}_i)$ を構築したとする。他のモデルについても同様のことを行い、$y(\mathrm{x};\mathcal{D}_1), \cdots, y(\mathrm{x};\mathcal{D}_M)$ を構築したとする。モデルの平均

$$
\begin{align*}
\mathbb{E}_\mathcal{D} [y(\mathrm{x};\mathcal{D})] = \frac{1}{M} \sum_{i=1}^M y(\mathrm{\mathrm{x};\mathcal{D}_i})
\tag{2}
\end{align*}
$$

を考える。これはこれで頑張れるだけ頑張った現実的に最良のモデルであろう。以下ではこのモデルを $\bar{y}(\mathrm{x})$ と書くことにする。

# 神様の視点で何が起きているかを見る

さて、ここで神様の気持ちになって $y(\mathrm{x};\mathcal{D}_i)$ と理想のモデル $h(\mathrm{x})$ との差を、我々の最良のモデル $\bar{y}(\mathrm{x})$ も踏まえて考えてみよう。

$$
\begin{align*}
(y(\mathcal{x};\mathcal{D}_i) - h(\mathrm{x}))^2 &= (\bar{y}(\mathrm{x}) - h(\mathrm{x}))^2 + (y(\mathcal{x};\mathcal{D}_i) - \bar{y}(\mathrm{x}))^2 \\
&+ 2 (y(\mathcal{x};\mathcal{D}_i) - \bar{y}(\mathrm{x})) (\bar{y}(\mathrm{x}) - h(\mathrm{x}))
\tag{3}
\end{align*}
$$

次に手持ちのデータセットすべてを使って期待値 $\mathbb{E}_\mathcal{D}[\cdots] = \frac{1}{M} \sum_{i=1}^M \cdots$ を考える。ところで (3) 式の右辺第 3 項は、$\mathbb{E}_\mathcal{D}[\cdots]$ をとると消える[^2]。

[^2]:$\mathbb{E}_\mathcal{D} [y(\mathrm{x};\mathcal{D})]$ は最早特定のデータセットには依存しない定数であることに注意すると (2) 式より、$$\frac{1}{M} \sum_{i=1}^M (y(\mathcal{x};\mathcal{D}_i) - \bar{y}(\mathrm{x})) = \bar{y}(\mathrm{x}) - \bar{y}(\mathrm{x}) = 0$$ となる。

従って、(3) 式をデータセットにわたって平均をとると、

$$
\begin{align*}
\frac{1}{M} \sum_{i=1}^M (y(\mathcal{x};\mathcal{D}_i) - h(\mathrm{x}))^2 &= \frac{1}{M} \sum_{i=1}^M (\bar{y}(\mathrm{x}) - h(\mathrm{x}))^2 + \frac{1}{M} \sum_{i=1}^M (y(\mathcal{x};\mathcal{D}_i) - \bar{y}(\mathrm{x}))^2 \\
&= (\bar{y}(\mathrm{x}) - h(\mathrm{x}))^2 + \frac{1}{M} \sum_{i=1}^M (y(\mathcal{x};\mathcal{D}_i) - \bar{y}(\mathrm{x}))^2
\tag{4}
\end{align*}
$$

を得る。或いは少し記号が混乱するが、

$$
\begin{align*}
\mathbb{E}_\mathcal{D}[(y(\mathcal{x};\mathcal{D}) - h(\mathrm{x}))^2] = (\bar{y}(\mathrm{x}) - h(\mathrm{x}))^2 + \mathbb{E}_\mathcal{D}[(y(\mathcal{x};\mathcal{D}) - \bar{y}(\mathrm{x}))^2]
\tag{4'}
\end{align*}
$$

を得る。

最後に、(4') 式について $\int \cdots p(\mathrm{x}) d\mathrm{x}$ をとる。積分の順序交換 $\int \mathbb{E}_\mathcal{D} [\cdots] p(\mathrm{x}) d\mathrm{x} = \mathbb{E}_\mathcal{D} [\int \cdots p(\mathrm{x}) d\mathrm{x}]$ に注意して、

$$
\begin{align*}
\mathbb{E}_\mathcal{D}\left[ \int (y(\mathcal{x};\mathcal{D}) - h(\mathrm{x}))^2 p(\mathrm{x}) d\mathrm{x} \right] &= \int \mathbb{E}_\mathcal{D}[(y(\mathcal{x};\mathcal{D}) - h(\mathrm{x}))^2] p(\mathrm{x}) d\mathrm{x} \\
&= \int (\bar{y}(\mathrm{x}) - h(\mathrm{x}))^2 p(\mathrm{x}) d\mathrm{x} \\
&+ \int \mathbb{E}_\mathcal{D}[(y(\mathcal{x};\mathcal{D}) - \bar{y}(\mathrm{x}))^2] p(\mathrm{x}) d\mathrm{x}
\tag{5}
\end{align*}
$$

を得る。(1) 式を $y(\mathcal{x};\mathcal{D})$ について書き直すと、データセット $\mathcal{D}$ に対するものということを明記して、

$$
\begin{align*}
\mathbb{E}[L;\mathcal{D}] = \int (y(\mathcal{x};\mathcal{D}) - h(\mathrm{x}))^2 p(\mathrm{x}) d\mathrm{x} + \int \operatorname{var}[t|\mathrm{x}] p(\mathrm{x}) d\mathrm{x}
\tag{1'}
\end{align*}
$$

となる。これの $\mathbb{E}_\mathcal{D}[\cdots]$ をとって (5) 式と組み合わせると、

$$
\begin{align*}
\mathbb{E}_\mathcal{D} [ \mathbb{E}[L;\mathcal{D}] ] &= \int (\bar{y}(\mathrm{x}) - h(\mathrm{x}))^2 p(\mathrm{x}) d\mathrm{x} + \int \mathbb{E}_\mathcal{D}[(y(\mathcal{x};\mathcal{D}) - \bar{y}(\mathrm{x}))^2] p(\mathrm{x}) d\mathrm{x} \\
&+ \int \operatorname{var}[t|\mathrm{x}] p(\mathrm{x}) d\mathrm{x}
\tag{6}
\end{align*}
$$

となる。PRML では右辺第 1 項を (bias)$^2$、第 2 項を variance、第 3 項をノイズと呼んでいる。

既に見たように、第 3 項のノイズはデータセットに起因するので手を入れようがない。

第 2 項は各データセットでのモデルが**平均的なモデル**に近づけば小さくなる。全部同じようなぼんやりとした推定で微妙に的外れで、日本の家賃事情とアメリカの家賃事情を細かくは反映できていない可能性がある。これが**アンダーフィッティング**状態ということになる。[^3]

[^3]: どの国において、どういう賃貸条件を入力してもなんとなく「5 万円くらい」と推定されるような個性のない状態。全モデルがそういう状態であれば、平均化されたモデルも「5 万円くらい」と返すモデルであり、モデル間のバラつきはないがそもそも何も説明できておらず役に立たない。

第 1 項は平均したモデルが手持ちのデータセットすべてである $\mathcal{D}$ の上では**神様の知っている大正解** $h(\cdot)$ に近づくと小さくなる。反面、個々のデータセットでの個別のモデルを蔑ろにしている可能性があり、まったく新しいデータセット $\mathcal{D}_{M+1}$ を持ってきてその上で推定させると当たらずとも遠からずくらいの良くも悪くもない程度の汎化性能の可能性がある。これが**オーバーフィッティング**状態ということになる。[^4]

[^4]: 日本、アメリカ、フランス、ドイツの家賃事情はうまく説明できるようになったが、そのモデルでオーストラリアでの家賃事情を説明させようとしたみたら、微妙にズレた感じになる・・・みたいなことを指す。

$\mathcal{D}$ だけでは偶然にも $\bar{y}(\mathrm{x})$ が未知のデータセットを含めて $h(\mathrm{x})$ に近づくことは期待できないので、現実的にはバイアスとバリアンスのバランスをとりつつ第 1 項と第 2 項が共に小さそうなところで訓練を止める必要がある。つまり個性は大事にしたいが、あまり尖りすぎているのも困るというトレードオフ状態である。

# まとめ

というところまでが教科書的な部分である。とは言え、絶対にいつもトレードオフ状態になることを数式は示唆しておらず、たまたま「$\mathcal{D}$ だけでは偶然にも $\bar{y}(\mathrm{x})$ が未知のデータセットを含めて $h(\mathrm{x})$ に近づく」ということが起きているのか或は「$\mathcal{D}$ は非常に良くありとあらゆるデータセットを代表するような要素から構成されていた」のか、`train_loss` が下がりながら `val_loss` も下がっていくという訓練結果になることもある。そういう時は有難い気持ちで訓練を終了すれば良い。

Kaggle の [Theoretical ML Interview Question: Bias-Variance Tradeoff](https://www.kaggle.com/general/198890) のようなことが起きた場合には、(6) 式を思い出したら良いことになる。が、やはり (6) 式はどういう訓練をしたら良いかを示唆するものでもないので、結局は ML におけるベストプラクティスというか、**経験的にうまくいくとされる方法**で対策することになる。

この範囲の学びはよく分からないけど、「**バイアスもバリアンスも低そうなところが数式的には良さそうではあるので、訓練でそういうポイントに入ったら実際に推論して使い物になるか試してみよう。どうやってもダメな場合、データセットのノイズが大きいのかもしれないし、うまくクレンジングしてノイズの少なそうなデータセットを準備してリトライしてみよう**」みたいな感じで良いだろうか？
