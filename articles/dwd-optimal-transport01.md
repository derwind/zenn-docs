---
title: "最適輸送について考える (1) — 何も分からないところから始める"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "Python", "ポエム"]
published: false
---

# 目的

最適輸送が分からない。最適輸送距離が分からない。Earth Mover's Distance が分からない。Wasserstein 距離が分からない。分からない時は実験しながらそれっぽいものを探るのが良い気がするということで、試行錯誤してみる。

なお、厳密にはこれらは細かい区別があるようだが、今回は特に区別せずに混同して用いる。

# 事の起こり（ポエム）

特に何というわけではないが、たまに最適輸送距離というのを聞くので、適当に検索したら [最適輸送入門](https://speakerdeck.com/joisino/zui-shi-shu-song-ru-men) があって、3 周くらい読んだが何も分からなかった・・・。疑問は単純で、

- 片方の分布上の点を他方の分布上の点にマッピングするのってどうするの？前提知識として与えるわけではなさそうだが・・・

ということに尽きよう。ページをめくると、行列が出て来て、最適化問題が出て来て、エントロピーが出てきたので、無理になった🤯😵‍💫😵💫

参考文献として arXiv:1506.05439 [Learning with a Wasserstein Loss](https://arxiv.org/abs/1506.05439) があがっていたので見た。Eq. (2) がそれっぽい:

$$
\begin{align*}
W_c (\mu_1, \mu_2) = \inf_{\gamma \in \prod (\mu_1, \mu_2)} \int_{\mathcal{K} \times \mathcal{K}} c(\kappa_1, \kappa_2) \gamma (d \kappa_1, d \kappa_2)
\end{align*}
$$

この式は **最適輸送距離** と呼ばれるもので、

- $c: \mathcal{K} \times \mathcal{K} \to \R$ は**与えられた**コスト関数
- $\mu_1$ と $\mu_2$ は $\mathcal{K}$ 上の確率測度
- $\prod (\mu_1, \mu_2)$ は $\mathcal{K} \times \mathcal{K}$ 上の同時確率分布であって、$\mu_1$ と $\mu_2$ を周辺確率分布として持つようなものの集合

ということらしい。$c$ として重要なのは $\mathcal{K}$ 上の距離 $d_{\mathcal{K}}(\cdot, \cdot)$ の時などらしい。まったくピンと来ない。更に引用文献として、Cédric Villani の「Optimal transport, old and new」があがっているが、これはとても分厚いのでページ数だけで心が折れた。

引き続き検索すると Yossi Rubner et al. の [The Earth Mover's Distance as a Metric for Image Retrieval](https://link.springer.com/article/10.1023/A:1026543900054) が見つかった。p.8 の内容が恐らく該当するのだが、今度は **周辺分布の代わりに不等式制約が出てきた** のでますます分からなくなった。

続けて佐久田 祐子氏らの [Earth Mover’s Distanceを用いた画像の印象推定](https://www.jstage.jst.go.jp/article/jjske/19/1/19_TJSKE-D-19-00038/_article/-char/ja) を見つけた。日本語なので優しい気がする。「図3 EMD による色ヒストグラム間距離の計算」が何かそれっぽいイメージである。 **ソースの分布をより分けてターゲットの分布に持ち込むようだ。でもそのマップについて陽に書いた論文がなかったが？** というところで行き詰まった。

ということで完全にすっかり心が折れたのでもう忘れようと思ったが、今日は日曜日なので NumPy で適当にやってみることにした。

# 一番簡単なケース

[最適輸送入門](https://speakerdeck.com/joisino/zui-shi-shu-song-ru-men) の p.13 を再度見ると、同じ高さの棒グラフを平行移動している図がある。近い移動ほど最適輸送距離は短いことを言っていそうだ。そしてこの棒が 1 点に凝集したデルタ函数の場合にはサポートであるその 1 点同士の距離が最適輸送距離だ、ということのようだ。

そこで、まずは以下のように $\mathcal{K} = \{ 0, 1, 2, 3, 4 \}$ として、擬似的に $\delta_0 (x)$ と $\delta_3 (x)$ を用意して考えたい。たぶん最適輸送距離は $0$ から $3$ への移動である 3 となるのだろう。

![](/images/dwd-optimal-transport01/001.png)

そして結論としてある最適化を実行した結果、“最適輸送” を実現する計算が見つかって、近似的に求めた最適輸送距離は 2.764 となった。この距離を実現する計算式から再構成した周辺分布は以下であり、まぁまぁ元の $\mu_1$ と $\mu_2$ を維持している。

![](/images/dwd-optimal-transport01/002.png)

最適輸送を実現する “遷移” は近似的には以下のようになり、

$$
\begin{align*}
\hat{T} = \begin{bmatrix}
0.034 & 0 & 0 & 0 & 0 \\
0.009 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0.914 & 0 & 0.007 & 0.032 & 0.007 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
\end{align*}
$$

$\hat{T}$ の列ごとに行の要素を足し合わせると概ね $[1 \quad 0 \quad 0 \quad 0 \quad 0]$ となり、$\mu_1$ に対応する。次に、行ごとに列の要素を足し合わせると概ね $[0 \quad 0 \quad 0 \quad 1 \quad 0]$ となり $\mu_2$ に対応する。

つまり、列番号 0 にいた確率 1 が行番号 0, 1, 3 にそれぞれ 0.034, 0.009, 0.914 で分配されたことを意味している。他のものは誤差である。よって、このケースでは最適輸送距離は $0.034 \times |0 - 0| + 0.009 \times |1 - 0| + 0.914 \times |3 - 0| = 2.751$ で、誤差もすべて含めると 2.764 となる。

なお、真の遷移は

$$
\begin{align*}
T = \begin{bmatrix}
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
\end{align*}
$$

で、真の最適輸送距離は 3 である。

・・・ということが最終的に分かった。これはもう少し一般的な設定での実験結果で感触を確かめた後に得られた帰結であるが、この明快な結果故に実験は間違っていなかろうという自信につながった。

# 実験

上記を得た実験の流れを以下に記述していきたい。

まず必要なモジュールを import する:

```python
import numpy as np
import pprint
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
```

素朴な発想は以下であった:

```python
np.random.seed(seed=42)

N = 5  # 空間 K を離散的に考えた時の幅
joint_prob = np.random.rand(N * N).reshape(N, N)  # 何となく 2 次元のリスト
```

ここで、

- `np.sum(joint_prob, axis=0)` $\approx$ `mu1`
- `np.sum(joint_prob, axis=1)` $\approx$ `mu2`

が近いと良いと感じたが、未知数が $N^2$ 個に対し、方程式が $2N$ 個なので解が決定できない。それでも何か宜しく決めて欲しいとなると・・・ということで「最適化問題」の意味が分かってきた。

## 最適化して輸送するソースとターゲットの分布を決める

適当なデータを決めて問題設定をする。

```python
N = 5
X = np.arange(N, dtype=int)

np.random.seed(seed=42)

# 分布 mu1 と mu2 をランダムに定める。再現性のために seed は固定している。
mu1 = np.random.rand(N)
mu1 = mu1 / np.sum(mu1)

mu2 = np.random.rand(N)
mu2 = mu2 / np.sum(mu2)

# 可視化
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].stairs(mu1, fill=True)
axs[0].set_title(r"$\mu_1$")
axs[0].set_xlabel("x")
axs[0].set_ylabel(r"$\mu_1$")
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].yaxis.set_major_locator(MultipleLocator(0.1)) 
axs[0].set_ylim([0, 1])
axs[0].grid()

axs[1].stairs(mu2, fill=True)
axs[1].set_title(r"$\mu_2$")
axs[1].set_xlabel("x")
axs[1].set_ylabel(r"$\mu_2$")
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].yaxis.set_major_locator(MultipleLocator(0.1)) 
axs[1].set_ylim([0, 1])
axs[1].grid()

plt.tight_layout()
plt.show()
```

![](/images/dwd-optimal-transport01/003.png)

この時点ではよく分からないがこの 2 つ分布の最適な輸送・・・っぽいものを表現する行列 `joint_prob` を求めたい。

## PyTorch に乗せる

`joint_prob` を操作したいが、勾配を使わない最適化の類で値を決定するのは不安・・・と言うか今回の場合はよく分からないので、徐々に良くする雰囲気を重視して勾配による最適化を使う。そこで PyTorch に乗せることにした。

PyTorch を使うための追加のモジュールの import:

```python
import torch
from torch import nn
from torch import optim
```

以下、某 Quadratic Unconstrained Binary Optimization (QUBO) の気持ちで制約条件を書き出した。同時確率分布であることと、周辺分布が $\mu_1$ と $\mu_2$ になることを制約条件とした。

まずこの制約条件を満たすリスト `joint_prob` を訓練で求めるコードを書いた:

```python
# ランダムな 2 次元リストから始める
joint_prob = torch.rand(N * N).reshape(N, N)
# mu1 をテンソルにする
mu1_tensor = torch.tensor(mu1)
# mu2 をテンソルにする
mu2_tensor = torch.tensor(mu2)

# joint_prob を訓練する
params = nn.parameter.Parameter(joint_prob, requires_grad=True)
optimizer = optim.Adam([params])

for epoch in range(5000):
    optimizer.zero_grad()

    # joint_prob はそれ自身確率分布なので、すべてを足すと 1 である。
    prob_constraint = (torch.sum(params) - 1) ** 2
    # 行に関して足し合わせて得た周辺分布が mu1 になる。
    mu1_constraint = torch.sum((torch.sum(params, dim=0) - mu1_tensor) ** 2) / N
    # 列に関して足し合わせて得た周辺分布が mu2 になる。
    mu2_constraint = torch.sum((torch.sum(params, dim=1) - mu2_tensor) ** 2) / N
    constraint = prob_constraint + mu1_constraint + mu2_constraint
    loss = constraint
    loss.backward()
    optimizer.step()

    # 確率分布なので、値の範囲は [0, 1] であり、はみ出たものはクリップする。
    with torch.no_grad():
        params.clamp_(0, 1)
```

これはまぁまぁうまくいって、`params.cpu().detach().numpy()` から得る周辺分布は概ね `mu1` と `mu2` に一致する。

## 最適輸送距離を最適化で求める

最適化が完了した時点で「で、最適輸送距離とは？」と思ったので、上記の制約条件だけでなく、本来の損失関数が必要であることに気づいた。もう一度

$$
\begin{align*}
W_c (\mu_1, \mu_2) = \inf_{\gamma \in \prod (\mu_1, \mu_2)} \int_{\mathcal{K} \times \mathcal{K}} c(\kappa_1, \kappa_2) \gamma (d \kappa_1, d \kappa_2)
\end{align*}
$$

を思い出すと、$\gamma$ に対しては `joint_prob`、$\inf_{\gamma \in \prod (\mu_1, \mu_2)}$ に対しては上記の最適化ループの形で実装したことが分かる。よって残っているのは、$\int_{\mathcal{K} \times \mathcal{K}} c(\kappa_1, \kappa_2) \gamma (d \kappa_1, d \kappa_2)$ である。この部分は離散的に書くと以下のようなものであろう。$c$ という「与えられたコスト関数」は自分で決めるしかなさそうなので、適当に位置の差分の絶対値、つまり $L^1$ 距離とした。実装しながら

> $c$ として重要なのは $\mathcal{K}$ 上の距離 $d_{\mathcal{K}}(\cdot, \cdot)$ の時などらしい。

の意味が自己解決した。

```python
# この損失関数の値の最適値が最適輸送距離
EM_loss = 0
for col in range(N):
    for row in range(N):
        amount = params[row, col]  # γ(dκ₁, dκ₂)
        dist = abs(row - col)  # c(κ₁, κ₂); L1 distance
        EM_loss += dist * amount
```

これを損失関数として先ほどの制約条件と組み合わせて目的関数を作り、訓練ループを完全なものとしたのが以下である。

最適化を進めていくと、最初は制約条件を満たすように、つまり確率分布であることと周辺分布が指定のものに一致することを守ってくれたが、途中から制約条件を破ってでも損失関数の値を下げることで目的関数の全体値を下げるような動きをしだした。このため early stopping の考えを用いて訓練を打ち切れるようにした。

```python
joint_prob = torch.rand(N * N).reshape(N, N)
mu1_tensor = torch.tensor(mu1)
mu2_tensor = torch.tensor(mu2)

params = nn.parameter.Parameter(joint_prob, requires_grad=True)
optimizer = optim.Adam([params])

losses = []
EM_losses = []
constraints = []

best_constraint = 10000
best_constraint_epoch = -1
patience = 20  # early stopping 的な考え
patience_cnt = 0

for epoch in range(12000):
    optimizer.zero_grad()

    EM_loss = 0
    for col in range(N):
        for row in range(N):
            amount = params[row, col]
            dist = abs(row - col)
            EM_loss += amount * dist
    EM_loss = EM_loss / (N * N)

    prob_constraint = (torch.sum(params) - 1) ** 2
    mu1_constraint = torch.sum((torch.sum(params, dim=0) - mu1_tensor) ** 2) / N
    mu2_constraint = torch.sum((torch.sum(params, dim=1) - mu2_tensor) ** 2) / N
    constraint = prob_constraint + mu1_constraint + mu2_constraint
    loss = EM_loss + 4 * constraint  # 損失関数と制約条件を適当に重みづけて足す
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        params.clamp_(0, 1)

    # early stopping の考えを用いて、制約条件が破れだしたら訓練を終了する
    if constraint.item() < best_constraint:
        best_constraint = constraint.item()
        best_constraint_epoch = epoch
        patience_cnt = 0
    else:
        patience_cnt += 1
    if patience_cnt >= patience:
        break

    losses.append(loss.item())
    EM_losses.append(EM_loss.item())
    constraints.append(constraint.item())
```

これは約 20 秒で訓練が終了する。

# 実験結果の確認

まずは損失関数類の値の推移を可視化する。結構値が小さくなるので縦方向は対数をとった。

```python
fig, axs = plt.subplots(1, 3, figsize=(18, 4))
axs[0].set_yscale("log")
axs[0].plot(losses)
axs[0].set_title("losses")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("loss")
axs[0].grid()

axs[1].set_yscale("log")
axs[1].plot(EM_losses)
axs[1].set_title("EM_losses")
axs[1].set_xlabel("epoch")
axs[1].set_ylabel("EM_losse")
axs[1].grid()

axs[2].set_yscale("log")
axs[2].plot(constraints)
axs[2].set_title("constraints")
axs[2].set_xlabel("epoch")
axs[2].set_ylabel("constraint")
axs[2].grid()

plt.tight_layout()
plt.show()
```

![](/images/dwd-optimal-transport01/004.png)

全体の値が下がって行っているが、少しだけ早く制約条件項が収束しだしている。この後値が上昇し始めてしまったので、early stopping で打ち切った。

## 周辺分布の確認

最適化後の `joint_prob` は周辺分布 $\mu_1$ と $\mu_2$ をうまく表現できているであろうか？

```python
final_joint_prob = params.cpu().detach().numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
mu1_ = np.sum(final_joint_prob, axis=0)
axs[0].stairs(mu1_, fill=True)
axs[0].set_title(r"$\mu_1$")
axs[0].set_xlabel("x")
axs[0].set_ylabel(r"$\mu_1$")
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].yaxis.set_major_locator(MultipleLocator(0.1)) 
axs[0].set_ylim([0, 1])
axs[0].grid()

mu2_ = np.sum(final_joint_prob, axis=1)
axs[1].stairs(mu2_, fill=True)
axs[1].set_title(r"$\mu_2$")
axs[1].set_xlabel("x")
axs[1].set_ylabel(r"$\mu_2$")
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].yaxis.set_major_locator(MultipleLocator(0.1)) 
axs[1].set_ylim([0, 1])
axs[1].grid()

plt.tight_layout()
plt.show()
```

細かいところは多少のズレはあるが概ね似ているであろう。

![](/images/dwd-optimal-transport01/005.png)

[参考] 最初に設定した確率分布:

![](/images/dwd-optimal-transport01/003.png)

## 同時確率分布であること

周辺分布は良いとして、そもそも同時確率分布と見做せる状態なのだろうか？

```python
print(f"{np.sum(final_joint_prob)=}")
```

> np.sum(final_joint_prob)=0.9981522

すべてを足して 1 になっているので、ちゃんと確率分布として見做せそうである。

## 同時確率分布の中身は？

`pprint.pprint(final_joint_prob)` すれば良いのだが、少し整形してみよう:

$$
\begin{align*}
T = \begin{bmatrix}
0.092 & 0 & 0 & 0 & 0 \\
0 & 0.055 & 0 & 0 & 0 \\
0.018 & 0.157 & 0.152 & 0.026 & 0 \\
0 & 0.109 & 0.032 & 0.078 & 0.011 \\
0 & 0 & 0.080 & 0.121 & 0.066
\end{bmatrix}
\end{align*}
$$

列番号 0 にいた量 0.11 くらいのものが 0.092 が行番号 0 に、0.018 が列番号 2 に分散したらしい。ちょっと離れたところに移動したのは驚いた。但し基本的には、対角成分辺りに値が集中しているので、わりと近隣に値を輸送して調整しているケースが多そうである。列 3 に注目すると、0.225 くらい持っていた量のうち半分くらいの 0.121 を行 4 に移動させていることが分かる。$\mu_2$ の右端が結構高いのに対し、$\mu_1$ では右端は低いので、その差分をお隣からもらったということである。行 3 として足りなくなった分は列 1 から行 3 への輸送で大部分補ってもらったようである。

この遷移行列を見ているうちに、[Earth Mover’s Distanceを用いた画像の印象推定](https://www.jstage.jst.go.jp/article/jjske/19/1/19_TJSKE-D-19-00038/_article/-char/ja) の「図3 EMD による色ヒストグラム間距離の計算」のお気持ちが分かってきた。

## 最適輸送距離

肝心の最適輸送距離は幾らであろうか？

```python
print(f"EM distance={EM_losses[-1] * (N * N)}")
```

ということである。今回この値自体は大して意味がない。

> EM distance=0.7671486120671034

[The Earth Mover's Distance as a Metric for Image Retrieval](https://link.springer.com/article/10.1023/A:1026543900054) や arXiv:1701.07875 [Wasserstein GAN](https://arxiv.org/abs/1701.07875) ではこのメトリクスを更に最適化する形でニューラルネットワークを最適化するのであろう。但し、今回の実験で見たように、ナイーブに最適輸送距離を求めるのは結構計算コストが大きそうである。

なお、この実験コードを

```python
mu1 = np.zeros(N)
mu1[0] = 1

mu2 = np.zeros(N)
mu2[3] = 1
```

で行った結果が冒頭の「一番簡単なケース」である。

# まとめ

最適輸送や最適輸送距離についてさっぱり何も分からないところから始めて、雰囲気で実装していく中で何となく概要をつかむことができたように思う。与えられた「距離」函数 $c$ のもとで、分布 $\mu_1$ と $\mu_2$ を合わせこむために「量」を $\gamma (d \kappa_1, d \kappa_2)$ に従って移動するとして、どうすれば最適であるか？を問う問題であった。

少し状況を書き換えて、量 1 を動かすために必要な力が 1 とすると「力 x 距離 = 仕事」を最小化したものが最適輸送 “仕事” であり、荷物運びに必要な労力の最小値ということである。
