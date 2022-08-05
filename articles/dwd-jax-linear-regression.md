---
title: "JAX で遊んでみる (1) — 線形回帰"
emoji: "⛓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "ポエム", "Python"]
published: true
---

# 目的

[JAX](https://jax.readthedocs.io/en/latest/) をインストールして少し触ってみたという記録。大体 [Linear Regression with JAX](https://coax.readthedocs.io/en/latest/examples/linear_regression/jax.html) に書いているのと同じような内容で、普通に線形回帰をしましたという備忘録。

これだけだと記事がすぐに終わってしまうので、統計学による直接計算や scikit-learn の使用例も交えて水増ししてみた。

# おさらい

ディープラーニングは画像分類、画像生成、画像認識や物体検出、自然言語処理など色々ジャンルはあると思うが、基本的には何かしら入力データの空間 $\mathcal{X} = \R^n$ から出力データの空間 $\mathcal{Y} = \R^m$ への可微分写像 $f$ を求める問題に帰着されると思われる。例えばの例としては、以下のようなものがあろう:

- $\mathcal{Y} = \R^1$ は物件の家賃であり、$\mathcal{X} = \R^3$ は駅からの距離、築年数、部屋が角部屋か否かという情報からなる。
- $\mathcal{Y} = \R^{64\times 64}$ は人物の顔写真であり、$\mathcal{X} = \R^{128}$ は正規分布に従うランダムノイズからなる。
- $\mathcal{Y} = \R^{3\times 128}$ は 3 つの日本語の単語からなる集合で、個々の単語は $128$ 次元のベクトルで符号化されている。$\mathcal{X} = \R^{3\times 128}$ は 3 つの英語の単語からなる集合で、個々の単語は $128$ 次元のベクトルで符号化されている。

色々ある問題の中で、特に教師あり学習と呼ばれるものは、例 $\{(x_i, y_i)\} \subset \mathcal{X} \times \mathcal{Y}$ が与えられた時に、次のような可微分写像を求める問題である:

1. 既知のデータ $\hat{y_i} = f(x_i)$ に対して $\hat{y}_i \approx y_i$ が成り立つ。
1. 未知のデータ $x \not\in \{y_i\}$ に対して $f(x) \in \mathcal{Y}$ は何らかの意味でもっともらしい。

# 今回は何をする？
大袈裟なことは避け、$\mathcal{X} =\R^1$ および $\mathcal{Y} = \R^1$ として、摂氏と華氏の変換公式を JAX に求めさせる。教師データとしては、「摂氏を測れる温度計および華氏を測れる温度計を与えられた人物たちが、あるタイミングで計測を依頼されたことによって得られる、摂氏-華氏のペアのデータ」とする。目視での計測のためある程度のゆらぎがあるものとする。

# データ作成

```python
import numpy as np
import random
import matplotlib.pyplot as plt

xs = np.arange(-5, 15, 0.05)
ys = np.array([x*9/5+32 + random.gauss(0,3) for x in xs])

ys_ideal = np.array([x*9/5+32 for x in xs])
plt.scatter(xs,ys)
plt.plot(xs,ys_ideal, color='red')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.show()
```

![](/images/dwd-jax-linear-regression/001.png)

を今回の教師データとする。なお、よく知られているように、摂氏 $C$ 度に対応する華氏 $F$ 度は

$$
F = \frac{9}{5} C + 32
$$

で与えられ、上図で赤線で引いたものがこの直線に対応する。

# 統計学で解いてみる

摂氏-華氏のデータセットを $\{x_i, y_i\}_{1 \leq i \leq N}$ とする。この時線形回帰 $\hat{y}_i = \alpha x_i + \beta$ によって

$$
\begin{align*}
\argmax\limits_{\alpha,\beta} \sum_{j=1}^N (\hat{y}_j - y_j)^2
\tag{1}
\end{align*}
$$

で係数 $\alpha$ と $\beta$ を求めたい。データセットの標本平均を $\bar{x} = \frac{1}{N}\sum x_i$ および $\bar{y} = \frac{1}{N}\sum y_i$ と置いて、(1) 式を $\alpha$ と $\beta$ についてそれぞれ偏微分して 0 とすることで、

$$
\begin{align*}
\alpha &= \frac{\sum (x_i -\bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} \\
\beta &= \bar{y} - \alpha \bar{x}
\end{align*}
$$

が解として求まる。念の為に Python で解くと

```python
xs_mean = np.mean(xs)
ys_mean = np.mean(ys)
alpha = np.sum((xs - xs_mean)*(ys - ys_mean))/np.sum((xs - xs_mean)**2)
beta = ys_mean - alpha * xs_mean

print('estimate:', alpha, beta)
print('ideal:', 9/5, 32)
```

estimate: 1.8103115195284565 32.08597124079442
ideal: 1.8 32

という結果であった。

# scikit-learn でも解いてみる

正直、今回程度の問題なら　scikit-learn で解くのがベストだと思う。参考までに解いてみよう。

```python
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(xs.reshape(-1,1), ys.reshape(-1,1))
coef, intercept = model_lr.coef_[0][0], model_lr.intercept_[0]

print('estimate:', coef, intercept)
print('ideal:', 9/5, 32)
```

estimate: 1.8103115195284571 32.08597124079442
ideal: 1.8 32

簡単であるし、何ら問題はない。

# JAX で解いてみる

漸くメインである。そして結果は分かっているのでまったく盛り上がらないが仕方ない。
ところでこの手のフレームワークはデータを正規化しないとうまく結果が得られないことが常なので、標準的な正規化を行いたい。

$\sigma_x$ と $\sigma_y$ をそれぞれ $\{x_j\}$ と $\{y_j\}$ の標準偏差とする[^1]。正規化されたデータに対する線形回帰問題を

[^1]: 統計学では不偏分散の平方根で不偏標準偏差を求めると思うが、ディープラーニングのコンテキストではそこまでしていないように見えるので、普通に標本分散の平方根による標準偏差を用いる。

$$
\begin{align*}
\frac{y_j - \bar{y}}{\sigma_y} = \tilde{\alpha} \frac{x_j - \bar{x}}{\sigma_x} + \tilde{\beta}
\tag{2}
\end{align*}
$$

とする。良い係数 $\tilde{\alpha}$ と $\tilde{\beta}$ が求まった時、(1) の問題の $\alpha$ と $\beta$ に関連づけて言うと、

$$
\begin{align*}
\alpha &= \tilde{\alpha} \frac{\sigma_y}{\sigma_x} \\
\beta &= \bar{y} + \tilde{\beta} \sigma_y - \bar{x} \frac{\sigma_y}{\sigma_x}
\tag{3}
\end{align*}
$$

という対応になっている。

まずはデータの正規化を Python で実装しよう:

```python
from jax import grad
import jax.numpy as jnp

xs_std = np.std(xs)
ys_std = np.std(ys)

xs_n = (xs - xs_mean) / xs_std
ys_n = (ys - ys_mean) / ys_std
```

次に線形回帰モデルを実装する:

```python
def model(params, x):
    W, b = params
    return x * W + b

def loss(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y)**2)

def update(params, x, y, lr=0.1):
    return params - lr * grad(loss)(params, x, y)
```

ここまでできると、後は訓練ループを回すだけである。今回は何も考えずに 5000 回イテレーションを回す。回した後に得られた “最適値” を (3) 式に基づいて “元に戻す”:

```python
params = jnp.array([0., 0.])

for _ in range(5000):
    params = update(params, xs_n, ys_n)

a, b = params
a = a * ys_std / xs_std
b = ys_mean + b * ys_std - xs_mean * ys_std / xs_std
```

一応、結果を表示すると、

```
print('estimate:', a, b)
print('ideal:', 9/5, 32)
```

estimate: 1.8103114 31.748932
ideal: 1.8 32

という感じになる。記念にプロットもしておこう:

```python
plt.scatter(xs,ys)
params = jnp.array([a, b])
plt.plot(xs,model(params,xs), color='red')
plt.xlabel('Celsius')
plt.ylabel('Fahrenheit')
plt.show()
```

![](/images/dwd-jax-linear-regression/002.png)

めでたく期待通りの結果が得られた。これでモデルは摂氏と華氏の変換の知識を獲得したことになる。

# まとめ

特にとりたてて書くほどのまとめもないが、TensorFlow や PyTorch に比べて質素に書けたように思う。データローダの準備だとかテンソルを GPU に乗せるといったことを意識せずに NumPy のように雑に書いて、Python 的な書き方で訓練ループも回せた。`loss` を `grad` で包むだけで自動微分が実行されるのも楽で良い。
