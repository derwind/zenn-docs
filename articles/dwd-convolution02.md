---
title: "ニューラルネットの畳み込み層 (2)"
emoji: "⛓"
type: "tech"
topics: ["機械学習", "ポエム", "Python"]
published: true
---

# 目的

畳み込み層の理解を深めるために、少し数学的な視点で眺めたい。いまいち理解がこれで正しいのか分からないが、ポエムの 1 つとしてなら許容範囲ではないかと思う。要するにあまり厳密性には拘っていない。
またポエムなので厳密性は無視して、転置畳み込みまで踏み込んでみたい。

# 畳み込み

ここから関数 $f, g, h, \cdots$ や $k$ は台が有界であるとする。画像や畳み込み層のカーネルを想定しているので無限に広がっている必要がないからである。また、必要に応じて十分に滑らかであると仮定する。

カーネル $k$ をカーネルサイズ $2r+1$ とする。つまりよくあるカーネルサイズ $3$ の場合 $r=1$ という感じである。この時、$f: \mathbb{R}^2 \to [0, 255]$ を画像を適当に滑らかな関数で近似したものとする[^1]。畳み込みは以下で定義される:

[^1]: 適当に軟化子を適用したものでも考えることにすれば良いと思う。

$$
\begin{align*}
(k * f)(x) = \int_{|y-x| \leq r} \!\!\!\!\!\!\!\!\! k(x-y) f(y) dy
\tag{1}
\end{align*}
$$

## サンプル

[前回](/articles/dwd-convolution01)の書き方で、畳み込み層を以下のように実装すると畳み込みの出力は入力と一致する。

```python
conv = nn.Conv2d(1, 1, kernel_size = 3, stride=1, padding=1, bias=False)

kernel = np.array([[
    [
        [0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]
    ]
]])

conv.weight = Parameter(torch.from_numpy(kernel).float(), requires_grad=False)
```

これはカーネルが一種の（離散的な）デルタ関数になっており、畳み込み積分で $\delta * f = f$ となることに対応する。

## ストライド $n$ の場合

[Convolution animations](https://github.com/vdumoulin/conv_arithmetic#convolution-animations) を参考にさせていただくと、ストライドが $n \geq 1$ の時に畳み込み層の実装は以下のようになることが想像される:

$$
\begin{align*}
(K_n f)(x) = \int_{|y-nx| \leq r} \!\!\!\!\!\!\!\!\!\!\!\! k(nx-y) f(y) dy
\tag{1}
\end{align*}
$$

出力画像上のピクセル座標 $x$ に対し、入力画像においては座標 $nx$ に周辺の情報を畳み込んだものが得られるというニュアンスである。畳み込み層の適用を演算子 $K_n$ と表現した。

# 転置畳み込み

## Wikipedia や API ドキュメントを眺めてみる

ML のコンテキストで転置畳み込み、或は逆畳み込みと呼ばれる演算がある。ところで [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) の p.493 の欄外脚注を読むと

> This type of layer is sometimes referred to as a _deconvolution layer_, but it does _not_ perform what mathematicians call a deconvolution, so this name should be avoided

とある。実際、Wikipedia で[逆畳み込み](https://ja.wikipedia.org/wiki/%E9%80%86%E7%95%B3%E3%81%BF%E8%BE%BC%E3%81%BF)を読んでも、[nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) がやっていることとちょっと違うんじゃないかなと思う。雑な計算をしてみよう。Wikipedia に書いてあることと同じなのだが、$f * g = h$ から $f$ を解きたい場合、両辺をフーリエ変換して、$\hat{f} \hat{g} = \hat{h}$ となるが、更に逆フーリエ変換を使用して $f = h * \widecheck{1/\hat{g}}$ という形になる。一応は畳み込み的なものは出てきそうだが、ここから転置畳み込みで知られる処理に持ち込めるのか自信がない。よって今回は Wikipedia の逆畳み込みとは別の考えをしてみる。

## 共役演算子

唐突であるが、共役演算子なる概念を導入したい。関数 $f,g$ に対して

$$
\begin{align*}
\langle f, g \rangle = \int f(x) g(x) dx
\end{align*}
$$

という記号を定める[^2]。関数を関数にうつす演算子 $T$（例えば微分）に対する共役演算子 $T^\prime$ と呼ばれるものが $\langle Tf, g \rangle = \langle f, T^\prime g \rangle$ によって（少なくとも形式的には）定まる。例えば、$f,g$ を滑らかな実数値関数で遠方では $0$ になるようなものとすると、部分積分によって

[^2]: 実数値関数の空間の内積である。

$$
\begin{align*}
\int \frac{df}{dx}(x) g(x) dx = \int f(x) \left(- \frac{dg}{dx}(x)\right) dx
\end{align*}
$$

が得られる。この時、$T = \frac{d}{dx}$ とすると $T^\prime = - \frac{d}{dx}$ と考えられる。

この考えを使って転置畳み込みを考えられないか？というのがこのポエム記事のゴールである。

# $K_n$ の共役演算子

カーネル $k$ に対して $\tilde{k}(x) = k(-x)$ と定める。この時、積分の順序交換を適用して以下のような計算ができる:

$$
\begin{align*}
\langle K_n f, g \rangle &= \int \left( \int_{|y-nx| \leq r} \!\!\!\!\!\!\!\!\!\!\!\! k(nx-y)f(y) dy \right) g(x) dx \\
&= \int f(y) \left( \int_{|x-y| \leq r} \!\!\!\!\!\!\!\!\! \tilde{k}(y-x)g\left(\frac{x}{n}\right) \frac{dx}{n} \right) dy = \langle f,  K_n^\prime g \rangle
\end{align*}
$$

よって、記号を置き直して以下を得る。

$$
\begin{align*}
(K_n^\prime f)(x) = \frac{1}{n} \int_{|x-y| \leq r} \!\!\!\!\!\!\!\!\! \tilde{k}(x-y)g\left(\frac{y}{n}\right) dy
\tag{2}
\end{align*}
$$

個人的にはこの畳み込み積分のことを転置畳み込みと呼んでいるのではないかと思う。注意として $f$ の台が $|x| \leq R$ くらいの時、$K_n^\prime f$ の台は $|x| \leq nR$ くらいになっており、$n$ 倍にスケールされている、或は**アップスケールされている**と考えることができる。

## 画像処理的な視点で見ると？

例えば、出力画像におけるピクセル位置 $nx$ の値は、入力画像の位置 $x$ を中心として、ピクセルとピクセルの間の空間を含めて畳み込むような積分になっている。実際の転置畳み込みは [Transposed convolution animations](https://github.com/vdumoulin/conv_arithmetic#transposed-convolution-animations) を見ると分かる。入力画像が縦横に $n$ 倍にスケールされ、ピクセルとピクセルの間がゼロ埋めされた形[^3]になった後でカーネルを畳み込んでいる。その気持ちが (2) 式に現れているように思う。

[^3]: ゼロ埋め以外を試そうと思ったら API に怒られたので、PyTorch では現状はゼロ埋めしかできないらしい。

# 何故「転置」なのか？

最後に名称の由来について思いを馳せてみたい。

ちゃんと調べ切っていないので分からないが[^4]、以下のようなことを考えるとまぁまぁ納得できる気がした。$z, w \in \mathbb{R}^d$ を $d$ 次元実数ベクトル空間の元とする。$A \in \mathrm{M}_d(\mathbb{\R})$ を $d \times d$ 次正方行列とする時、今度は、$\langle \cdot, \cdot \rangle$ を内積にとってしまい、$\langle Az, w \rangle = \langle z, A^\prime w \rangle$ を考える時、$A^\prime$ は $A$ の転置行列になっている。“転置演算子” という表現は知らないのだがひょとしたら使われている分野もあるのかもしれない。

[^4]: 実際のところ思いつきを書いているだけなので、実質何も調べていない。根拠を調べてもいない殴り書きを備忘録的に記事にするのでポエムなのである。

# まとめ

本当はちゃんと調べたほうが良いのだが、調べる前に頭の中の雑な思考を書き出すのも良いかなと思って書き出してみた。間違ってたら分かった時に修正すれば良いと思う。とりあえず、結構それっぽいものが出てきたように思うので、転置畳み込みの理解が進んだように思う。標語的に書くと以下のように見ることもできるのではないだろうか？

- [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): ダウンスケール + カーネル畳み込みの演算
- [nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html): アップスケール（ゼロ埋め） + カーネル畳み込みの演算
