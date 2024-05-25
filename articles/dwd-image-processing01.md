---
title: "画像処理について考える (1) — 高速Fourier変換"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "画像処理", "フーリエ変換"]
published: false
---

# 目的

画像処理の教科書を開くと、ローパスフィルタの説明などで、画像を Fourier 変換して高周波成分をカットするような話が出て来る。そして、Fourier 変換の可視化として以下のような絵が掲載されている。

![](/images/dwd-image-processing01/001.png)

このグランドクロスかグランドクルスかそういった光の十字架は何なのだろうか？ということについて軽く試してみたい。

# 高速 Fourier 変換

Wikipedia で [高速フーリエ変換](https://ja.wikipedia.org/wiki/%E9%AB%98%E9%80%9F%E3%83%95%E3%83%BC%E3%83%AA%E3%82%A8%E5%A4%89%E6%8F%9B) を見ると、

> 高速フーリエ変換（こうそくフーリエへんかん、英: fast Fourier transform, FFT）は、離散フーリエ変換（英: discrete Fourier transform, DFT）を計算機上で高速に計算するアルゴリズムである。

ということで、正体は離散 Fourier 変換である。以下では、高速なアルゴリズム部分を忘れてしまって、離散 Fourier 変換について復習しよう。

1 次元の連続函数 $f$ に対する一般的な Fourier 変換の定義は、Lebesgue 測度を $\mu$ として例えば以下のようになるであろうか:

$$
\begin{align*}
\hat{f}(\xi) = \int_{-\infty}^\infty f(x) \exp (-2\pi i \xi x) d \mu(x)
\end{align*}
$$

適当に $N \in \mathbb{N}$ をとって、

$$
\begin{align*}
F(\xi) := \hat{f}(\xi / N) = \int_{-\infty}^\infty f(x) \exp \left(-2\pi i \frac{\xi x}{N} \right) d \mu(x)
\end{align*}
$$

と置いてみよう。

次に $f$ が $\mathcal{D}_N := \{0, 1, \ldots, N-1 \}$ 上で離散的に定義された函数としよう。これに対応するように、更に測度を Lebesugue 測度から $\lambda(x) = \sum_{n=0}^{N-1} \delta_n (x)$ にしてみよう。ここで $\delta_n (x)$ は $x = n \in \mathbb{N}$ にピークを持つ Dirac 測度とする。測度は変えるが左辺の記号は使いまわすものとする:

$$
\begin{align*}
F(\xi) &= \int_{-\infty}^\infty f(x) \exp \left(-2\pi i \frac{\xi x}{N} \right) d \lambda(x) \\
&= \sum_{n=0}^{N-1} \int_{-\infty}^\infty f(x) \exp \left(-2\pi i \frac{\xi x}{N} \right) \delta_n (x) dx \\
&= \sum_{n=0}^{N-1} f(n) \exp \left(-2\pi i \frac{n \xi}{N} \right)
\tag{1}
\end{align*}
$$

このような導出が良いのか良く知らないが、Eq. (1) が **離散 Fourier 変換** である。一方、逆離散 Fourier 変換は

$$
\begin{align*}
f(x) &= \frac{1}{N} \sum_{n=0}^{N-1} F(n) \exp \left(2\pi i \frac{n x}{N} \right)
\end{align*}
$$

となる。

# 1 次元離散 Fourier 変換のナイーブな実装

後で使うので、以下を import しておく:

```python
import numpy as np
from PIL import Image
```

そして、1 次元の離散 Fourier 変換を定義からナイーブに実装する:

```python
def naive_dft(a: np.ndarray):
    N = a.shape[0]
    exps = np.exp(-2*np.pi*1j*np.arange(N)/N)
    return np.array([np.sum(a * exps ** m) for m in range(N)])


def naive_idft(a: np.ndarray):
    N = a.shape[0]
    exps = np.exp(2*np.pi*1j*np.arange(N)/N)
    return np.array([np.sum(a * exps ** m) for m in range(N)]) / N
```

適当にテストしてみよう:

```python
a = np.arange(10, dtype=complex)
print(np.allclose(naive_dft(a), np.fft.fft(a)))
print(np.allclose(naive_idft(a), np.fft.ifft(a)))
print(np.allclose(np.fft.ifft(np.fft.fft(a)), a))
print(np.allclose(naive_idft(naive_dft(a)), a))
```

> True
> True
> True
> True

たぶん良さそうだ。

# 2 次元離散 Fourier 変換のナイーブな実装

1 次元版と同じノリで考えると、通常の Fourier 変換の多次元版から考えて実装は以下だ。大分ナイーブすぎて効率は最悪そうだが、今回の目的はそこではない。

```python
def naive_dft2(a: np.ndarray):
    def coeff(a, m, n):
        h, w = a.shape
        exps = [np.exp(-2*np.pi*1j*p*m/h)*np.exp(-2*np.pi*1j*np.arange(w)*n/w)
                for p in range(h)]
        exps = np.array(exps).reshape(a.shape)
        return np.sum(a * exps)
    a_ = np.array([coeff(a, m, n)
                   for m in range(a.shape[0]) for n in range(a.shape[1])])
    return a_.reshape(a.shape)


def naive_idft2(a: np.ndarray):
    def coeff(a, m, n):
        h, w = a.shape
        exps = [np.exp(2*np.pi*1j*p*m/h)*np.exp(2*np.pi*1j*np.arange(w)*n/w)
                for p in range(h)]
        exps = np.array(exps).reshape(a.shape)
        return np.sum(a * exps)
    a_ = np.array([coeff(a, m, n)
                   for m in range(a.shape[0]) for n in range(a.shape[1])])
    return (a_ / np.prod(np.array(a.shape))).reshape(a.shape)
```

適当にテストしてみよう:

```python
b = np.arange(4*5, dtype=complex).reshape(4, 5)
print(np.allclose(naive_dft2(b), np.fft.fft2(b)))
print(np.allclose(naive_idft2(b), np.fft.ifft2(b)))
```

> True
> True

たぶん良さそうだ。

# 画像処理での応用

画像処理の本にある光の十字架が出て来るところでは `fftshift` という API が使われる。これはどうやら検索した限りでは、$\mathbb{Z}^2 / [0, 1]^2$（2 次元 torus）で視点を変えるような処理で、周波数空間の原点が画像の中央にくるような感じでスクロールするような API のようである。よって以下のように実装する:

```python
def naive_fftshift(fimg):
    return np.roll(fimg, [v // 2 for v in fimg.shape], [0, 1])
```

画像として何を使おうか考えたが、[浮世絵や日本画も膨大！シカゴ美術館が5万件超の所蔵作品を無料ダウンロード公開！商用利用OK](https://mag.japaaan.com/archives/84771) という記事があったので、[Under the Wave off Kanagawa (Kanagawa oki nami ura), also known as The Great Wave, from the series “Thirty-Six Views of Mount Fuji (Fugaku sanjūrokkei)”](https://www.artic.edu/artworks/24645/under-the-wave-off-kanagawa-kanagawa-oki-nami-ura-also-known-as-the-great-wave-from-the-series-thirty-six-views-of-mount-fuji-fugaku-sanj%E7%AC%9Brokkei) を使ってみることにした。

ナイーブな実装でも計算時間的に耐えるようにリサイズして、かつチャネル数を 1 のグレースケールに落としている。

さっそく計算してみよう:

![](/images/dwd-image-processing01/002.png)

```python
%%time

img = np.array(im)
fimg = naive_dft2(img)

mag = 15*np.log(np.abs(naive_fftshift(fimg)))
img2 = naive_idft2(fimg)
```

> CPU times: user 4min 9s, sys: 1.56 s, total: 4min 11s
> Wall time: 4min 12s

![](/images/dwd-image-processing01/003.png)

びっくりするほど遅いことに目を瞑ればよく見る結果が得られていることに気が付く。

光の十字架の正体は

1. 低周波成分が中央に来るようにスクロールして
2. 複素数値のデータなので絶対値をとって実数にして
3. 対数をとって適当にスケールしたもの

ということになる。

大部分が低周波成分なので、高周波成分をカットして逆離散 Fourier 変換したらローパスフィルタになるよ、というよくある話につながる。

# まとめ

内容的には画像処理の教科書の通りなので、特に新しい部分は何もないのだが、何も考えずに使っていた API の実装を自分で行うことで、中で何が行われているかがより明確になったと思う。
