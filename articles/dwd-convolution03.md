---
title: "ニューラルネットの畳み込み層 (3)"
emoji: "⛓"
type: "tech"
topics: ["機械学習", "ポエム", "Python"]
published: true
---

# 目的

転置畳み込み [nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) を実行して理解を深める。`stride=1` のケースではただの畳み込みと同じ動作になり興味がないので、`stride > 1` のケースを見る。

また、API ドキュメントには

> `dilation` controls the spacing between the kernel points; also known as the à trous algorithm. It is harder to describe, but the link [here](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md) has a nice visualization of what `dilation` does.

と書かれている引数 `dilation` があるのだが、_It is harder to describe_ な部分を感覚的に見てみたい。

# 実験

## 入力画像

今回、転置畳み込みをストライド 2 で実行したい。アップスケーリングで画像サイズが縦横 2 倍になるので、事前に 1/2 にダウンスケールおよび二値化した画像を入力画像としたい。

![](/images/dwd-convolution03/001.png)
*半分のレナさん*

## 畳み込みカーネル

```python
conv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)

kernel = np.array([[
    # ラプラシアンフィルタ
    [
        [1.,  1., 1.],
        [1., -8., 1.],
        [1.,  1., 1.]
    ],
    # 恒等フィルタ
    [
        [0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.]
    ]
]])

conv.weight = Parameter(torch.from_numpy(kernel).float(), requires_grad=False)
```

というカーネルを使って順伝播してみよう。

```python
im_tensor = transforms.ToTensor()(im).unsqueeze(0)
convolved_im = conv.forward(im_tensor).cpu().squeeze(0)

im2 = transforms.ToPILImage()(convolved_im[0])
im3 = transforms.ToPILImage()(convolved_im[1])
```

# 結果

以下のようにアップスケーリングでサイズは 2 倍になるが、エッジ検出もできていないし、元のレナさんも出てきていない。

![](/images/dwd-convolution03/002.png)
*ラプラシアンフィルタ*

![](/images/dwd-convolution03/003.png)
*恒等フィルタ*

これについては [Transposed convolution animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#transposed-convolution-animations) の下段中央 `Padding, strides, transposed` を見ると良さそうだ。

# 考察

## ラプラシアンフィルタの場合

ゼロ埋めのせいで元々の白黒の境界が壊れていることが想像できると思われる。つまり、元々が白 (255) だった箇所に交互に黒 (0) が混じってきて細かい境界が沢山できている状態だ。
このため元々は白かった領域では

■■■
■□■
■■■

か

□■□
■■■
□■□

にカーネルを畳み込んでしまい、交互に白っぽいピクセルと黒っぽいピクセルが並んだざらざらした結果になる。わりと色味がはっきりとするのは、中央のピクセルが白の時 x8 で増強されるためと思われる。

## 恒等フィルタの場合

上記と似たようなものだが、上記のような 2 つのパターンで中央のピクセル値を出力画像に採用するので、交互に白っぽいピクセルと黒っぽいピクセルが並んだざらざらした結果になる。

# `dilation` を設定する

さて、最後に `dilation` を設定する。正確にはここが実は目玉商品だったりする。
[Dilated convolution animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations) を眺める。`stride=2` とすると、元画像に対してピクセルとピクセルの間にゼロ埋めがされるような形でアップスケーリングがされていた。一方、`dilation=2` は、カーネルの画像においてピクセルとピクセルの間にゼロ埋めがされるような形でカーネルがアップスケーリングされていると見ることはできないだろうか？

こう考えた場合、`stride=2` かつ `dilation=2` を同時に設定すれば、元画像もカーネルもゼロ埋めでアップスケーリングされてスケール感が合うのでは？と期待される。つまり、上記では破綻してしまったラプラシアンフィルタのエッジ検出が幾分機能するようになるのではないかと期待される。そこで、

```python
conv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False, dilation=2)
conv.weight = Parameter(torch.from_numpy(kernel).float(), requires_grad=False)
```

として順伝播すると以下を得る。

![](/images/dwd-convolution03/004.png)
*ラプラシアンフィルタ*

薄暗いがエッジ検出的な画像が出てきたことが見て取れる。何故薄暗くなるかというと、交互に現れる以下のピクセルパターンの上に dilated なカーネルを畳み込むと 0 になるためである。

■□■□■
■■■■■
■□■□■
■■■■■
■□■□■

よって、エッジ検出画像のピクセルとピクセルの間に黒いピクセルが入り込んで薄暗くなるような結果になる。

# まとめ

[前回](/articles/dwd-convolution02)の記事では

> - [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): ダウンスケール + カーネル畳み込みの演算
> - [nn.ConvTranspose2d](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html): アップスケール（ゼロ埋め） + カーネル畳み込みの演算

などと単純にまとめたが、実際には転置畳み込みは感覚的には遥かに難しい挙動になることが分かる。1 つには、パディングが周辺のコンテキストに応じた連続な補完**ではなく**常に 0 を埋めるためであり。このため、絶え間なく不連続なデータを生成した後でカーネルを畳み込むことになる。Keras の [Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) を見ても同様の仕様なのでとりあえずそういうものと思うしかない。

また、`dilation` 引数については

- カーネルのアップスケーリング

とでも思っておけば良さそうなことが分かった。

## 補足―GAN（敵対的生成ネットワーク）の実装から眺めてみる

転置畳み込みはややこしいので可能なら避けたい気持ちもあるが、有名な [U-Net](https://arxiv.org/abs/1505.04597) の “up-conv” 部分や或はその構造を生成器に適用した [Pix2pix](https://arxiv.org/abs/1611.07004) でも使われているので受け入れたほうが良いだろう。例えば Pix2pix の実装として [junyanz/pytorch-CycleGAN-and-pix2pix/models/networks.py#L518-L520](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1/models/networks.py#L518-L520) を見ると実際に `nn.ConvTranspose2d` が使われているわけである。他の同類のモデルも大体似たような状況である。
