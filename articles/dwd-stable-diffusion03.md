---
title: "Stable Diffusion で遊んでみる (3) — わざわざ Diffusers で VAE を実装する"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "stablediffusion", "poem"]
published: false
---

# 目的

`diffusers` を使ったコードを見ると、`AutoencoderKL` というものが使われていることがあって、arXiv:1312.6114 [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) とどういう関係にあるのかな？と思ったので少し調べた話。ほぼ備忘録。

# AutoencoderKL

`AutoencoderKL` は [autoencoder_kl.py](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/autoencoders/autoencoder_kl.py#L35) で定義されていて、[forward](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/autoencoders/autoencoder_kl.py#L435-L461) を見ると、

```python
    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
```

という実装である。冒頭で

```python
from .vae import Decoder, DecoderOutput, DiagonalGaussianDistribution, Encoder
```

をしていて、`AutoencoderKL.encode` がやっていることは、ほぼ `Encoder.forward` だし、`decode` も同様なので、結局は色々機能が追加された VAE（変分オートエンコーダ）ということかな？と思った。

# VAE を実装してみる

「鶏を割くに焉んぞ牛刀を用いん」な部分は否めないが、わざわざ `diffusers` を使って VAE を実装してみる。

必要なモジュールは全て `diffusers` の中に揃っているので、後は VAE の実装テンプレがあれば良い。何でも良いのだが、[ゼロから作る Deep Learning ❺](https://www.oreilly.co.jp//books/9784814400591/) 用のコードが GitHub の [vae.py](https://github.com/oreilly-japan/deep-learning-from-scratch-5/blob/main/step07/vae.py) にあって、道筋として単純だしこれに沿ってみることにした。

データセットは `QMNIST` を使ってみる。

必要なモジュールの import をする:

```python
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from diffusers.models.autoencoders.vae import (
    Decoder,
    DiagonalGaussianDistribution,
    Encoder
)
```

パラメータ類の設定:

```python
epochs = 30
learning_rate = 3e-4
batch_size = 32

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
```

データローダの作成:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
])
root = os.path.join(os.getenv("HOME"), ".torch")
dataset = datasets.QMNIST(root=root, train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)
```

VAE のクラスの実装:

[ゼロから作る Deep Learning ❺](https://www.oreilly.co.jp//books/9784814400591/) の GitHub のコードをベースに `AutoencoderKL` の実装を合わせこんでみた。

```python
class VAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        latent_channel = 4
        self.encoder = Encoder(in_channels=in_channels, out_channels=latent_channel)
        self.decoder = Decoder(in_channels=latent_channel, out_channels=out_channels)

    def get_loss(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()  # reparametrization trick
        x_hat = self.decoder(z)

        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = torch.sum(posterior.kl())
        return (L1 + L2) / batch_size
```

# 訓練

```python
%%time

model = VAE()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0
    for x, label in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_avg}")
```

> Epoch [1/30], Loss: 47.38421940511068
> Epoch [2/30], Loss: 42.57707533162435
> ...
> Epoch [29/30], Loss: 37.03056034037272
> Epoch [30/30], Loss: 37.003285912068684
> CPU times: user 32min 16s, sys: 3.79 s, total: 32min 20s
> Wall time: 32min 15s

潜在空間の次元が大きくて結構時間がかかってしまった。どれかの引数を指定して次元をもっと下げたら良いのかもしれないが、今回雑にやっているのでそこまでちゃんと調べていない。

# 可視化

```python
with torch.no_grad():
    sample_size = 64
    latent_channel = 4
    z = torch.randn(sample_size, 4*28*28).reshape(sample_size, latent_channel, 28, 28)
    z = z.to(device)
    x = model.decoder(z)
    x = x.detach().cpu()
    generated_images = x.view(sample_size, 1, 28, 28)

grid_img = torchvision.utils.make_grid(
    generated_images, nrow=8, padding=2, normalize=True
)

plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.show()
```

![](/images/dwd-stable-diffusion03/001.png)

手書き数字風味の画像がランダムノイズから生成された。

# まとめ

[AutoencoderKLOutput](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/modeling_outputs.py#L7C7-L17) を見ても潜在変数の入れ物に過ぎず、`AutoencoderKL` が名前ほどに KL 成分が陽に表に出ているかは疑問なのだが、内部でやっていることは概ね VAE に近いものだなというのが分かった（但し、バニラ VAE として使っているわけではないので、標準正規分布 $N(0, I)$ とのKL ダイバージェンスはとってなさそう）。

ついでに “内部でやっていることは概ね VAE に近い” であろうことを確認するために、内部実装の一部を書き換えて普通の変分オートエンコーダを実装してみて確認してみた。
