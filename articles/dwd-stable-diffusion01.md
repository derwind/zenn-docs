---
title: "Stable Diffusion で遊んでみる (1) — Stable Diffusion 3"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "stablediffusion"]
published: true
---

# 目的

どこかで Stable Diffusion 3 というキーワードを見たので動かしてみた。

# Stable Diffusion 3

[arXiv:2403.03206 Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) の内容らしい。論文はまだ読んでいない。

> In this work, we improve existing noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales.

辺りがウリなのだと思う。

# セットアップ

## モデルアクセスのための認証とアクセストークンの発行

まず、モデル [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) をダウンロードできるようにする必要がある。

すべて済んでダウンロード可能になった際には、`~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers` にモデルがダウンロードされるが、

```sh
$ du -h
...
15G     .
```

なので、ちょっと気持ち的には大きくてつらい。

さて、内容を読んで非商用である事を理解した上で個人情報的なものを登録して authentication の手続きが必要。終わったら、Hugging Face のポータルの右上のユーザーメニューから [Settings] - [Access Tokens] で適当にアクセストークンを発行する。

トークンをディスク内に永続化するために、

```sh
$ pip install 'huggingface_hub[cli,torch]'
```

して

```sh
$ huggingface-cli login
```

したら、アクセストークンが `~/.cache/huggingface/token` に書き出されるので、これで準備完了のはず。

## 必要なモジュールのセットアップ

標準的な PyTorch の環境に加え、

```sh
$ pip install -U diffusers transformers peft
```

くらいが必要そう。

- diffusers

```sh
$ pip list | grep diffusers
diffusers                 0.29.0
```

そもそも [v0.29.0: Stable Diffusion 3](https://github.com/huggingface/diffusers/releases/tag/v0.29.0) なので、バージョン >= 0.29.0 の必要がある。細かいことは [Add Stable Diffusion 3](https://github.com/huggingface/diffusers/pull/8483) を見れば良さそう。

- transformers

```sh
$ pip list | grep transformers
transformers              4.41.2
```

でバージョン > 4.37.2 であることを確認する。c.f. [data did not match any variant of untagged enum PyPreTokenizerTypeWrapper at line 960 column 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/discussions/18)

- peft

`peft` がないと、

> Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.

というメッセージがだらだら出てしまう。c.f. [how can we fix the message "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/discussions/13)

# 動かしてみる

よく見るチュートリアルと同様に以下くらいで絵が出る:

```python
from __future__ import annotations

from diffusers import DiffusionPipeline
import torch


pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16,
)
pipeline.to("cuda")

prompt = "XXXXXXXXXXXXXXXXX"
image = pipeline(prompt, num_inference_steps=25).images[0]
display(image)
```

`text_encoder_3=None` と `tokenizer_3=None` はよく分からないけど、これがないと

> OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB. GPU

が起きた。c.f. [How much VRAM ?](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/discussions/14)

上記の設定なら GTX 1080 (8 GB VRAM) でも動作するらしい。`nvidia-smi` で確認すると、使い方にもよるが、6～12GB くらいの VRAM 消費量で推移していた。

# 出力サンプル

適当なプロンプトで RPG の勇者っぽい画像を生成してみた。デフォルトでは 1024x1024 のサイズで出力されるらしい。こんな大きさで出せるって凄い。

![](/images/dwd-stable-diffusion01/001.png)

# まとめ

何も分からないけど、Stable Diffusion 3 が動いたらしい。
