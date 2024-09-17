---
title: "Stable Diffusion で遊んでみる (7) — FLUX.1 を試してみる"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "stablediffusion"]
published: false
---

# 目的

世間で [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) が流行っているらしいので、N 番煎じで動かして自己満足したまとめ。

# FLUX.1-dev-gguf

噂では巨大なモデルらしいので、恐る恐るということで量子化されたバージョンからいきたい。`gguf` ファイルは `diffusers` では直接ロードできず、[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) を経由する必要があるらしいのでこれで試す。

念のため VRAM 24 GB の NVIDIA L4 を使ったが、12359MiB くらいの消費量のようだったので、たぶんそこまで VRAM を使っていないのではないか？という気がする。

## OpenBLAS / cuBLAS の準備

```sh
$ ldconfig -p | grep openblas
```

で OpenBLAS が入っていなさそうなら

```sh
$ sudo apt-get install libopenblas-dev
```

でもして導入する。

```sh
$ cat /usr/local/cuda/include/cublas_api.h | grep CUBLAS_VER
```

で `cuBLAS` が入っているならそれでも良いらしい。

## stable-diffusion.cpp のビルド

以下でビルド。OpenBLAS を使うなら「-DSD_CUBLAS=ON」を「-DGGML_OPENBLAS=ON」で良いらしい[^1]。

[^1]: ビルドは 1 時間以上かかった気がするがあまりよく覚えていない。

```sh
$ sudo apt install ccache
$ git clone --recursive https://github.com/leejet/stable-diffusion.cpp
$ cd stable-diffusion.cpp
$ mkdir build
$ cd build
$ cmake .. -DSD_CUBLAS=ON
$ cmake --build . --config Release
```

> [  1%] Building C object thirdparty/CMakeFiles/zip.dir/zip.c.o
> In file included from /home/xxx/git_work/stable-diffusion.cpp/thirdparty/zip.c:40:
> /home/xxx/git_work/stable-diffusion.cpp/thirdparty/miniz.h:4988:9: note: ‘#pragma message: Using fopen, ftello, fseeko, stat() etc. path for file I/O - this path may not support large files.’
>  4988 | #pragma message(                                                               \
>       |         ^~~~~~~
> [  1%] Built target zip
> ...
> [ 98%] Building CXX object examples/cli/CMakeFiles/sd.dir/main.cpp.o
> [100%] Linking CXX executable ../../bin/sd
> [100%] Built target sd

## モデルのダウンロード

色々参考にしたので詳細は忘れたが以下をダウンロード。

```sh
$ mkdir models
$ cd models
$ curl -LO https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf
$ curl -LO https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors
$ curl -LO https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
$ curl -LO https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors
```

```sh
$ du -h
...
22G     .
```

結構なサイズ。

## 画像生成

以下のような感じで画像生成した。

```sh
./build/bin/sd --diffusion-model  models/flux1-dev-Q8_0.gguf \
  --vae ./models/ae.safetensors --clip_l ./models/clip_l.safetensors \
  --t5xxl ./models/t5xxl_fp16.safetensors \
  -p "A muscular macho male warrior wearing golden armor is holding a large sword. The sword is covered with flames. He is now fighting with a dragon." \
  --cfg-scale 1.0 --sampling-method euler -v
```

![](/images/dwd-stable-diffusion07/001.png =400x)

炎をまとった剣を素手で直接持っちゃダメですよ～ってのはあるけど、綺麗な画像が出来た。

# FLUX.1-dev

やはり `diffusers` 経由で使いたいのでもう少し調査。[sayakpaul/flux.1-dev-nf4-with-bnb-integration](https://huggingface.co/sayakpaul/flux.1-dev-nf4-with-bnb-integration) や [[Quantization] Add quantization support for `bitsandbytes` #9213](https://github.com/huggingface/diffusers/pull/9213) の内容が良さそうだったので、これを試した。正確には前者のページに以下のリンクがあって、この先の `sayakpaul/flux.1-dev-nf4-pkg` を試した。

> Check out [sayakpaul/flux.1-dev-nf4-pkg](https://huggingface.co/sayakpaul/flux.1-dev-nf4-pkg) that shows how to run this checkpoint along with an NF4 T5 in a free-tier Colab Notebook.

FLUX.1-dev を 16 GB の VRAM で動かすという内容だが、Colab 以外の 16 GB 環境ではあるオマジナイが必要だった。15098MiB くらい VRAM を使っていて、画像を出力する直前で 

> OutOfMemoryError: CUDA out of memory.

が発生したのだ。なお、必要なストレージサイズは 16 GB くらいだった。つまり、16 GB くらいのストレージと 16 GB くらいの VRAM が必要になる。

## 準備

以下を実行して、開発中の `diffusers` を導入する。

```sh
$ pip install bitsandbytes
$ pip install sentencepiece
$ pip install -U git+https://github.com/huggingface/diffusers@c795c82df39620e2576ccda765b6e67e849c36e7
```

## JupyterLab を起動

今回 JupyterLab を使ったが、T4 / A4000 のように VRAM が 16 GB の場合にはオマジナイをかけてから起動した。[Optimizing memory usage with `PYTORCH_CUDA_ALLOC_CONF`](https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) によると、VRAM の使い方を最適化する感じのオプションらしい。L4 のように VRAM が 24 GB くらい使えるならオマジナイなしでもいける。

```sh
$ export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$ jupyter lab
```

## Colab Notebook の通りに実行

[sayakpaul/flux.1-dev-nf4-pkg](https://huggingface.co/sayakpaul/flux.1-dev-nf4-pkg) からリンクされている Colab Notebook をそのままローカル実行するだけである。

```python
from transformers import T5EncoderModel
from diffusers import FluxTransformer2DModel, FluxPipeline
import torch
import gc


def flush():
    """Wipes off memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return f"{(bytes / 1024 / 1024 / 1024):.3f}"

flush()
```

そのまま続いて、

```python
%%time

nf4_model_id = "sayakpaul/flux.1-dev-nf4-pkg"
text_encoder_2 = T5EncoderModel.from_pretrained(
    nf4_model_id, subfolder="text_encoder_2", torch_dtype=torch.float16
)
transformer = FluxTransformer2DModel.from_pretrained(
    nf4_model_id, subfolder="transformer", torch_dtype=torch.float16
)
```

ここでダウンロードで 4～8 分くらいかかると思われる。トータル 13 GB くらいダウンロードが発生する。

```python
%%time

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_2,
    transformer=transformer,
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
```

ここはダウンロードに 2 分くらいかかる。

## 画像生成

```python
%%time

prompt = "An anime-style muscular macho male warrior wearing golden armor is holding a large sword. The sword is covered with flames. He is now fighting against a dragon."

image = pipe(
    prompt,
    guidance_scale=3.5,
    num_inference_steps=50,
    generator=torch.manual_seed(0)
).images[0]

torch.cuda.empty_cache()
memory = bytes_to_giga_bytes(torch.cuda.memory_allocated())
print(f"{memory=} GB.")

image.resize((image.size[0] // 3, image.size[1] // 3))
```

![](/images/dwd-stable-diffusion07/002.png =400x)

良い感じ。A4000 で 1min 40s くらいで、L4 で 1min 57s くらいで画像が生成された。

プロンプトを `prompt = "An anime-style winged goddess flies in the sky and watches over the earth."` にして、`seed` をランダムにして何度かガチャを引いたら以下のような画像が生成された。なかなかそれっぽい。

![](/images/dwd-stable-diffusion07/003.png =400x)

# まとめ

世間で流行っているらしい FLUX.1-dev を実行できた。相変わらず指が 6 本になったりすることはあるし、時々言うことを聞いてくれない・・・というか「う～ん」ってものが生成されることがあるのはご愛敬。
