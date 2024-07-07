---
title: "Stable Diffusion で遊んでみる (5) — ネガティブプロンプトを試す（実装編）"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "stablediffusion"]
published: false
---

# 目的

ネガティブプロンプトについて知りたい。Stable Diffusion の中でどう使われて、普通の（ポジティブ）プロンプトとどう違うのかが知りたいというもの。

これについて [Stable Diffusion で遊んでみる (4) — ネガティブプロンプトを試す（なんちゃって理論編）](/derwind/articles/dwd-stable-diffusion04) で論文情報についてまとめたので、Diffuser を使った動きについて見たい。

**最終的には論文に対応する以下が実行されていることを見るのが目的**である:

$$
\begin{align*}
\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t) = (1 + w) \epsilon_\theta (\mathbf{x}_t, c (p_+), t) - w \epsilon_\theta (\mathbf{x}_t, c (p_-), t)
\end{align*}
$$

※ 今回、併せて Texual Inversion も眺めたため文章量が相当に多くなってしまった・・・。

# Counterfeit-V3.0 と EasyNegativeV2

[gsdf/Counterfeit-V3.0](https://huggingface.co/gsdf/Counterfeit-V3.0) が `EasyNegative` という “シェフのおススメ” のネガティブプロンプト集と一緒に公開されているモデルなので、これを使ってみたい。`model_index.json` が何故か置いていないので、使おうとすると一手間必要である。

## セットアップ

まず以下のように `.safetensors` をダウンロードする。`Counterfeit-V3.0_fix_fp16.safetensors` でも良さそうな気がするし、こちらのほうが小さいので好みで。

```sh
$ mkdir ~/.cache/huggingface/hub/models--gsdf--Counterfeit-V3.0
$ cd ~/.cache/huggingface/hub/models--gsdf--Counterfeit-V3.0
$ curl -LO https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fp16.safetensors
```

## 使い方

普通に使う。

```python
from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


hub_dir = Path(os.getenv("HOME"))/".cache/huggingface/hub"
model = str(hub_dir/"models--gsdf--Counterfeit-V3.0/Counterfeit-V3.0_fp16.safetensors")

pipe = StableDiffusionPipeline.from_single_file(
    model,
    torch_dtype=torch.float16
).to("cuda")

# EasyNegativeV2 を使いたい場合
pipe.load_textual_inversion(
    "gsdf/Counterfeit-V3.0",
    weight_name="embedding/EasyNegativeV2.safetensors", 
    token="EasyNegativeV2"
)

prompt = "girl eating pizza"
negative_prompt = None
#negative_prompt="EasyNegativeV2, extra fingers, fewer fingers"

#generator = None
# seed を固定したい場合
generator = torch.Generator()
generator.manual_seed(1234)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
)

display(result.images[0])
```

![](/images/dwd-stable-diffusion05/001.png =256x)

上記ではネガティブプロンプトを使っていないが、

```python
negative_prompt="EasyNegativeV2, extra fingers, fewer fingers"
```

を指定すると以下のようになる。何故ピザが大きくなるのかはさて置き、何だか高解像度感が出た。

![](/images/dwd-stable-diffusion05/002.png =256x)

このネガティブプロンプトは一体何を阻止しているのだろうか？ということで、ポジティブプロンプトに内容を入れてみよう。

```python
prompt = "girl eating pizza, EasyNegativeV2"
```

![](/images/dwd-stable-diffusion05/003.png =256x)

お察しください・・・という結果である。

# Texutual Inversion

なるほど、ネガティブプロンプトの効果は何となく分かったが、指定している「EasyNegativeV2」は一体全体どうして 3 つ目のような絵の生成を抑制できているのだろうか？我々は「EasyNegativeV2」という単語を見てそれをイメージできるのだろうか？そこで出て来るのが “Texutual Inversion”[^1] である。

[^1]: 通常はテキスト → 画像だと思うので、画像 → テキストの方向なので inversion というところだろうか。GAN inversion[^2] にヒントを得ているようだ。

[^2]: GAN インバージョン (GAN inversion) とは、事前に訓練された敵対的生成ネットワーク (GAN) の潜在空間に画像をマッピングするプロセスである。簡単に言うと、特定の実画像に対して、GANのジェネレーターを通して生成される画像が元の画像と非常に似ているような潜在ベクトル（GANの入力空間での表現）を見つけることを指す。(by ChatGPT-4o)

arXiv:2208.01618 [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618) で触れられているように、画像入力から概念のトークン $S_*$ を作るという技術がある。実際には、今回は阻止したい “何か” を埋め込んだテンソル (embedding) とトークンの対である既製の「EasyNegativeV2」を用いた形になる。

この技術を用いると、テキストによるプロンプトでこの概念に対応する embedding が呼び出せるようになる。この embedding とトークンの紐づけをロードするのが `DiffusionPipeline.load_textual_inversion` であり、Diffusers では `TextualInversionLoaderMixin` というミックスインの形で機能が提供されている。

## StableDiffusionPipeline.load_textual_inversion

`load_textual_inversion` で「EasyNegativeV2」を使えるようにしたのだが、該当するコードは [diffusers/loaders/textual_inversion.py#L267-L461](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/loaders/textual_inversion.py#L267-L461) である。

「# 7.4 Load token and embedding」の辺りが重要そうだ。

```python
    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        token: Optional[Union[str, List[str]]] = None,
        tokenizer: Optional["PreTrainedTokenizer"] = None,  # noqa: F821
        text_encoder: Optional["PreTrainedModel"] = None,  # noqa: F821
        **kwargs,
    ):
        # 1. Set correct tokenizer and text encoder
        tokenizer = tokenizer or getattr(self, "tokenizer", None)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        ...
        # 4. Load state dicts of textual embeddings
        state_dicts = load_textual_inversion_state_dicts(pretrained_model_name_or_paths, **kwargs)
        ...
        # 4. Retrieve tokens and embeddings
        tokens, embeddings = self._retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer)
        ...
        # 7.3 Increase token embedding matrix
        text_encoder.resize_token_embeddings(len(tokenizer) + len(tokens))
        input_embeddings = text_encoder.get_input_embeddings().weight
```

（以下の部分に注目したい）

```python
        # 7.4 Load token and embedding
        for token, embedding in zip(tokens, embeddings):
            # add tokens and get ids
            tokenizer.add_tokens(token)
            token_id = tokenizer.convert_tokens_to_ids(token)
            input_embeddings.data[token_id] = embedding
            logger.info(f"Loaded textual inversion embedding for {token}.")

        input_embeddings.to(dtype=dtype, device=device)
        ...
```

トークンをトークナイザに登録してトークン ID を発行させ、テキストエンコーダの管理する情報において、このトークン ID の場所に embedding を登録している。

## 実験

上記を踏まえると、以下のようにして既知のトークンの一覧を取得できる。

```python
pipe = StableDiffusionPipeline.from_pretrained(...)  # or .from_single_file

tokenizer = pipe.tokenizer
for token_id in range(50):
    token = tokenizer.convert_ids_to_tokens(token_id)
    print(f"{token_id} -> {token}")
```

> 0 -> !
> 1 -> "
> 2 -> #
> 3 -> $
> 4 -> %
> 5 -> &
> 6 -> '
> 7 -> (
> 8 -> )
> 9 -> *
> ...

で既知の ID をトークンに戻すと Unicode の Basic Latin の U+0021 の範囲がずらずら出て来る。

`embedding` のテンソルは以下のようなコードで取得できる。

```python
embedding = pipe.text_encoder.get_input_embeddings().weight.data[token_id]
```

## EasyNegativeV2

既にみたように EasyNegativeV2 の取り込みは以下のようにする。

```python
pipe.load_textual_inversion(
    "gsdf/Counterfeit-V3.0",
    weight_name="embedding/EasyNegativeV2.safetensors", 
    token="EasyNegativeV2"
)
```

これを実行すると、`pipe.text_encoder.get_input_embeddings().weight.data` のサイズが `[49408, 768]` から `[49424, 768]` に変化する。

```python
tokenizer = pipe.tokenizer

for token_id in range(49408, 49424):
    token = tokenizer.convert_ids_to_tokens(token_id)
    print(f"{token_id} -> {token}")
```

> 49408 -> EasyNegativeV2
> 49409 -> EasyNegativeV2_1
> ...
> 49423 -> EasyNegativeV2_15

が見える。よって、`EasyNegativeV2` は 1～15 の 15 個の概念の詰め合わせのように思われる。それぞれがどういう概念かは分からないが、既に見た絵で凡その察しはつく。

# ネガティブプロンプトの指定の考察

ネガティブプロンプトの指定について考察したい。

これも既に見たが、プロンプトとネガティブプロンプトは以下のように使う:

```python
prompt = "girl eating pizza"
negative_prompt="EasyNegativeV2, extra fingers, fewer fingers"

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=50,
)
```

このことから、`StableDiffusionPipeline.__call__` を見るのが良さそうだと分かる。[\_\_call\_\_](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L756-L1067) を見ると、

## StableDiffusionPipeline.\_\_call\_\_

```python
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        ...
        negative_prompt: Optional[Union[str, List[str]]] = None,
        ...
    ):
        ...
        # 3. Encode input prompt
        ...
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            ...
            negative_prompt,
            ...
        )
        ...
```

という感じになっている。次に [encode_prompt](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L302-L482) を見よう。

## StableDiffusionPipeline.encode_prompt

`prompt_embeds` と `negative_prompt_embeds` が明示的に与えられていない場合にはこれらを CLIP の `tokenizer` や `text_encoder` を用いて作り出すことが分かる。

```python
    def encode_prompt(
        self,
        prompt,
        ...
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ...
    ):
        ...
        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                ...
            )
            text_input_ids = text_inputs.input_ids
            ...
            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), ...)
                ...
            ...

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
            ...
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            ...
            uncond_input = self.tokenizer(
                uncond_tokens,
                ...
            )
            ...
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                ...
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            ...
        return prompt_embeds, negative_prompt_embeds
```

特にネガティブプロンプトに注目して追いかけると、どうやら、

- テンソル `prompt_embeds` を与えておらず、かつ `negative_prompt` も与えていない場合は “空文字列を使ったネガティブプロンプト” が生成され、
- 与えられている場合は `negative_prompt` が

`uncond_tokens` に格納されるようである。そしてこれらが最終的には `negative_prompt_embeds` に変換されて返される。
この辺は [Stable Diffusion で遊んでみる (4) — ネガティブプロンプトを試す（なんちゃって理論編）](/derwind/articles/dwd-stable-diffusion04) で見た、Eqs. (4) と (5) を比較すると、これらに対応していることが分かる。つまり、クラス識別の埋め込みに空を入れたら無条件、意味のあるものを入れたらネガティブプロンプトになるのである。

```python
    def encode_prompt(
        self,
        prompt,
        ...
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ...
    ):
        ...
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            ...
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            ...
            else:
                uncond_tokens = negative_prompt
            ...
            uncond_input = self.tokenizer(
                uncond_tokens,
                ...
            )
            ...
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                ...
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            ...
        return prompt_embeds, negative_prompt_embeds
```

[\_\_call\_\_](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L756-L1067) の続きを見ると、ネガティブプロンプトとプロンプトは結合されて、「ノイズを推定する U-Net」[^3]に入力されることが分かる。

[^3]: $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ に対して、$\epsilon$ を推定するネットワーク: $\epsilon_\theta (x_t, t)$。入出力が同じサイズのテンソルになるアーキテクチャである U-Net で実装されるのが主流のようである。

```python
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        ...
        negative_prompt: Optional[Union[str, List[str]]] = None,
        ...
    ):
        ...
        # 3. Encode input prompt
        ...
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            ...
            negative_prompt,
            ...
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        ...
        # 7. Denoising loop
        ...
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                ...
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    ...
                )[0]
        ...
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
```


$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ を近似するノイズ推定器が

$$
\begin{align*}
\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t) = (1 + w) \epsilon_\theta (\mathbf{x}_t, c (p_+), t) - w \epsilon_\theta (\mathbf{x}_t, c (p_-), t)
\end{align*}
$$

であったことを思い出すと、$\epsilon_\theta$ を右辺で 2 回使っているので、関数呼び出しとして 2 回の順伝播が発生しコストがかかる。これを 1 回で済ませるためにプロンプトたちを `torch.cat` しているのである。つまり、数式的には以下のようなことをしている:

$$
\begin{align*}
\epsilon_\theta \left(\mathbf{x}_t, \begin{pmatrix}c (p_-) \\ c (p_+)\end{pmatrix}, t \right)
\end{align*}
$$



`StableDiffusionPipeline` の場合、U-Net は `UNet2DConditionModel` であるので次にこれを見よう。

# UNet2DConditionModel 突入前後

[UNet2DConditionModel](https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/unets/unet_2d_condition.py) がプロンプトを処理するが、上記の `StableDiffusionPipeline.__call__` を見直すと、手前にネガティブプロンプト（無条件の場合は空文字列によるネガティブプロンプト）、後ろにポジティブプロンプトが連結された `prompt_embeds` が `encoder_hidden_states` に渡されている。つまり、`UNet2DConditionModel` としては自身の使われ方の詳細を知らない限りはプロンプトの区別がつかない。実際コードを見ると知らないようである。


よって、ネガティブプロンプトとポジティブプロンプトの事後処理をしているのは U-Net を抜けた以下の処理となる。`noise_pred_uncond` という変数で受けているが、実際にはこれが無条件 ($c = \empty$) またはネガティブプロンプトによるノイズであることになる。

```python
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
```

snippet の最後の線形結合は以下に対応する。なお `self.guidance_scale` が $w$ のことである。 

$$
\begin{align*}
\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t) = (1 + w) \epsilon_\theta (\mathbf{x}_t, c (p_+), t) - w \epsilon_\theta (\mathbf{x}_t, c (p_-), t)
\end{align*}
$$

残りの処理は「分類器なしガイダンス」の通常の処理となる。

これでこの記事の目的は達成された。

# Reverse Activation の確認

目的達成後のオマケであるが、arXiv:2406.02965 [Understanding the Impact of Negative Prompts: When and How Do They Take Effect?](https://arxiv.org/abs/2406.02965) の主張を思い出そう:

1. 遅延効果: ネガティブプロンプトは、ポジティブプロンプトが対応するコンテンツを表示した後、遅れて効果が観察される
1. 中和による削除: ポジティブプロンプトとネガティブプロンプトの潜在空間での相互キャンセルで生成された概念を打ち消す
1. ネガティブプロンプトの早期適用は逆に望まない生成 (“Reverse Activation”) の可能性

ネガティブプロンプトを使用する場合、逆に意図しない生成が生じる可能性が示唆されている。

論文でやたら眼鏡推しになっているので、今回もそれをネタとして試そう。

## 眼鏡をかけさせる

ポジティブプロンプトに「wearning glasses」を追加しよう:

```python
result = pipe(
    prompt="girl eating pizza wearning glasses",
    negative_prompt=None,
    ...
    num_inference_steps=50,
    ...
)
```

![](/images/dwd-stable-diffusion05/004.png =256x)

普通に意図通りである。

## 眼鏡をかけさせない

**ネガティブ**プロンプトに「wearning glasses」を指定しよう:

```python
result = pipe(
    prompt="girl eating pizza",
    negative_prompt="wearning glasses",
    ...
    num_inference_steps=50,
    ...
)
```

![](/images/dwd-stable-diffusion05/005.png =256x)

確かに論文が言っているような「画像の本来の構造を歪めてしまうという潜在的な危険性」的なものが目元に見られるような・・・。

## 眼鏡をかけさせないはずが・・・かけている！？

**ネガティブ**プロンプトに「wearning glasses」を指定しつつステップ数を減らしてみよう:

```python
result = pipe(
    prompt="girl eating pizza",
    negative_prompt="wearning glasses",
    ...
    num_inference_steps=30,
    ...
)
```

![](/images/dwd-stable-diffusion05/006.png =256x)

不完全ではあるが、本来は抑制すべき眼鏡をかけてしまったような絵になってしまった・・・。

# まとめ

- ネガティブプロンプトとは、以下のように従来は無条件のデノイジングであった部分にネガティブプロンプトの embedding を入力することで得られる「望まない画像の抑制」のテクニックであることが分かった。

$$
\begin{align*}
\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t) = (1 + w) \epsilon_\theta (\mathbf{x}_t, c (p_+), t) - w \epsilon_\theta (\mathbf{x}_t, c (p_-), t)
\end{align*}
$$

- 既製のネガティブプロンプト集を使う場合、“Textual Inversion” と呼ばれる技術が利用できることを見た。

- ネガティブプロンプトの雑な適用は、本来望まない画像を逆に生成してしまうという “Reverse Activation” を引き起こす可能性があることを見た。

“Reverse Activation” を回避するには `StableDiffusionPipeline` を改造して、ネガティブプロンプトの投入を「クリティカルステップ」以降まで遅らせる必要がありそうだが、それはまた別の機会に。
