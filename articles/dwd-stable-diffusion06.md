---
title: "Stable Diffusion で遊んでみる (6) — ネガティブプロンプトを試す（Reverse Activation 抑制）"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "stablediffusion"]
published: false
---

# 目的

ネガティブプロンプトについてもっと知りたい。arXiv:2406.02965 [Understanding the Impact of Negative Prompts: When and How Do They Take Effect?](https://arxiv.org/abs/2406.02965) によると、

3. ネガティブプロンプトの早期適用は逆に望まない生成 (“Reverse Activation”) の可能性

というのがあって、ネガティブプロンプトを最初から適用するのは良くなさそうという話があった。[Stable Diffusion で遊んでみる (5) — ネガティブプロンプトを試す（実装編）](/derwind/articles/dwd-stable-diffusion05) では保留にしたが、最小限の手間で何とかなるかも？と思ったので試してみた。

# 特異メソッド

Ruby に “特異メソッド” という手法があって、「黒魔術」ほどではないにせよ、インスタンスに動的にメソッドを追加するという荒業がある。これを Python で実行して、`__call__` と言うか `forward` の代わりに `forward_lazy_negative` とか言う今適当に名付けたメソッドを注入できたら良いのではないかと思った。

方法については適当に検索すると出てくるのだが `from types import MethodType` を使いなさいという内容のようだ。今回よく分からないなりに適当に試したらできた。

## `forward_lazy_negative` 追加

要するに、以下のようにすれば良い:

```python
from __future__ import annotations

...
from types import MethodType
...


pipe = StableDiffusionPipeline.from_single_file(
    model,
    torch_dtype=torch.float16
).to("cuda")

pipe.load_textual_inversion(
    "gsdf/Counterfeit-V3.0",
    weight_name="embedding/EasyNegativeV2.safetensors", 
    token="EasyNegativeV2"
)

# ★注入したいメソッドを定義する
@torch.no_grad()
def forward_lazy_negative(
    self,
    ...
    **kwargs,
):
    ...

# ★特異メソッドを注入
pipe.forward_lazy_negative = MethodType(forward_lazy_negative, pipe)

# ★特異メソッドを呼ぶ
result = pipe.forward_lazy_negative(
    prompt=prompt,
    ...
)

display(result.images[0])
```

# `forward_lazy_negative` の定義

`StableDiffusionPipeline.__call__` でネガティブプロンプトの投入を遅らせるための引数 `critical_step` を追加すれば良い。新規に追加はせずに `**kwargs` に潜り込ませる。

幾つか追加で型を見えるようにして、実行時例外が起こらないようにする。

内容的には汎用性は考えておらず、今回の引数の与え方で意図した動作になる程度の改修に留めた。

「★」を付けた箇所が `StableDiffusionPipeline.__call__` に対して変更または追加した部分である。

```python
from __future__ import annotations

import torch
from diffusers import StableDiffusionPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

@torch.no_grad()
def forward_lazy_negative(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    **kwargs,
):
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)
    critical_step = kwargs.pop("critical_step", 0)  # ★追加
    print(f"{critical_step=}")  # ★追加

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 0. Default height and width to unet
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    # to deal with lora scaling and other possible forward hooks

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Encode input prompt
    lora_scale = (
        self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
    )

    prompt_embeds_ = prompt_embeds  # ★追加
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # ★追加
    _, negative_prompt_embeds_uncond = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        self.do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=prompt_embeds_,
        negative_prompt_embeds=None,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if self.do_classifier_free_guidance:
        # ★追加/変更
        prompt_embeds_main = torch.cat([negative_prompt_embeds, prompt_embeds])  # ★名前変更
        prompt_embeds_uncond = torch.cat([negative_prompt_embeds_uncond, prompt_embeds])  # ★追加
        prompt_embeds = prompt_embeds_main  # ★追加

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    # 5. Prepare latent variables
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
        else None
    )

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            prompt_embeds = prompt_embeds_main  # ★追加
            if i < critical_step:  # ★追加
                prompt_embeds = prompt_embeds_uncond  # ★追加

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
```

# 実験

## (1a) 前回と同じ高解像度風味の画像

```python
generator = torch.Generator()
generator.manual_seed(1234)

result = pipe(
    prompt="girl eating pizza",
    negative_prompt="EasyNegativeV2, extra fingers, fewer fingers",
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
)
```

![](/images/dwd-stable-diffusion06/001.png =256x)

ちょっと顔色が悪い？気がしていた。

## (1b) 高解像度風味 + Reverse Activation 抑制

```python
result = pipe.forward_lazy_negative(
    prompt="girl eating pizza",
    negative_prompt="EasyNegativeV2, extra fingers, fewer fingers",
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
    critical_step=5,
)
```

![](/images/dwd-stable-diffusion06/002.png =256x)

ちょっと顔色良くなった？

## (2a) 前回と同じ眼鏡抑制

```python
result = pipe(
    prompt="girl eating pizza",
    negative_prompt="wearning glasses",
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
)
```

![](/images/dwd-stable-diffusion06/003.png =256x)

目つきがあやしい・・・。悩みを抱えている？

## (2b) 眼鏡抑制 + Reverse Activation 抑制

```python
result = pipe.forward_lazy_negative(
    prompt="girl eating pizza",
    negative_prompt="wearning glasses",
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
    critical_step=5,
)
```

![](/images/dwd-stable-diffusion06/004.png =256x)

目つきがそこそこ健康的になった気がする。

## (3a) 高解像度眼鏡

完全にオマケで、普通にうまく行っていたケースでも Reverse Activation 抑制の有無で比較してみたい。

```python
result = pipe(
    prompt="girl eating pizza wearning glasses",
    negative_prompt="EasyNegativeV2, extra fingers, fewer fingers",
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
)
```

![](/images/dwd-stable-diffusion06/005.png =256x)

わりと良い感じでは？でも、ピザがぐしゃっとなってて食べ方が汚い・・・。

## (3b) 高解像度眼鏡 + Reverse Activation 抑制

```python
result = pipe.forward_lazy_negative(
    prompt="girl eating pizza wearning glasses",
    negative_prompt="EasyNegativeV2, extra fingers, fewer fingers",
    width=512,
    height=512,
    num_inference_steps=50,
    generator=generator,
    critical_step=5,
)
```

![](/images/dwd-stable-diffusion06/006.png =256x)

眼鏡のフレームが一部欠損したような・・・。代わりにピザは綺麗になった食べ方が綺麗になった。

# まとめ

- Reverse Activation 抑制をお気持ち程度に軽くかけると、良くなることもあるし、それほどでもないこともあるし・・・という感じでよく分からない。
- Reverse Activation 抑制をかけたほうがピザの大きさやぐしゃぐしゃ具合がマシになる気もするが、よく分からない。
- そもそも実装的にこれで論文の主旨を反映できているのだろうか・・・。
