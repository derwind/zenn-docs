---
title: "Stable Diffusion で遊んでみる (4) — ネガティブプロンプトを試す（なんちゃって理論編）"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "stablediffusion"]
published: false
---

# 目的

ネガティブプロンプトについて知りたい。Stable Diffusion の中でどう使われて、普通の（ポジティブ）プロンプトとどう違うのかが知りたいというもの。

ここでは、論文ベースの情報を “なんちゃって” で集約する。以下、論文ごとの記号類を統一するのは手間なので、扱っている論文ごとにその論文での記号をそのまま流用する。よって、この記事内では同じものを指す記号が結構頻繁にマイナーチェンジされる。式番号についても、扱っている論文における式番号をそのまま流用する。

# ノイズの推定器

arXiv:2006.11239 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM) では、「3.4 Simplified training objective」で触れられているように、$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ に対して、$\mathbf{x}_t$ から $\epsilon$ を予測するために関数近似器、またはノイズの推定器 $\epsilon_\theta$ を用いて

$$
\begin{align*}
L_\text{simple} (\theta) := \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta (\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t) \|^2 \right]
\tag{14}
\end{align*}
$$

を最適化対象の目的関数として訓練を実行した[^1]。また、$\epsilon_\theta$ は通常 U-Net を用いて実装される。入出力のテンソルの次元が同じアーキテクチャなので使いやすいというのもあるのだろう。

[^1]: ここに至る式の導出は [ゼロから作るDeep Learning ❺](https://www.oreilly.co.jp//books/9784814400591/) に詳しい。

ネガティブプロンプトを扱うに当たっては、$\epsilon_\theta$ がクラス識別子 $c$ を受け取れるようにした上で、結論としては $p_+$ を (ポジティブ) プロンプト、$p_-$ をネガティブプロンプトとして、U-Net $\epsilon_\theta$ に以下のようにプロンプトは渡される:

$$
\begin{align*}
\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t) = (1 + w) \epsilon_\theta (\mathbf{x}_t, c (p_+), t) - w \epsilon_\theta (\mathbf{x}_t, c (p_-), t)
\end{align*}
$$

この $\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t)$ をもって、$\epsilon \approx \hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t)$ という近似がなされることになる。

これに至る流れを簡単に眺めよう。

# 条件付けとガイダンス

## 分類器ありガイダンス

arXiv:2105.05233 [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) でいわゆる「分類器ありガイダンス」が導入された。

**スコア関数**という概念があり、任意の確率密度関数 $p(x),\ x \in \R^d$ に対して、

$$
\begin{align*}
s (x) &= \nabla_x \log p (x) \\
&= \nabla_x p(x) / p(x)
\end{align*}
$$

で表されるようなものとなっている[^2]。

[^2]: この辺の話は [拡散モデル ― データ生成技術の数理](https://www.iwanami.co.jp/book/b619864.html) に詳しい。

「4.1 Conditional Reverse Noising Process」によると、無条件の逆ノイズ過程 $p_\theta (x_t | x_{t+1})$ を持つ拡散モデルの場合、この過程に関しての（DDPM で登場するような）ノイズを予測するモデル $\epsilon_\theta (x_t)$ がある場合、スコア関数に対する以下のような式が成立するそうである。

$$
\begin{align*}
\nabla_{x_t} \log p_\theta (x_t) = - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta (x_t)
\tag{11}
\end{align*}
$$

または、

$$
\begin{align*}
\epsilon_\theta (x_t) = - \sqrt{1 - \bar{\alpha}_t} \; \nabla_{x_t} \log p_\theta (x_t).
\end{align*}
$$

これは、「ノイズを予測するモデル $\epsilon_\theta (x_t)$」が重みをつけたスコア関数として解釈できることを意味している。
分類器を $p_\phi (y | x_t, t)$ とする時、上記を用いて $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ に対する新しい予測

$$
\begin{align*}
\hat{\epsilon} (x_t) := \epsilon_\theta (x_t) - \sqrt{1 - \bar{\alpha}_t} \; \nabla_{x_t} \log p_\phi (y | x_t)
\tag{14}
\end{align*}
$$

が定義されている。

## 分類器なしガイダンス

arXiv:2207.12598 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) でいわゆる「分類器なしガイダンス」が導入された。「3.1 CLASSIFIER GUIDANCE」では上記の Eqs. (11), (14) は

- $x_t$ を潜在 $\mathbf{z}_\lambda$
- クラス識別子を $y$ から $\mathbf{c}$
- 時刻 $t$ を連続変数 $\lambda \in [\lambda_\text{min}, \lambda_\text{max}]$
- 分類器を $p_\phi$ から $p_\theta$

に書き換えて、同等の式

$$
\newcommand{\bepsilon}{\boldsymbol{\epsilon}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\bc}{\mathbf{c}}
\begin{align*}
\bepsilon_\theta (\bz_\lambda, \bc) \approx -  \sigma_\lambda \nabla_{\bz_\lambda} \log p (\bz_\lambda | \bc)
\end{align*}
$$

と

$$
\newcommand{\bepsilon}{\boldsymbol{\epsilon}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\bc}{\mathbf{c}}
\begin{align*}
\tilde{\bepsilon}_\theta (\bz_\lambda, \bc) := \bepsilon_\theta (\bz_\lambda, \bc) - w \sigma_\lambda \nabla_{\bz_\lambda} \log p_\theta (\bc | \bz_\lambda)
\end{align*}
$$

として引用されている。既に触れたように、「ノイズを予測するモデル」が重みをつけたスコア関数として解釈できることから、前者の式のことを**拡散スコア**  (diffusion score) とか**スコア推定器** (score estimator) と呼んでいるようである。

また、無条件モデルの場合用の特別なクラス識別子として null トークン $\varnothing$ が定義されている。

分類器なしガイダンスでは拡散モデルとは別の分類器モデルを訓練する代わりに、無条件のデノイジング拡散モデル (DDPM) と条件付きモデルを同時に訓練する。
無条件のデノイジング拡散モデルについては、$\newcommand{\bepsilon}{\boldsymbol{\epsilon}}\newcommand{\bz}{\mathbf{z}}\newcommand{\bc}{\mathbf{c}} \bepsilon_\theta (\bz_\lambda) := \bepsilon_\theta (\bz_\lambda, \bc = \varnothing)$ と置けば良く、同時訓練用に修正されたスコア評価器は条件ありとなしの線形結合を用いて以下のようになる:

$$
\newcommand{\bepsilon}{\boldsymbol{\epsilon}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\bc}{\mathbf{c}}
\begin{align*}
\tilde{\bepsilon}_\theta (\bz_\lambda, \bc) = (1 + w) \bepsilon_\theta (\bz_\lambda, \bc) - w \bepsilon_\theta (\bz_\lambda)
\end{align*}
$$

# ネガティブプロンプトの与え方

arXiv:2406.02965 [Understanding the Impact of Negative Prompts: When and How Do They Take Effect?](https://arxiv.org/abs/2406.02965) では上の最後の式は以下のように書かれている:

$$
\begin{align*}
\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t) = (1 + w) \epsilon_\theta (\mathbf{x}_t, c (s), t) - w \epsilon_\theta (\mathbf{x}_t, c (\empty), t)
\tag{4}
\end{align*}
$$

ネガティブプロンプトを与える場合は、$p_+$ を (ポジティブ) プロンプト、$p_-$ をネガティブプロンプトとして、

$$
\begin{align*}
\hat{\epsilon}_\theta (\mathbf{x}_t, c (s), t) = (1 + w) \epsilon_\theta (\mathbf{x}_t, c (p_+), t) - w \epsilon_\theta (\mathbf{x}_t, c (p_-), t)
\tag{5}
\end{align*}
$$

とする。

実際には、本論文は以下を主張しており、「クリティカルステップ」と呼ばれる「ネガティブプロンプトが生成過程に影響を及ぼし始めるステップ」で適用を始める必要性があるようだ (但し遅すぎると画像の形と構造が本質的に決定されているため削除効果が間に合わない)。

現状の [Diffusers](https://github.com/huggingface/diffusers) がこれを考慮しているかは不明であり、今後の課題なのかもしれないが当面は気にしないことにする。

1. 遅延効果: ネガティブプロンプトは、ポジティブプロンプトが対応するコンテンツを表示した後、遅れて効果が観察される
1. 中和による削除: ポジティブプロンプトとネガティブプロンプトの潜在空間での相互キャンセルで生成された概念を打ち消す
1. ネガティブプロンプトの早期適用は逆に望まない生成 (“Reverse Activation”) の可能性

また、そもそも冒頭で

> ネガティブプロンプトの概念は，生成してはいけないものを指定することでモデルを誘導するものであり，その有効性から大きな注目を集めている．しかし，その多くは実験結果に頼ったものであり，ネガティブプロンプトがどのように機能するかについての深い理解がない．

という趣旨のことが書かれており、どちらかと言うとテクニックのようなものとして今のところ使われているようである。

# まとめ

ネガティブプロンプトは、分類器なしガイダンスの仕組みにおいて、無条件のデノイジングにおける $\epsilon_\theta (\mathbf{x}_t, c (\empty), t)$ を $\epsilon_\theta (\mathbf{x}_t, c (p_-), t)$ に置き換えるテクニックであることが分かった。

また、arXiv:2406.02965 [Understanding the Impact of Negative Prompts: When and How Do They Take Effect?](https://arxiv.org/abs/2406.02965) にてかなり詳細な調査が行われおり、本来はネガティブプロンプトの与え方にはタイミングが重要であることも分かった。
