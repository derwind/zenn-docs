---
title: "PyTorch について考える (1) — nn.CrossEntropyLoss と nn.NLLLoss"
emoji: "🔥"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "PyTorch"]
published: false
---

# 目的

PyTorch は長らく使っているものの、API を叩くだけということが多いので、多クラス交差エントロピー誤差について少し確認したくなった。

多分内容的には色々飛ばし過ぎていたり、一般性を損ねた書き方になっているので有識者には怒られてしまうものであろうが、とりあえずは気にしないことにする。

# 多クラス交差エントロピー誤差

API 的には [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) と [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) (The negative log likelihood loss) がよく使うところではないかと思う。

簡単のため、ミニバッチのサイズは 1 とする。

ニューラルネット（の特に全結合層）からの出力であるロジット $\mathbf{x} = (x_1, \ldots, x_n)$ とクラスラベル $y \in \mathbb{N}$ があるというところからスタートする。

まず、`softmax` を

$$
\begin{align*}
\operatorname{softmax}(\mathbf{x}) := \left( \frac{\exp(x_1)}{\sum_{i=1}^n \exp (x_i)}, \ldots, \frac{\exp(x_n)}{\sum_{i=1}^n \exp (x_i)} \right)
\end{align*}
$$

としておく。

## CrossEntropyLoss

確率分布的な部分は、極端な形でワンホットエンコーディングされていて、データがクラス $y$ に属する確率が 1 で他は 0 であるような分布を考えると、`CrossEntropyLoss` は素朴には以下のようになる:

$$
\begin{align*}
\hat{\mathbf{x}} &= \operatorname{softmax}(\mathbf{x}) \\
\operatorname{CE}(\hat{\bf{x}}, y) &= -\log \hat{\bf{x}} \cdot (0, \ldots, 0, \overbrace{1}^{y}, 0, \ldots, 0) \quad \geq 0
\end{align*}
$$

この損失値は、$\argmax (\mathbf{x}) = \argmax (\hat{\mathbf{x}}) = y$ の時に最小となる。

## NLLLoss

こちらも大変雑に書くと以下のようになる:

$$
\begin{align*}
\hat{\mathbf{x}} &= \log ( \operatorname{softmax}(\mathbf{x}) ) \\
\operatorname{NLLLoss}(\hat{\bf{x}}, y) &= - \hat{\bf{x}} \cdot (0, \ldots, 0, \overbrace{1}^{y}, 0, \ldots, 0) = - \hat{x}_y \quad \geq 0
\end{align*}
$$

この損失値も、$\argmax (\mathbf{x}) = \argmax (\hat{\mathbf{x}}) = y$ の時に最小となる。

# API を動かして確認してみる

上記を観察して思うところとしては「$\log$ をどこでとるかの問題で、`CrossEntropyLoss` も `NLLLoss` も使う分にはほとんど同じでは？」ということなのだが、これを確認したい。と言うかその認識で使っていたので、**再確認**したい。

まずは必要なモジュールを import する:

```python
import numpy as np

import torch
from torch import nn
```

必要な API の準備をする:

```python
softmax = nn.Softmax(dim=1)
log_softmax = nn.LogSoftmax(dim=1)
ce_loss = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()
```

## CrossEntropyLoss

[Softmax + Cross-Entropy Loss](https://discuss.pytorch.org/t/softmax-cross-entropy-loss/125383) や [Should I use softmax as output when using cross entropy loss in pytorch?](https://stackoverflow.com/questions/55675345/should-i-use-softmax-as-output-when-using-cross-entropy-loss-in-pytorch) に見られるように、PyTorch では `CrossEntropyLoss` の中で `Softmax` を適用するような動作になっているそうなので `Softmax` の適用を見合わせる[^1]。またラベルもワンホットエンコーディングしなくて良い:

[^1]: DeZero で言う [SoftmaxCrossEntropy](https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/dezero/functions.py#L454-L473) みたいな状態なのだろう。

```python
tensor = torch.tensor([[12.3, -5.2, 31.4]], dtype=torch.float)
#output = softmax(tensor)  # 不要
output = tensor
print(f"{output=}")
for label in [0, 1, 2]:
    target = torch.tensor([label], dtype=torch.long)
    print(" ", ce_loss(output, target))
```

> output=tensor([[12.3000, -5.2000, 31.4000]])
>   tensor(19.1000)
>   tensor(36.6000)
>   tensor(0.)

このような結果になった。次に `NLLLoss` を見よう。

## NLLLoss

```python
tensor = torch.tensor([[12.3, -5.2, 31.4]], dtype=torch.float)
output = log_softmax(tensor)
print(f"{output=}")
for label in [0, 1, 2]:
    target = torch.tensor([label], dtype=torch.long)
    print(" ", label, nll_loss(output, target))
```

> output=tensor([[-19.1000, -36.6000,   0.0000]])
>   0 tensor(19.1000)
>   1 tensor(36.6000)
>   2 tensor(0.)

`CrossEntropyLoss` と同様の結果になった。

# NumPy でも見てみる

PyTorch の実装を追いかける気にはなれないので、NumPy で試して誤魔化したい。

```python
def _softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x))

def _log_softmax(x: np.ndarray):
    return np.log(_softmax(x))

def _nll_loss(t, y):
    return -t[y]

def _onehot(t, y):
    v = np.zeros(t.shape[0])
    v[y] = 1
    return v

def _ce_loss(t, y):
    return np.sum(-np.log(t) * _onehot(t, y))

t = np.array([12.3, -5.2, 31.4], dtype=float)

output = _softmax(t)
print(f"{output=}")
for label in [0, 1, 2]:
    print(" ", label, _ce_loss(output, label))

output = _log_softmax(t)
print(f"{output=}")
for label in [0, 1, 2]:
    print(" ", label, _nll_loss(output, label))
```

> output=array([5.06961984e-09, 1.27298111e-16, 9.99999995e-01])
>   0 19.100000005069617
>   1 36.60000000506962
>   2 5.069619958913665e-09
> output=array([-1.91000000e+01, -3.66000000e+01, -5.06961996e-09])
>   0 19.100000005069617
>   1 36.60000000506962
>   2 5.069619958913665e-09

PyTorch と同様の結果を得られたと思う。

# まとめ

細かいところは色々間違っていそうだったり、結果としてはそうなるけどという部分がありそうな気がするが、ざっくりとした API の挙動を明示的に見ておきたかったので実験してみた。
