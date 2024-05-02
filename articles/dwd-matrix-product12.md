---
title: "行列積状態について考える (12) — テンソルネットワークで MNIST 分類"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "PyTorch", "TensorNetwork"]
published: true
---

# 目的

文献 [E] arXiv:1906.06329「[TensorNetwork for Machine Learning](https://arxiv.org/abs/1906.06329)」を読んで PyTorch で実装してみたので記事にする。

# テンソルネットワーク画像分類器

以下のようなテンソルトレインを用いた画像分類モデルを考える。

![](/images/dwd-matrix-product12/001.png =400x)

論文の FIG. 2. のテンソルのノードに適当に名前を付けるとこのような感じになる。

$$
\begin{align*}
T_{i_1 i_2 i_3 i_4 i_5 i_6}^{\ell} = \sum_{\alpha_1,\alpha_2,\alpha_3,\alpha_4,\alpha_5,\alpha_6=0}^9 A^{(1)}_{i_1 \alpha_1} A^{(2)}_{i_2 \alpha_1 \alpha_2} A^{(3)}_{i_3 \alpha_2 \alpha_3} F^{\ell}_{\alpha_3 \alpha_4} A^{(4)}_{i_4 \alpha_4 \alpha_5} A^{(5)}_{i_5 \alpha_5 \alpha_6} A^{(6)}_{i_6 \alpha_6}
\tag{1}
\end{align*}
$$

ここで、$F^{\ell}_{\alpha_3 \alpha_4}$ は論文で言う「_“label” node_」でどこに入れても良いが、論文の図のように真ん中に差し込んだ。今回は皆が大好きな MNIST

![](/images/dwd-matrix-product12/002.png =400x)

を想定しているので $\ell \in \{0, \ldots, 9\}$ となる。$i_j \in \{0,1\}$ は画素空間のデータと縮約するための “脚” である。テンソルトレイン間の仮想インデックスの結合次元を $\chi = 10$ とした。2 次元画素空間は $X = [0, 1]^2$ であるが、特徴マップ $\Phi$ によって、画像は $\mathscr{X} = X^{\otimes 6}$ にエンコードされる。

例えば 6 画素の画像は平坦化することで $\bm{p} = (p_1, p_2, p_3, p_4, p_5, p_6)$ という画素の列になる。これを特徴マップ $\Phi$ によって $\mathscr{X}$ にうつすと

$$
\begin{align*}
\Phi(\bm{p}) &= \Phi(p_1) \otimes \Phi(p_2) \otimes \Phi(p_3) \otimes \Phi(p_4) \otimes \Phi(p_5) \otimes \Phi(p_6) \\
&=: (x_1, x_2, x_3, x_4, x_5, x_6) \\
&= \bm{x}
\end{align*}
$$

となる。これに対して、先ほどのテンソル分類器 $T^{\ell}$ を作用させると $(f^{(\ell)}(\bm{x}))_{0 \le \ell \le 9} =(T^{\ell} \cdot \bm{x})_{0 \le \ell \le 9} \in \R^{10}$ となる。

## ここからは機械学習勢おなじみ

後は機械学習でよくあるように、**softmax を通して各ラベルごとの確率を出して、正解ラベルとのクロスエントロピー損失をとれば良い** ということになる。

ここまでの内容は文献 [S] で扱われいるものであるが、計算方法が 1992 年に S. White によって開発された DMRG (密度行列繰り込み群) アルゴリズムという計算物理の手法によっているため、文献 [E]（解説が文献 [N] 第 8 章に詳しい）では自動微分を用いることで機械学習の実践者に優しい内容にしたということである。オリジナルは TensorFlow を用いているが、今回は PyTorch を使ってスクラッチから実装してみた。

# データとテンソルトレインの縮約

各 $\ell \in \{0, \ldots, 9\}$ に対する

$$
\begin{align*}
f^{(\ell)}(\bm{x}) = T^{\ell} \cdot \bm{x}
\end{align*}
$$

を掘り下げる。右辺を明示的に書くと

$$
\begin{align*}
f^{(\ell)}(\bm{x}) = \sum_{i_1,i_2,\ldots,i_6=0}^{1} T_{i_1 i_2 i_3 i_4 i_5 i_6}^{\ell} x_{i_1} x_{i_2} x_{i_3} x_{i_4} x_{i_5} x_{i_6}
\end{align*}
$$

となる。式 (1) を思い出すと全体として、“脚” に考えられる全パターンの組み合わせが実行されることになる。

## 数値計算上の懸念

テンソルトレインの各ノードを $A$ で代表することにして、$A$ の各成分と $F$ の各成分が凡そ $w$ 程度の値とする。すると式 (1) より、各インデックスの組み合わせごとに

$$
\begin{align*}
A^{(1)}_{i_1 \alpha_1} A^{(2)}_{i_2 \alpha_1 \alpha_2} A^{(3)}_{i_3 \alpha_2 \alpha_3} F^{\ell}_{\alpha_3 \alpha_4} A^{(4)}_{i_4 \alpha_4 \alpha_5} A^{(5)}_{i_5 \alpha_5 \alpha_6} A^{(6)}_{i_6 \alpha_6} \approx w^7
\end{align*}
$$

となる。$w$ が大きいと特徴量の個数に合わせて指数関数的に値が爆発するし、$w$ が小さいと指数関数的に値が消失する。

次に

$$
\begin{align*}
\sum_{\alpha_1,\alpha_2,\alpha_3,\alpha_4,\alpha_5,\alpha_6=0}^9
\end{align*}
$$

の部分だが、組み合わせ数は $10^6$ 個である。よって、$T_{i_1 i_2 i_3 i_4 i_5 i_6}^{\ell} \approx 10^6 w^7$ である。これが現実問題としては、`torch.float32` の範囲で扱われる必要が出てくる。

また、縮約計算中に $F^{\ell}_{\alpha_3 \alpha_4}$ を計算に含めると、$\ell \in \{0, \ldots, 9\}$ のため、**GPU メモリ上に保存している計算途中のデータが 10 倍になる** ので、これを縮約するのは最後にするほうが良い。つまり、テンソルトレインの両端から徐々に縮約していって、最後に分類器のノードとの縮約をとるという形だ。

FIG. 3. に詳細が書かれているが、今回は少々さぼって両端から 2 特徴量ずつ関連するテンソルを縮約していくことにした。

## テンソルの初期化

実際、上記にように計算がデリケートなため、ランダム初期化を用いてしまうと $w$ がとてもシビアな範囲で初期化されないと計算が爆発したり消失 (？) してしまう。公式実装の文献 [M] の「mnist_example.ipynb」では「**単位行列を並べたような疎な行列を初期化の基本とし、それをランダムに僅かに摂動する**」という手法をとっている。試した限りこれはうまく行く。

ランダム初期化でもいけるかもしれないが、いけるパターンを探すのがどんどん難しくなり、28x28 の MNIST では断念した。7x7 までリサイズした MNIST だとある程度は学習できるパターンが見つかったが、今回は断念した。

# 実装

以上を踏まえて実装したい。テストは Google Colab 上で T4 を用いて行った。

## 必要なモジュールの import

```python
from __future__ import annotations

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchinfo
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
```

## MNIST のダウンロード

```python
root = os.path.join(os.getenv('HOME'), '.torch')

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

default_pix_dims = 2  # dimension of feature space
default_bond_dims = 10  # common dimension of virtual indices
img_size = 28
n_feature = img_size * img_size

transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.QMNIST(
    root=root,
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.QMNIST(
    root=root,
    train=False,
    download=True,
    transform=transform
)
```

## データローダーの作成

```python
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size = 32,
    shuffle=True,
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size = 32,
    shuffle=False,
)
```

## モデルの実装

```python
def make_tt(
    n_feature: int, pix_dims: int, bond_dims: int, n_class: int,
    classifier_idx: int | None = None, std: float = 1e-3,
) -> list[torch.Tensor]:
    tt_cores = []
    for i in range(n_feature):
        if i == 0:
            dims = (pix_dims, bond_dims)
            core = torch.zeros(dims)
            core[:, 0] = 1
            core += torch.normal(mean=0.0, std=std, size=core.shape)
        elif i == n_feature - 1:
            dims = (bond_dims, pix_dims)
            core = torch.zeros(dims)
            core[0, :] = 1
            core += torch.normal(mean=0.0, std=std, size=core.shape)
        else:
            dims = (bond_dims, pix_dims, bond_dims)
            core = torch.tensor(
                np.array(pix_dims * [np.eye(bond_dims)],
                dtype=np.float32)
            ).permute(1, 0, 2)
            core += torch.normal(mean=0.0, std=std, size=core.shape)
        tt_cores.append(core)
    if classifier_idx is not None:
        dims = (bond_dims, n_class, bond_dims)
        core = torch.tensor(
            np.array(n_class * [np.eye(bond_dims)], dtype=np.float32)
        ).permute(1, 0, 2)
        core += torch.normal(mean=0.0, std=std, size=core.shape)
        tt_cores.insert(classifier_idx, core)
    return tt_cores


class FeatureEmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torchTensor) -> tuple[torchTensor, ...]:
        x = torch.flatten(x, start_dim=1)
        x = torch.stack([1-x, x], axis=1).permute((2, 0, 1))
        x = tuple(t.squeeze() for t in x.split(1))
        # n_feature tensors whose shape is (n_batch, pix_dims)
        return x


class WeightLayer(nn.Module):
    def __init__(self, n_feature: int, pix_dims: int, bond_dims: int, n_class: int):
        super().__init__()
        self.n_feature = n_feature
        classifier_idx = self.classifier_loc(n_feature)
        tt_cores = make_tt(
            n_feature, pix_dims, bond_dims, n_class, classifier_idx=classifier_idx
        )
        self.n_cores = len(tt_cores)
        for i, core in enumerate(tt_cores):
            param_core = nn.parameter.Parameter(core)
            setattr(self, f"tt_core{i}", param_core)

    def forward(self, x: tuple[torchTensor, ...], n_sub_features: int = 2):
        classifier_idx = self.classifier_loc(self.n_feature)
        assert(
            n_sub_features * 2 < self.n_feature,
            f"{n_sub_features*2=} must be < {self.n_feature=}"
        )
        n_left_right_block, n_remaining_fea = \
            self.left_sub_feature_num(n_sub_features)

        start = 0
        prev_t = None
        for i in range(n_left_right_block):
            end = start + n_sub_features
            equation = self._make_equation(
                self.n_feature, start_fea=start, end_fea=end
            )
            t = torch.einsum(equation, *x[start:end], *self.tt_cores[start:end])
            if prev_t is None:
                prev_t = t
            else:
                prev_t = torch.einsum("Bb,Bbc->Bc", prev_t, t)

            start += n_sub_features
        left_t = prev_t

        start = n_sub_features * (n_left_right_block * 2 - 1) + n_remaining_fea
        prev_t = None
        for i in range(n_left_right_block):
            end = start + n_sub_features
            equation = self._make_equation(
                self.n_feature, start_fea=start, end_fea=end
            )
            # +1 for tt_cores means consideration of shift for the classifier site
            t = torch.einsum(
                equation, *x[start:end], *self.tt_cores[start + 1:end + 1]
            )
            if prev_t is None:
                prev_t = t
            else:
                prev_t = torch.einsum("Bbc,Bc->Bb", t, prev_t)

            start -= n_sub_features
        right_t = prev_t

        start = n_left_right_block * n_sub_features
        end = start + n_remaining_fea
        equation = self._make_equation(self.n_feature, start_fea=start, end_fea=end)
        # +1 for tt_cores means consideration of including the classifier site
        classifier_t = torch.einsum(
            equation, *x[start:end], *self.tt_cores[start:end + 1]
        )
        output = torch.einsum("ab,abAc,ac->aA", left_t, classifier_t, right_t)
        return output

    def forward_full(self, x: tuple[torchTensor, ...]):
        equation = self._make_equation(self.n_feature)
        output = torch.einsum(equation, *x, *self.tt_cores)
        return output

    def left_sub_feature_num(self, n_sub_features) -> tuple[int, int]:
        remaining = self.n_feature
        i = 0
        while remaining > n_sub_features * 2:
            remaining = remaining - n_sub_features * 2
            i += 1
        return i, remaining

    @property
    def tt_cores(self):
        return [getattr(self, f"tt_core{i}") for i in range(self.n_cores)]

    @classmethod
    def classifier_loc(cls, n_feature):
        return n_feature // 2

    @classmethod
    def _make_equation(
        cls, n_feature: int, start_fea: int = 0, end_fea: int | None = None,
        batch_c: str = "a", class_c: str | None = "A"
    ):
        """make an equation for range(start_i, end_i)
        """

        if start_fea < 0:
            raise ValueError(f'{start_fea=} must be >= 0.')

        if end_fea is None:
            end_fea = n_feature

        if end_fea > n_feature:
            raise ValueError(f'{end_fea=} must be <= {n_feature=}.')

        if start_fea >= end_fea:
            raise ValueError(f'{start_fea=} must be less than {end_fea=}.')

        if end_fea - start_fea > ord("Z") - ord("B") + 1:
            raise ValueError(
                f'{end_fea=} - {start_fea=} must be less than {ord("Z")-ord("B")+1}.'
            )

        classifier_idx = cls.classifier_loc(n_feature)

        includes_classifier = start_fea <= classifier_idx <= end_fea - 1

        fea_i = ord("B")  # U+0042-
        vir_i = ord("b")  # U+0062-

        fea_idx: list[str] = []
        vir_idx: list[str] = []  # ["Bb", "bCc", "cD"]
        pre_vir_idx: list[tuple[str, ...]] = []
        for i in range(start_fea, end_fea):
            fea_idx.append(chr(fea_i))
            if i == 0:
                vir_i -= 1  # preserve first vir_i
                pre_vir_idx.append((chr(fea_i), chr(vir_i + 1)))
            elif i == n_feature - 1:
                pre_vir_idx.append((chr(vir_i), chr(fea_i)))
            else:
                pre_vir_idx.append((chr(vir_i), chr(fea_i), chr(vir_i + 1)))
            fea_i += 1
            vir_i += 1

        classifier_loc = classifier_idx - start_fea
        for i, idx in enumerate(pre_vir_idx):
            if includes_classifier:
                if i == classifier_loc:
                    if i == 0:
                        classifier_idx = f"{idx[0]}{class_c}{chr(ord(idx[0]) + 1)}"
                        vir_idx.append(classifier_idx)
                    else:
                        last_vir_idx = vir_idx[-1][-1]
                        classifier_idx = \
                            f"{last_vir_idx}{class_c}{chr(ord(last_vir_idx) + 1)}"
                        vir_idx.append(classifier_idx)
                if i >= classifier_loc:
                    idx = [c if ord(c) < ord("a") else chr(ord(c) + 1)
                            for c in list(idx)]
            vir_idx.append("".join(idx))

        fea_idx_s = ",".join([f"{batch_c}{c}"for c in fea_idx])

        out_index = batch_c
        vir_idx_s = ",".join(vir_idx)
        if ord(vir_idx_s[0]) >= ord("a"):
            out_index += vir_idx_s[0]
        if class_c in vir_idx_s:
            out_index += class_c
        if ord(vir_idx_s[-1]) >= ord("a"):
            out_index += vir_idx_s[-1]

        equation = f'{fea_idx_s + "," + vir_idx_s}->{out_index}'

        return equation


class Net(nn.Module):
    def __init__(self, n_feature: int, pix_dims: int, bond_dims: int, n_class: int):
        super().__init__()

        self.fea_layer = FeatureEmbeddingLayer()
        self.wgt_layer = WeightLayer(
            n_feature=n_feature, pix_dims=pix_dims, bond_dims=bond_dims,
            n_class=n_class
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fea_layer(x)
        x = self.wgt_layer(x)
        x = self.softmax(x)
        return x
```

## 訓練と検証ループの実装

```python
def train(net, device, train_loader, optimizer, epoch, log_interval):
    losses = []
    nll_loss = nn.NLLLoss()

    net.train()
    running_loss = 0
    n_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_samples += len(data)
        if batch_idx % log_interval == 0:
            losses.append(running_loss / n_samples)
            running_loss = 0
            n_samples = 0
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    return losses


def test(net, device, test_loader):
    nll_loss = nn.NLLLoss()

    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += nll_loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

## モデルの作成と訓練と検証

```python
%%time

net = Net(n_feature=n_feature, pix_dims=default_pix_dims,
          bond_dims=default_bond_dims, n_class=10).to(device)

optimizer = optim.Adam(net.parameters())

log_interval = 50
epochs = 1

losses = []
for epoch in range(1, epochs + 1):
    sublosses = train(net, device, trainloader, optimizer, epoch, log_interval)
    losses += sublosses
    test(net, device, testloader)
```

> Train Epoch: 1 [0/60000 (0%)]	Loss: 0.071956
> Train Epoch: 1 [1600/60000 (3%)]	Loss: 0.071391
> Train Epoch: 1 [3200/60000 (5%)]	Loss: 0.071370
> Train Epoch: 1 [4800/60000 (8%)]	Loss: 0.033660
> Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.045731
> Train Epoch: 1 [8000/60000 (13%)]	Loss: 0.028915
> Train Epoch: 1 [9600/60000 (16%)]	Loss: 0.029875
> Train Epoch: 1 [11200/60000 (19%)]	Loss: 0.032353
> Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.018047
> Train Epoch: 1 [14400/60000 (24%)]	Loss: 0.023287
> Train Epoch: 1 [16000/60000 (27%)]	Loss: 0.022523
> Train Epoch: 1 [17600/60000 (29%)]	Loss: 0.026113
> Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.015155
> Train Epoch: 1 [20800/60000 (35%)]	Loss: 0.017743
> Train Epoch: 1 [22400/60000 (37%)]	Loss: 0.023616
> Train Epoch: 1 [24000/60000 (40%)]	Loss: 0.020549
> Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.021021
> Train Epoch: 1 [27200/60000 (45%)]	Loss: 0.019376
> Train Epoch: 1 [28800/60000 (48%)]	Loss: 0.020880
> Train Epoch: 1 [30400/60000 (51%)]	Loss: 0.030461
> Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.037976
> Train Epoch: 1 [33600/60000 (56%)]	Loss: 0.013896
> Train Epoch: 1 [35200/60000 (59%)]	Loss: 0.006066
> Train Epoch: 1 [36800/60000 (61%)]	Loss: 0.009239
> Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.017888
> Train Epoch: 1 [40000/60000 (67%)]	Loss: 0.003709
> Train Epoch: 1 [41600/60000 (69%)]	Loss: 0.031147
> Train Epoch: 1 [43200/60000 (72%)]	Loss: 0.023640
> Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.005690
> Train Epoch: 1 [46400/60000 (77%)]	Loss: 0.012589
> Train Epoch: 1 [48000/60000 (80%)]	Loss: 0.007181
> Train Epoch: 1 [49600/60000 (83%)]	Loss: 0.014580
> Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.007948
> Train Epoch: 1 [52800/60000 (88%)]	Loss: 0.012705
> Train Epoch: 1 [54400/60000 (91%)]	Loss: 0.020037
> Train Epoch: 1 [56000/60000 (93%)]	Loss: 0.013667
> Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.023198
> Train Epoch: 1 [59200/60000 (99%)]	Loss: 0.015644
>
> Test set: Average loss: 0.0131, Accuracy: 54402/60000 (90.67%)
>
> CPU times: user 47min 25s, sys: 5.34 s, total: 47min 31s
> Wall time: 47min 37s

## 損失の推移

```python
plt.plot(losses)
plt.show()
```

![](/images/dwd-matrix-product12/003.png =400x)

# 驚きの高次元

$28 \times 28 = 784$ の画素データを特徴マップでテンソル積にうつしたので、概念としては $2^{784} =$ 101745825697019260773923519755878567461315282017759829107608914364075275235254395622580447400994175578963163918967182013639660669771108475957692810857098847138903161308502419410142185759152435680068435915159402496058513611411689167650816 次元の Hilbert 空間の中での分類を行った・・・みたいな状態になっているのではないかと思うが、そういう見方もできそうではあるが、物凄い大袈裟感はある。

# まとめ

とりあえず、何とか論文が主張するような精度が出る実装ができて良かった。

ところで論文によると

> _The purpose of this note is to be a resource for machine learning practitioners who wish to learn how tensor networks can be applied to classification problems._

ということなのだが、_machine learning practitioners_ の立場としてはちょっときつかった・・・。結構簡単に使えるライブラリと初期化手法、そしてパフォーマンス等々が用意されていないと、自分で実装するのは大変だなと思った。勿論 [TensorNetwork](https://github.com/google/TensorNetwork) があるにはあるが、ただの _machine learning practitioners_ が使いこなすには **テンソルネットワークの概念を含めて** ハードルが高いな・・・と感じるわけである。

どちらかと言うと [行列積状態について考える (5) — ニューラルネットワークのモデル圧縮](https://zenn.dev/derwind/articles/dwd-matrix-product05) のように、既に訓練が完了しているニューラルネットワークをテンソルネットワークに変換して微調整を行うといった用途のほうが扱いも簡単だと感じていて、スクラッチでテンソルネットワークを訓練するのはきついなと感じるのだがどうなのだろうか。

# 参考文献

[E] [TensorNetwork for Machine Learning, arXiv:1906.06329, Stavros Efthymiou, Jack Hidary, Stefan Leichenauer](https://arxiv.org/abs/1906.06329)
[M] [MPS_classifier@TensorNetwork, GitHub, Google LLC](https://github.com/google/TensorNetwork/tree/0.1.0/experiments/MPS_classifier)
[S] [Supervised Learning with Quantum-Inspired Tensor Networks, arXiv:1605.05775, E. Miles Stoudenmire, David J. Schwab](https://arxiv.org/abs/1605.05775)
[N] [西野友年, テンソルネットワーク入門, 講談社, 2023](https://www.kspub.co.jp/book/detail/5316535.html)
