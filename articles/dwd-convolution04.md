---
title: "ニューラルネットの畳み込み層 (4) — 量子畳み込みニューラルネットワークと比較"
emoji: "⛓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "ポエム", "Python"]
published: true
---

# 目的

[Qiskit で遊んでみる (18) — Quantum Convolutional Networks その 1](/derwind/articles/dwd-qiskit18) で量子畳み込みニューラルネットワークの訓練をしたが、普通の畳み込みニューラルネットワークでもやっておくという内容。

# データセット

[11_quantum_convolutional_neural_networks.ipynb](https://github.com/qiskit-community/qiskit-machine-learning/blob/stable/0.6/docs/tutorials/11_quantum_convolutional_neural_networks.ipynb) のデータセットを PyTorch で使いやすいように箱詰めする:

```python
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import train_test_split


class HorVerBars:
    classes = [
        "-1 - horizontal",
        "1 - vertical",
    ]

    def __init__(
        self,
        train: bool = True,
        data_size: int = 50,
        test_size: float = 0.0,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.data_size = data_size
        self.test_size = test_size
        self.data, self.targets = self._load_data()

    @classmethod
    def create_train_and_test(
        cls,
        data_size: int = 50,
        test_size: float = 0.0,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> tuple[HorVerBars, HorVerBars]:
        trainset = HorVerBars(
            data_size=data_size, transform=transform,
            target_transform=target_transform
        )
        testset = HorVerBars(
            data_size=0, transform=transform,
            target_transform=target_transform
        )

        train_images, test_images, train_labels, test_labels = train_test_split(
            trainset.data, trainset.targets, test_size=test_size
        )

        trainset.data = train_images
        trainset.targets = train_labels
        trainset.train = True
        trainset.data_size = data_size
        trainset.test_size = test_size

        testset.data = test_images
        testset.targets = test_labels
        trainset.train = False
        testset.data_size = data_size
        testset.test_size = test_size

        return trainset, testset

    def _load_data(self) -> tuple[np.ndarray, np.ndarray]:
        images, labels = generate_dataset(self.data_size)
        if 0.0 < self.test_size < 1.0:
            train_images, test_images, train_labels, test_labels = \
            train_test_split(
                images, labels, test_size=self.test_size
            )
        elif self.test_size == 0.0:
            train_images, train_labels = images, labels
            test_images, test_labels = None, None
        elif self.test_size == 1.0:
            train_images, train_labels = None, None
            test_images, test_labels = images, labels
        else:
            raise ValueError("test_size should be in [0.0, 1.0]")

        if self.train:
            return train_images, train_labels

        return test_images, test_labels

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        data, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}


def generate_dataset(num_images):
    from qiskit.utils import algorithm_globals

    images = []
    labels = []
    hor_array = np.zeros((6, 8))
    ver_array = np.zeros((4, 8))

    j = 0
    for i in range(0, 7):
        if i != 3:
            hor_array[j][i] = np.pi / 2
            hor_array[j][i + 1] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, 4):
        ver_array[j][i] = np.pi / 2
        ver_array[j][i + 4] = np.pi / 2
        j += 1

    for _ in range(num_images):
        rng = algorithm_globals.random.integers(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = algorithm_globals.random.integers(0, 6)
            images.append(np.array(hor_array[random_image]))
        elif rng == 1:
            labels.append(1)
            random_image = algorithm_globals.random.integers(0, 4)
            images.append(np.array(ver_array[random_image]))

        # Create noise
        for i in range(8):
            if images[-1][i] == 0:
                images[-1][i] = algorithm_globals.random.uniform(0, np.pi / 4)
    return images, labels

```

# モジュールのインポート

```python
from __future__ import annotations

import sys
import math
import pickle
from collections.abc import Sequence
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import numpy as np
```

# データセットの準備

上で作ったものを使う:

```python
data_size = 50

trainset, testset = HorVerBars.create_train_and_test(
    data_size=data_size,
    test_size=0.3,
    transform=lambda x: torch.tensor(x.reshape(1, *x.shape), dtype=torch.float32)
)
```

# モデルの定義

あまり意味はないのだが、なんとなく [11_quantum_convolutional_neural_networks.ipynb](https://github.com/qiskit-community/qiskit-machine-learning/blob/stable/0.6/docs/tutorials/11_quantum_convolutional_neural_networks.ipynb) に似た感じのサイズダウンをする畳み込みニューラルネットワークを作る。$Z$ ハミルトニアンの期待値の代替物としては $[-1, 1]$ に値をとる活性関数として $\tanh$ にした。

```python
def create_model():
    model = nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=2, padding=0),  # 8 -> 7
        nn.AvgPool1d(kernel_size=1, stride=2),  # 7 -> 4
        nn.Conv1d(1, 1, kernel_size=2, padding=0),  # 4 -> 3
        nn.AvgPool1d(kernel_size=1, stride=2),  # 3 -> 2
        nn.Conv1d(1, 1, kernel_size=1, padding=0),  # 2 -> 2
        nn.AvgPool1d(kernel_size=1, stride=2),  # 2 -> 1
        nn.Tanh()
    )

    return model
```

## モデルの確認

`torchinfo` の `summary` で念のため確認しておく:

```python
summary(
    model,
    input_size=(len(trainset), 1, 8),
)
```

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [35, 1, 1]                --
├─Conv1d: 1-1                            [35, 1, 7]                3
├─AvgPool1d: 1-2                         [35, 1, 4]                --
├─Conv1d: 1-3                            [35, 1, 3]                3
├─AvgPool1d: 1-4                         [35, 1, 2]                --
├─Conv1d: 1-5                            [35, 1, 2]                2
├─AvgPool1d: 1-6                         [35, 1, 1]                --
├─Tanh: 1-7                              [35, 1, 1]                --
==========================================================================================
Total params: 8
Trainable params: 8
Non-trainable params: 0
Total mult-adds (M): 0.00
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
==========================================================================================
```

# 訓練ループ

結構適当だが、分類モデル用のありがちな実装を使う。

```python
def RunTrain(
    dataset: Dataset,
    batch_size: int,
    model: nn.Module,
    init: Sequence[float] | None = None,
    epochs: int = 1,
    interval: int = 100
):
    loss_list = []

    dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    cnt = -1

    for epoch in range(epochs):
        for batch, label in dataloader:
            label = label.float().view(-1, 1, 1)
            output = model(batch)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            loss_list.append(loss_value)

            cnt += 1
            if cnt % interval == 0:
                print(f'{loss_value=}')

    return loss_list
```

# 訓練

概ね 1 秒未満で完了するはず。

```python
%%time

model = create_model()

loss_list = RunTrain(trainset, len(trainset), model, epochs=200, interval=10)

print(f'final loss={loss_list[-1]}')
```

> ...
> final loss=0.8817483186721802
> CPU times: user 4.2 s, sys: 177 ms, total: 4.38 s
> Wall time: 785 ms

![](/images/dwd-convolution04/001.png)

# 評価

とても小さなデータセットだったり、ネットワークの初期値を調整していなかったり、エポック数や early stoppoing を考慮していないなどの様々な理由で精度は大変不安定で、良い精度が出るところで訓練が終わるかはガチャ要素が大きい。

見栄え優先で、なんどもガチャを回して、良いテスト精度が出るまで頑張ってみた。

```python
testloader = DataLoader(testset, len(testset))

total = 0
total_correct = 0

model.eval()

with torch.inference_mode():
    for i, (batch, label) in enumerate(testloader):
        label = label.float().view(-1, 1, 1)
        output = model(batch)
        predict_labels = torch.sign(output)
    
        total_correct += torch.sum(predict_labels == label).numpy()
        total += batch.shape[0]

print(f'test acc={np.round(total_correct/total, 2)}')
```

> test acc=0.87

# まとめ

フルバッチの 200 エポックくらいなら 1 秒もかからないので最高に速い。

トイデータセット状態のようなのであまり精度等には意味がないのだが、ざっくりと量子畳み込みニューラルネットワークとの比較ができた。結論としては、微分可能なニューラルネットワークにおける逆誤差伝播法は本当に優れた数理で、高速に勾配が計算できると思う。対して、量子回路の場合のパラメータシフト則による勾配計算はパラメータごとに位相をずらして順伝播計算をしまくらないとならないので、かなりコストがかかってしまうと感じる。

今回の実験だけでは何も言えないが、かなりハイブリッド量子計算向きのデータセットでないと量子計算を使うメリットは薄そうな気はする。
