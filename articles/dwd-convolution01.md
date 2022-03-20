---
title: "ニューラルネットの畳み込み層 (1)"
emoji: "⛓"
type: "tech"
topics: ["機械学習", "ポエム", "Python"]
published: true
---

# 目的

ニューラルネットワークの畳み込み層については、特徴を自動的に抽出する層としてよく本に書かれているように思われる。ざっと手持ちの本を見てもなかなか分かりにくいので何番煎じか分からないが、自分が理解できるものと残したい。

# 実験

![](/images/dwd-convolution01/001.png)
*レナさん*

といういつもの[レナさん](https://ja.wikipedia.org/wiki/%E3%83%AC%E3%83%8A_(%E7%94%BB%E5%83%8F%E3%83%87%E3%83%BC%E3%82%BF))の画像を用いる。ここでは事前にグレースケールに落とすことでチャネル数を 1 にして、更に二値化している。

PyTorch を使うことにして、以下のようにモジュールを import する。

```python
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchvision import transforms
import numpy as np
from PIL import Image
```

以下では `im` は `im = Image.open('Lenna.png')` のようなもので得られた変数とする。
これに対して、パラメータを固定にした畳み込み層を適用して眺めることを考える。畳み込み層のカーネルにラプラシアンフィルタと平均化フィルタを設定し、1 チャネルの画像データを 2 チャネルに出力する。結果が分かりにくくなるのでバイアスはなしとし、画像サイズを変更したくないのでストライドを 1 とした。

```python
conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)

kernel = np.array([
    # ラプラシアンフィルタ（エッジ検出）
    [[
        [1.,  1., 1.],
        [1., -8., 1.],
        [1.,  1., 1.]
    ]],
    # 平均化フィルタ（ぼかし）
    [[
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]]
])

conv.weight = Parameter(torch.from_numpy(kernel).float(), requires_grad=False)
```

これを使って以下のようにすると、2 つのチャネルの情報を別々に取り出せる。

```python
im_tensor = transforms.ToTensor()(im).unsqueeze(0)
convolved_im = conv.forward(im_tensor).cpu().squeeze(0)

im2 = transforms.ToPILImage()(convolved_im[0])
im3 = transforms.ToPILImage()(convolved_im[1])
```

それぞれ表示すると以下のようになっており、エッジの検出とぼかしがかかっていることが分かる。

![](/images/dwd-convolution01/003.png)
*ラプラシアンフィルタ*

![](/images/dwd-convolution01/004.png)
*平均化フィルタ*

これが畳み込み層による特徴の抽出である。
畳み込み層を作成する時に、ストライド > 1 に設定することで値に応じたダウンスケールが適用されることになる。

```python
conv = nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1, bias=False)
```

![](/images/dwd-convolution01/005.png)
*ストライド 2 によるダウンスケール*

# まとめ

画像処理で使われる簡単なフィルタを畳み込み層に指定することで畳み込み層の動作を視覚的に確認できた。
