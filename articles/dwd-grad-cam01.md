---
title: "ニューラルネットの説明可能性について考える (1) — Grad-CAM"
emoji: "⛓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習"]
published: false
---

# 目的

機械学習のモデルが何故そのような結果を出すのかについて知りたいことがある。決定木のようなモデルの場合、かなり分かりやすいのだがニューラルネットワークの場合にはハッキリ言ってブラックボックスだ。

ところで [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) という研究がある。CAM (Class Activation Mapping) は分類における重要な因子を計算する技術らしく、Grad-CAM はその改善版らしい。今回は論文を読んでないので、深くは触れないことにする。

DeepL に概要を放り込むと以下のような内容が返ってきた:

> 我々は、CNNに基づく大規模なモデルの決定に対して「視覚的説明」を生成し、それらをより透明化する手法を提案する。我々の手法である勾配重み付けクラス活性化マッピング（Grad-CAM）は、任意のターゲット概念の勾配を利用し、最終畳み込み層に流入して、その概念を予測するための画像中の重要な領域を強調する粗い局所化マップを生成する。Grad-CAMは様々なCNNモデル群に適用可能である

要するに、何か良さそうだな、と。

# 目次？

今回、3 本立ての形にする。

1. あるデータセットに対する手持ちの二値分類器に Grad-CAM を適用
2. VGG16 と ImageNet っぽい画像に Grad-CAM を適用
3. 1. のデータセットに対するスクラッチから鍛えた二値分類器に Grad-CAM を適用

Grad-CAM の実装は [Advanced AI explainability for PyTorch](https://github.com/jacobgil/pytorch-grad-cam) を使う。

# 1. あるデータセットに対する手持ちの二値分類器に Grad-CAM を適用

[Concrete Crack Images for Classification](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification) というコンクリートにヒビが入っていたり入っていなかったりというデータセットがあって二値分類に使える。これについて、よくあるように ImageNet で訓練済みの VGG16 からの転移学習をしたモデルを作った。VGG16 を使った理由としては、ImageNet とは分布の異なるデータセットに対してさえ、訓練済み VGG16 の特徴量抽出力は汎化するような感じがしていて、雑に分類器を実装しても結果を出すからである。

以下、暫く特筆する部分もないので淡々と実装を列挙する。

## 必要なモジュールを import

```python
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchinfo

import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image
)
```

## 二値分類器

恐らく限りなく素朴な転移学習である。1 つだけ注意点があって、以下でロードする pretrained パラメータは転移学習によるものであり、訓練可能な (勾配が計算される) パラメータが大幅に少ない。

> Total params: 134,268,738
> Trainable params: 119,554,050
> Non-trainable params: 14,714,688

一方で、Grad-CAM では名前の通りに勾配情報を使うようなので、ネットワーク全体について勾配計算をできるようにする必要がある。

```python
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg16 = models.vgg16(weights="IMAGENET1K_V1", progress=True)

        for name, param in self.vgg16.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

        # 二値分類器
        self.vgg16.classifier[6] = torch.nn.Linear(4096, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.softmax(x)
        return x

net = Net()
net.load_state_dict(torch.load("checkpoint.pt"))

# ネットワーク全体を勾配計算可能なようにする。
for param in net.parameters():
    param.requires_grad = True

net = net.to(device)
```

## データローダ

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("crack_dataset", transform=transform)
# https://discuss.pytorch.org/t/how-to-use-sklearns-train-test-split-on-pytorchs-dataset/31521
train_idx, valid_idx = train_test_split(
    list(range(len(dataset.targets))), test_size=0.2,
    random_state=42, stratify=dataset.targets
)
val_dataset = torch.utils.data.Subset(dataset, valid_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
```

## データセットの可視化

```python
imgs, lbls = next(iter(val_loader))
imgs = imgs.numpy() * [
    [[[0.229]], [[0.224]], [[0.225]]]] + [[[[0.485]], [[0.456]], [[0.406]]]
]
imgs = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
lbls = [v.item() for v in lbls.numpy()]

row = 2
col = 5
n_data = row * col

fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(8, 3))
for i, img in enumerate(imgs[:n_data]):
    r= i // col
    c= i % col
    ax[r, c].set_title(lbls[i], fontsize=8)
    ax[r, c].axes.xaxis.set_visible(False)
    ax[r, c].axes.yaxis.set_visible(False)
    ax[r, c].imshow(img)
```

![](/images/dwd-grad-cam01/001.png)

## 分類精度の確認

以下のように転移学習によって、冒頭で言及したような十分な精度が出せている。

```python
def test(net, device, test_loader):
    nll_loss = nn.NLLLoss()

    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += nll_loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


test(net, device, val_loader)
```

> Test set: Average loss: 0.0102, Accuracy: 7953/8000 (99.41%)

## Grad-CAM を適用する

ここまでくると、素晴らしい転移学習の精度に、さぞかし説明可能な何かが得られそうな予感がする。

**入力テンソルの準備**

```python
imgs, lbls = next(iter(val_loader))
imgs = imgs.numpy() * [
    [[[0.229]], [[0.224]], [[0.225]]]] + [[[[0.485]], [[0.456]], [[0.406]]]
]
imgs = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

img = imgs[4]

img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
input_tensor = input_tensor.to(device)
```

**ラベル 0 (ヒビなし)**

```python
targets = [ClassifierOutputTarget(0)]  # ラベル 0
target_layers = [net.vgg16.features[29]]
with GradCAM(model=net, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
display(Image.fromarray(images))
```

![](/images/dwd-grad-cam01/002.png)

**ラベル 1 (ヒビあり)**

```python
targets = [ClassifierOutputTarget(1)]  # ラベル 1
...
display(Image.fromarray(images))
```

![](/images/dwd-grad-cam01/003.png)

ヒビなしの場合にはコンクリートの部分を見に行ってそうなので、なんとなくそれっぽい一方で、ヒビありの場合には**どこを見ているのかが分からない**。

```python
print(torch.argmax(net(input_tensor)))
```

> tensor(1, device='cuda:0')

なので、最終的にはヒビありのほうが確度が高いという判定になっているが、どこを見てそういう結果になったのかが分からないということになってしまった・・・。

# 2. VGG16 と ImageNet っぽい画像に Grad-CAM を適用

何か間違っているのかもしれないので、基本に戻ろう。

## ImageNet のような画像

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("images", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

imgs, lbls = next(iter(dataloader))
imgs = imgs.numpy() * [
    [[[0.229]], [[0.224]], [[0.225]]]] + [[[[0.485]], [[0.456]], [[0.406]]]
]
imgs = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
lbls = [v.item() for v in lbls.numpy()]

row = 2
col = 5
n_data = row * col

fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(8,3))
for i, img in enumerate(imgs[:n_data]):
    r= i // col
    c= i % col
    ax[r, c].set_title(lbls[i], fontsize=8)
    ax[r, c].axes.xaxis.set_visible(False)
    ax[r, c].axes.yaxis.set_visible(False)
    ax[r, c].imshow(img)
```

![](/images/dwd-grad-cam01/004.png)

かなり ImageNet っぽそうな画像である。

## モデル

VGG16 そのものを使う。

```python
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

model = models.vgg16(weights="IMAGENET1K_V1", progress=True)
model = model.to(device)
```

## Grad-CAM を適用する

**入力テンソルの準備**

気持ちの良さそうな車と雲の画像を使う。

```python
imgs, lbls = next(iter(dataloader))
imgs = imgs.numpy() * [
    [[[0.229]], [[0.224]], [[0.225]]]] + [[[[0.485]], [[0.456]], [[0.406]]]
]
imgs = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

cloud_and_car = imgs[3]

cloud_and_car = np.float32(cloud_and_car) / 255
input_tensor = preprocess_image(cloud_and_car, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
input_tensor = input_tensor.to(device)
```

**ラベル 159 (足の長い犬)**

```python
targets = [ClassifierOutputTarget(159)]  # 'Rhodesian ridgeback'
target_layers = [model.features[29]]
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(cloud_and_car, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*cloud_and_car), cam , cam_image))
Image.fromarray(images)
```

![](/images/dwd-grad-cam01/005.png)

**ラベル 511 (車)**

```python
targets = [ClassifierOutputTarget(511)]  # 'convertible'
target_layers = [model.features[29]]
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(cloud_and_car, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*cloud_and_car), cam , cam_image))
Image.fromarray(images)
```

![](/images/dwd-grad-cam01/006.png)

今度は「なるほどな」という結果になった。因みに空の部分はラベル 978 の「海岸」みたいなのを指定すると焦点が当たる。

## 仮説

さて、ヒビ割れコンクリートの場合はこのようなスッキリする形にはならなかった。何故だろう？

と考えると、転移学習のせいだろうなとなる。そもそも**ヒビ割れコンクリート**は ImageNet 的な画像ではないと思われるので、ImageNet とは分布の異なるデータセットである。それに対して無理やり ImageNet で訓練した VGG16 の特徴抽出器を適用して二値分類モデルを作ったものの、恐らくそれは我々が思うのとはちょっと違う観点でヒビの有無を見ているのだと思われる。

# 3. 1. のデータセットに対するスクラッチから鍛えた二値分類器に Grad-CAM を適用

仮説の 1 つの検証として、自前でカスタム分類器を作れば Grad-CAM でスッキリした結果を得られるのでは？というのがある。これを試そう。

## カスタム二値分類器

VGG のアーキテクチャにインスパイアされる形でひたすら畳み込んで分類する、コンパクトなネットワークを作る。特に意味はない。流行りもの？の `GELU` も使ってしまう。

```python
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.GELU(),
            nn.MaxPool2d(5),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.GELU(),
            nn.MaxPool2d(5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 2),
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


net = CustomNet()
net = net.to(device)
```

## 訓練

よくあるボイラープレート的なコードで 2 epochs 回してみる。

```python
def train(net, device, train_loader, optimizer, epoch, log_interval):
    nll_loss = nn.NLLLoss()

    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224, antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("crack_dataset", transform=transform)
train_idx, valid_idx = train_test_split(
    list(range(len(dataset.targets))), test_size=0.2,
    random_state=42, stratify=dataset.targets
)
train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

batch_size = 64
log_interval = 50
epochs = 2

train_dataset = torch.utils.data.Subset(dataset, train_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)

for epoch in range(1, epochs + 1):
    train(net, device, train_loader, optimizer, epoch, log_interval)
    test(net, device, val_loader)
```

> ...
> Test set: Average loss: 0.0027, Accuracy: 7873/8000 (98.41%)
> ...
> Test set: Average loss: 0.0010, Accuracy: 7955/8000 (99.44%)

適当な実装のわりにはそこそこ良さそうかなと思う。

## Grad-CAM を適用する

**入力テンソルの準備**

```python
imgs, lbls = next(iter(val_loader))
imgs = imgs.numpy() * [
    [[[0.229]], [[0.224]], [[0.225]]]] + [[[[0.485]], [[0.456]], [[0.406]]]
]
imgs = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

img = imgs[0]

img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
input_tensor = input_tensor.to(device)
```

**ラベル 0 (ヒビなし)**

```python
targets = [ClassifierOutputTarget(0)]  # ラベル 0
target_layers = [net.vgg16.features[29]]
with GradCAM(model=net, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
display(Image.fromarray(images))
```

![](/images/dwd-grad-cam01/007.png)

**ラベル 1 (ヒビあり)**

```python
targets = [ClassifierOutputTarget(1)]  # ラベル 1
...
display(Image.fromarray(images))
```

![](/images/dwd-grad-cam01/008.png)

今度はかなり納得感のあるところを見た上で判断しているように感じられるのではないだろうか？

```python
print(torch.argmax(net(input_tensor)))
```

> tensor(1, device='cuda:0')

その上で、正しくヒビありであると判断している。

# まとめ

実はかなり綺麗な結果が出る画像を選んだのだが、1. の転移学習では綺麗な結果になる画像が見つからなかったのに対し、3. のスクラッチから訓練したカスタムモデルでは綺麗な結果になる画像が見つかるのでまぁ良いかなと思う。

いずれにせよデータセットが簡単すぎて、あまり真面目に注目しなくても分類できてしまうと思うので、綺麗な可視化は難しいかもしれないが、ヒビに注目してくれるものもあって良かった。

結局ニューラルネットワークの気持ちは分からないが、何かしら説明を求めらた時に Grad-CAM が使えるかもしれない。
