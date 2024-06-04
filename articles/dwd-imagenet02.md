---
title: "ImageNet について考える (2) — Tiny ImageNet の分類"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "ImageNet", "PyTorch"]
published: true
---

# 目的

[ImageNet について考える (1) — Tiny ImageNet](/derwind/articles/dwd-imagenet01) で Tiny ImageNet を調べたので、実際に分類モデルを訓練してみたい。

# やること

VGG16 の転移学習ベースで訓練する。[ImageNet について考える (1) — Tiny ImageNet](/derwind/articles/dwd-imagenet01) でも触れた [ImageClassificationProject-IITK](https://github.com/ayushdabra/ImageClassificationProject-IITK) が分かりやすいので、これをベースとする。また [VGGNet and Tiny ImageNet](https://learningai.io/projects/2017/06/29/tiny-imagenet.html) という記事も参考になる部分が多かったので、一部適用している。

実装には PyTorch を用いて、val acc=0.5 程度で満足することにした。これくらいの画質で簡単なアーキテクチャで 1/2 の確率で 200 クラスの中から正解を引けるなら御の字であろう。

# データセット

`tiny-imagenet-200.zip` を展開すると

- `tiny-imagenet-200/train`
- `tiny-imagenet-200/val`

の 2 つのデータセットが見つかるが、どうも `val` のほうはそのままではラベルがすべて 0 のように見えることに後で気づいたので、`train` を split してつかうことにした[^1]。

[^1]: 少々手間ではあるが [pytorch-notebooks](https://github.com/rcamino/pytorch-notebooks/tree/master) の「Train Torchvision Models with Tiny ImageNet-200.ipynb」の方法でラベルを回復できる可能性がある。または [tinyimagenet.sh](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4) も同じことをしているのかもしれない。

[Adding dataset Tiny-Imagenet](https://github.com/pytorch/vision/issues/6127) が完了して、`torchvision` 経由で利用できるようになることを期待したい。

# モデル等の実装

まずは必要なモジュールを import する:

```python
from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torchvision import datasets, models
from torchvision.transforms import v2
import torchinfo
```

## データローダ

`torch.utils.data.random_split` を適用すると `torch.utils.data.Subset` が得られるが、これについて訓練セットではデータオーグメンテーションを適用したいが、検証セットでは適用したくない。これについて [Transforms on subset](https://discuss.pytorch.org/t/transforms-on-subset/166836) からリンクされている [Torch.utils.data.dataset.random_split](https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209) を参考にした。

データオーグメンテーションについては [VGGNet and Tiny ImageNet](https://learningai.io/projects/2017/06/29/tiny-imagenet.html) を参考にした。

```python
train_transform = v2.Compose([
    v2.RandomResizedCrop(size=64, scale=(56/64, 56/64), ratio=(1., 1.)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(saturation=(0.5, 2.0), hue=0.05),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

n_class = 200

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

batch_size = 64

dataset = datasets.ImageFolder("tiny-imagenet-200/train", transform=None)
generator = torch.Generator().manual_seed(42)
subset1, subset2 = torch.utils.data.random_split(dataset, [0.95, 0.05],
                                                 generator=generator)
trainset = MyDataset(subset1, transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
valset = MyDataset(subset2, transform=val_transform)
val_loader = torch.utils.data.DataLoader(
    valset, batch_size=batch_size, shuffle=False
)
```

## モデル

VGG16 の分類器だけを差し替える転移学習を行うので以下のようなアーキテクチャにした。基本的には [ImageClassificationProject-IITK](https://github.com/ayushdabra/ImageClassificationProject-IITK) の通りだが、Dropout の確率は分類器の初期化については [VGGNet and Tiny ImageNet](https://learningai.io/projects/2017/06/29/tiny-imagenet.html) を参考に少し手を入れた。「ReLU を使うなら He の初期化で良いかな・・・」という雑な理解だが。

本来 VGG16 は 224x224 の画像向けのものであるが、結果的には 64x64 の画像でも次元が噛み合うし、特徴量抽出をそれなりにうまく動くようなので、[ImageClassificationProject-IITK](https://github.com/ayushdabra/ImageClassificationProject-IITK) に倣ってそのようにした[^2]。

[^2]: [VGGNet and Tiny ImageNet](https://learningai.io/projects/2017/06/29/tiny-imagenet.html) でも特に書いていないのでリサイズはなしだろうか？但し、VGG16 の最後の最大プーリング層と 3 つの畳み込み層を削除して、訓練済みモデルではなくスクラッチで訓練したそうである。

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

        new_classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.5),
            nn.Linear(512, n_class),
        )

        self.vgg16.classifier = new_classifier
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.softmax(x)
        return x


def init_classifier(classifier):
    with torch.no_grad():
        for m in classifier:
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)  # He initialization
                nn.init.constant_(m.bias, 0)


net = Net()
init_classifier(net.vgg16.classifier)
net = net.to(device)
```

## 訓練ループと検証ループ

ありがちな感じだが以下のようにした。訓練ループ中にもたまにミニバッチだけでの acc を出力するようにして状況が分かるようにした。

```python
def train(net, device, train_loader, optimizer, epoch, log_interval, pred_interval):
    criterion = nn.NLLLoss()

    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if batch_idx % pred_interval == 0:
                pred = output.argmax(dim=1, keepdim=True)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    pred.eq(target.view_as(pred)).sum().item() / len(target)))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(net, device, test_loader):
    criterion = nn.NLLLoss()

    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

# 訓練

[ImageClassificationProject-IITK](https://github.com/ayushdabra/ImageClassificationProject-IITK) に倣って 2 段階方式で試した。つまり:

1. 分類器以外の層を固定して転移学習（ベースモデル作成）
2. モデル全体の層を固定解除してファインチューニング

`CyclicLR` という学習率スケジューリングを用いているようだが、そのまま使ってみることにした。

## 第一段階（ベースモデル作成）

```python
%%time

log_interval = 100
pred_interval = 300
init_epoch = 1
epochs = 10

net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.0001, max_lr=0.0006, step_size_up=1404, mode="triangular2"
)

for epoch in range(init_epoch, init_epoch + epochs):
    train(net, device, train_loader, optimizer, epoch, log_interval, pred_interval)
    scheduler.step()
    test(net, device, val_loader)

torch.save(net.to("cpu").state_dict(), "tiny-imagenet-vgg-basemodel.pt")
net = net.to(device)
```

> Train Epoch: 1 [0/95000 (0%)]	Loss: 7.145396	acc: 0.015625
> ...
> Train Epoch: 1 [89600/95000 (94%)]	Loss: 3.456964
> 
> Test set: Average loss: 0.0417, Accuracy: 2039/5000 (40.78%)
> ...
> Train Epoch: 10 [0/95000 (0%)]	Loss: 1.929856	acc: 0.437500
> ...
> Train Epoch: 10 [89600/95000 (94%)]	Loss: 2.101025
> 
> Test set: Average loss: 0.0343, Accuracy: 2353/5000 (47.06%)
> 
> CPU times: user 28min 30s, sys: 33.1 s, total: 29min 3s
> Wall time: 29min 18s

Colab の T4 でもそれほどは時間がかからなかった。本来は Early Stopping を使ったり、[TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) を使うべきだが、今回は完全にさぼった。

## 第二段階（ファインチューニング）

```python
%%time

for name, param in net.vgg16.named_parameters():
    param.requires_grad = True

log_interval = 100
pred_interval = 300
init_epoch = 1
epochs = 5

optimizer = optim.Adam(net.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.00001, max_lr=0.00006, step_size_up=1200, mode="triangular2"
)

for epoch in range(init_epoch, init_epoch + epochs):
    train(net, device, train_loader, optimizer, epoch, log_interval, pred_interval)
    scheduler.step()
    test(net, device, val_loader)

torch.save(net.to("cpu").state_dict(), "tiny-imagenet-vgg-finetuned.pt")
```

> Train Epoch: 1 [0/95000 (0%)]	Loss: 2.134282	acc: 0.421875
> ...
> Train Epoch: 1 [89600/95000 (94%)]	Loss: 2.183281
> 
> Test set: Average loss: 0.0326, Accuracy: 2489/5000 (49.78%)
> ...
> Train Epoch: 5 [0/95000 (0%)]	Loss: 1.696660	acc: 0.562500
> ...
> Train Epoch: 5 [89600/95000 (94%)]	Loss: 1.728446
> 
> Test set: Average loss: 0.0319, Accuracy: 2609/5000 (**52.18%**)
> 
> CPU times: user 15min 1s, sys: 17.2 s, total: 15min 18s
> Wall time: 15min 28s

当初の目的を達成する程度の val acc が達成できた。

# Top-1/5 性能確認

ImageNet では top-1 と top-5 を見るのが慣習のようなので、それに倣う。必要な関数を [Top k error calculation](https://discuss.pytorch.org/t/top-k-error-calculation/48815/2) から拝借する。元ネタは [examples/imagenet/main.py](https://github.com/pytorch/examples/blob/main/imagenet/main.py) のほうでも活用されているらしい[^3]。

[^3]: こちらは `100.0` を掛けていることに注意したい。

```python
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]
```

```python
net.eval()
total = 0
top1_corrects = []
top5_corrects = []

with torch.no_grad():
    for data, target in val_loader:
        batch_size = target.size(0)
        data, target = data.to(device), target.to(device)
        output = net(data)
        top1_accs, top5_accs = accuracy(output, target, topk=(1, 5))
        top1_corrects.append(top1_accs.cpu().numpy() * batch_size)
        top5_corrects.append(top5_accs.cpu().numpy() * batch_size)
        total += len(data)

print(f"Top-1 acc: {np.sum(top1_corrects) / total} / Top-5 acc: {np.sum(top5_corrects) / total}")
```

> Top-1 acc: 0.5218 / Top-5 acc: 0.7684

短時間の訓練ではあったが、まぁまぁの精度に到達しているのではないだろうか？

# まとめ

ほぼ [ImageClassificationProject-IITK](https://github.com/ayushdabra/ImageClassificationProject-IITK) の内容確認と TensorFlow から PyTorch への移植という程度の内容だが、概ね記載に近い val acc が達成できたように思う。[VGGNet and Tiny ImageNet](https://learningai.io/projects/2017/06/29/tiny-imagenet.html) のほうの記載でもそうなのだが、どうやら VGG16 ベースのアーキテクチャで頑張る場合、Top-1 acc が 55～56% くらいの検証精度くらいが出るようだ。

なお [Image Classification on Tiny ImageNet Classification](https://paperswithcode.com/sota/image-classification-on-tiny-imagenet-1) によると、Transformer と畳み込みのハイブリッドモデルの Astroformer や Transformer ベースの DeiT や Swin Transformer V2 だと 92% 以上くらいの精度まで出るようである。EfficientNet で 84% くらいらしい。
