---
title: "ImageNet について考える (3) — Tiny ImageNet の分類の説明可能性とモデル圧縮"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "ImageNet", "PyTorch", "TensorNetwork"]
published: true
---

# 目的

[ImageNet について考える (2) — Tiny ImageNet の分類](/derwind/articles/dwd-imagenet02) で Tiny ImageNet の分類モデルを訓練して、検証精度 52% 程度の分類器を得た。特に嬉しいのはモデルの全体を固定解除した上でのファインチューニングによって、**ImageNet というよりは全体的に Tiny ImageNet 用のモデルに特化**していることである。畳み込み層も Tiny ImageNet の特徴をとらえていることであろう。

今度はこのモデルに対して、

1. [ニューラルネットの説明可能性について考える (1) — Grad-CAM](/derwind/articles/dwd-grad-cam01) でやって Grad-CAM を用いて、この分類器がどういう根拠でそのクラスを推論するのかを見てみたい。
2. 更に欲張って、[行列積状態について考える (5) — ニューラルネットワークのモデル圧縮](/derwind/articles/dwd-matrix-product05) により行列積状態を用いた枝刈りを実行する。

枝刈りの結果、Grad-CAM による説明力は影響を受けるのだろうか？

# 準備

追加で必要なモジュールのインストールとセットアップを行う。枝刈り用のテンソルネットワークツールはお手製のものを使う。実は実験中にバグに気づいてしまい、今回一部枝刈りを弱めている・・・。内容的には [arXiv:1509:06569 Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569) の実装である。

```sh
%%bash

pip install -qU torchinfo
pip install -qU git+https://github.com/jacobgil/pytorch-grad-cam.git
pip install -qU git+https://github.com/derwind/ttz.git@0.2
```

そして import する:

```python
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image
)

from ttz.tt import TT_SVD, TTLayer
```

以下、変数類は [ImageNet について考える (2) — Tiny ImageNet の分類](/derwind/articles/dwd-imagenet02) からの継続でオンメモリで存在しているとする。

# Grad-CAM による分類の根拠の説明

クラス 0 は 'goldfish, Carassius auratus' なのであるが、これを Grad-CAM で見てみたい。実装はほぼ [ニューラルネットの説明可能性について考える (1) — Grad-CAM](/derwind/articles/dwd-grad-cam01) のコピペである。

```python
img, lbl = valset[1894]

# モデルの推論結果と GT ラベルの比較
with torch.no_grad():
    img = img.unsqueeze(0)
    img = img.to(device)
    output = net(img)
    print("label:", torch.argmax(output).cpu().numpy(), "true label:", lbl)

# Grad-CAM の適用
img = img.cpu().numpy() * [
    [[[0.229]], [[0.224]], [[0.225]]]] + [[[[0.485]], [[0.456]], [[0.406]]]
]
img = (img.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

img = img[0]

img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
input_tensor = input_tensor.to(device)

# Grad-CAM を適用するので、勾配計算できるようにモデルの固定を解除する。
net = net.train()
# unfreeze parameters
for name, param in net.vgg16.named_parameters():
    param.requires_grad = True

targets = [ClassifierOutputTarget(0)]  # ラベル 0
target_layers = [net.vgg16.features[29]]
with GradCAM(model=net, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
im = Image.fromarray(images)
# 64x64 だと小さいので拡大表示する。
display(im.resize((im.size[0] * 2, im.size[1] * 2)))
```

> label: 0 true label: 0

![](/images/dwd-imagenet03/001.png)

推論結果は GT ラベルと一致していて、根拠としては金魚の本体によるものであることが分かる。

他の例も見よう。クラス 31 は 'Persian cat' であるが、これのサンプルも見てみよう。上記と同様の実装でインデックスだけ変えて以下を得る[^1]。

[^1]: `1194` が該当するインデックスであった。

> label: 31 true label: 31

![](/images/dwd-imagenet03/002.png)

なんとなく猫を見ているようである。

次に失敗例も見ておこう。

> label: 30 true label: 31

![](/images/dwd-imagenet03/003.png)

ラベルがずれた時は変な場所を見てしまったようだ。ラベル 30 は 'tabby, tabby cat' で惜しいと言えば惜しい気もするが、トラ猫 (tabby cat) ではないのだ[^2]。元々 52% 程度の精度までしか鍛えてないので仕方ない気もする。

[^2]: ホットスポット的には猫の外だけど、やはり猫の部分も見て、その色味から tabby cat と推定したような気も・・・。

網羅的ではないが、以上から正しい推論結果の場合、それっぽい場所を見た上での判断になっているのではないかな？と思えた、ことにする・・・。

# 行列積状態を用いた枝刈り

### 線型層の TT-層への変換

まず、現状のモデルの状態を見たい:

```python
torchinfo.summary(net, input_size=(1, 3, 64, 64))
```

> ...
> \====================================================
> Total params: 27,927,560
> Trainable params: 27,927,560
> Non-trainable params: 0
> Total mult-adds (G): 1.27
> \====================================================
> Input size (MB): 0.05
> Forward/backward pass size (MB): 8.87
> Params size (MB): 111.71
> Estimated Total Size (MB): 120.62
> \====================================================

120MB くらいのモデルだということだ。

以下で線型層を TT-層に変換する。内容的には [行列積状態について考える (5) — ニューラルネットワークのモデル圧縮](/derwind/articles/dwd-matrix-product05) にごちゃごちゃ書いている通りで、連続的に SVD を行い、都度小さい特異値部分を捨てるという形である。分類器の最終層 `classifier[8]` だけツールのバグでうまく変換できなかったので今回諦めた。ツールを改修したいと思う・・・。

以下で、2 つの線型層を TT-そうに置換する。

```python
classifier = net.vgg16.classifier  # 元の分類器

tt_layer0 = TTLayer.from_linear_layer(
    [2**4 * 7, 2**5 * 7], [2**4, 2**5], classifier[0],
)
tt_layer4 = TTLayer.from_linear_layer(
    [2**4, 2**5], [2**4, 2**5], classifier[4],
)
# 2 つの線型層を TT-層に置き換える
new_classifier = nn.Sequential(
    tt_layer0,
    classifier[1],
    classifier[2],
    classifier[3],
    tt_layer4,
    classifier[5],
    classifier[6],
    classifier[7],
    classifier[8],
)

# 分類器以外を固定
for name, param in net.vgg16.named_parameters():
    if not name.startswith("classifier"):
        param.requires_grad = False

# 分類器を差し替え
net.vgg16.classifier = new_classifier
net.to(device)
```

モデルの状態を確認する:

```python
torchinfo.summary(net)
```

> ...
> \====================================================
> Total params: 28,503,560
> Trainable params: 13,788,872
> Non-trainable params: 14,714,688
> \====================================================


「Total params: 27,927,560」から「Total params: 28,503,560」に増えてしまったがこれは仕方ない。単純に行列積状態に置き換える行為はパラメータを増やしてしまう。一般の行列の SVD が要素数を増やすのと同様だ。

念のため検証ループを回すと:

```python
test(net, device, val_loader)
```

> Test set: Average loss: 0.0319, Accuracy: 2609/5000 (52.18%)

特に精度に目立った変化はない。数学的には等価な変換なのでそれはそうなのである。

### TT-層の枝刈り

以下のように大きい結合次元の部分の次元を絞って枝刈りしてみた。この辺は完全に trial & error でやったので、特にマル秘の計算式はない。それとも知らないだけど、どこかにあるんだろうか？

```python
# TT-層の結合次元を下げる
tt_layer0_pruned = TTLayer.from_linear_layer(
    [2**4 * 7, 2**5 * 7], [2**4, 2**5], classifier[0],
    bond_dims=[16, 100, 150]
)
tt_layer4_pruned = TTLayer.from_linear_layer(
    [2**4, 2**5], [2**4, 2**5], classifier[4],
    bond_dims=[16, 100, 32]
)

new_classifier_pruned = nn.Sequential(
    tt_layer0_pruned,
    classifier[1],
    classifier[2],
    classifier[3],
    tt_layer4_pruned,
    classifier[5],
    classifier[6],
    classifier[7],
    classifier[8],
)

# 分類器以外を固定
for name, param in net.vgg16.named_parameters():
    if not name.startswith("classifier"):
        param.requires_grad = False

# 分類器を差し替え
net.vgg16.classifier = new_classifier
net.to(device)
```

現状のモデルを見てみよう:

```python
torchinfo.summary(net)
```

> ...
> \====================================================
> Total params: 16,689,096
> Trainable params: 1,974,408
> Non-trainable params: 14,714,688
> \====================================================

「Total params: 27,927,560」から「Total params: 28,503,560」に増えたものが「Total params: 16,689,096」で結構減った。オリジナルの 60% 程度のパラメータ数になった。

そこで不安なのが精度だ・・・。

```python
test(net, device, val_loader)
```

> Test set: Average loss: 0.0335, Accuracy: 2478/5000 (49.56%)

52.18% から少し下がった・・・。

## ファインチューニング

ところがこれくらいの低下なら、特徴量抽出器を固定して、分類器だけ学習可能な状態で “転移学習” のノリでファインチューニングすれば精度が回復する可能性がある。

[ImageNet について考える (2) — Tiny ImageNet の分類](/derwind/articles/dwd-imagenet02) の「第一段階（ベースモデル作成）」と同じ内容で分類器だけ再度訓練しよう。僅かな精度低下なので今回は 1 epoch だけ行う。

```python
%%time

log_interval = 100
pred_interval = 300
init_epoch = 1
epochs = 1

net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.0006, step_size_up=1404, mode="triangular2")

for epoch in range(init_epoch, init_epoch + epochs):
    train(net, device, train_loader, optimizer, epoch, log_interval, pred_interval)
    scheduler.step()
    test(net, device, val_loader)

torch.save(net.to("cpu").state_dict(), "tiny-imagenet-vgg-tt-finetuned.pt")
```

> ...
> Test set: Average loss: 0.0323, Accuracy: 2571/5000 (51.42%)
> 
> CPU times: user 3min 7s, sys: 6.73 s, total: 3min 14s
> Wall time: 4min 16s

うまいこといって、元の 52.18% には劣るが 51.42% まで回復した。

# 分類の根拠の説明への影響は？

モデルサイズが 60% 程度になってしまったことで、Grad-CAM がどう影響を受けているか気になるところである。ところが結論としては、たぶん大きな影響はなさそう・・・という状態である。

以下が上記で試した画像での推論の根拠であるが、パッと見た目の違いはない。

> label: 0 true label: 0

![](/images/dwd-imagenet03/004.png)

> label: 31 true label: 31

![](/images/dwd-imagenet03/005.png)

つまり、モデルサイズは大きく低下（したことにする）できた上に、説明の根拠にもほぼ影響はなさそうだという状態である。

最後にモデルサイズをディスク上で確認しよう:

```sh
! ls -l *.pt
```

> -rw------- 1 root root 111385328 Jun  3 14:02 tiny-imagenet-vgg-basemodel.pt
> -rw------- 1 root root 111734128 Jun  4 15:29 tiny-imagenet-vgg-finetuned.pt
> -rw------- 1 root root  66782386 Jun  4 16:47 tiny-imagenet-vgg-tt-finetuned.pt

確かに一回り小さいデータサイズ（約 60%）になっている。

# まとめ

今回、以下を確認してわりと良い結果を得たと思う。

1. モデルの推論の根拠の可視化
2. モデルサイズの大幅な削減
3. モデルサイズ削減後の推論の根拠の可視化

枝刈りのやりかたやモデルのアーキクチャにもよるのかもしれないが、特徴量抽出器の本体は無傷な状態で削減を実行したこともあり、推論の根拠も大きな変化がなさそうで良かった[^3]。

[^3]: 予想より大幅に良かったので本当に良かった。

これで Tiny ImageNet は概ね遊びきった・・・ような気がする。後は Transformer 系でどれくらい検証精度が出せるかを追い込むとかなんだろうか・・・。

その前に、枝刈り用のテンソルネットワークツールの修正をしなくては・・・。
