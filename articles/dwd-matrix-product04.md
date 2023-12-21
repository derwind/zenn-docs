---
title: "行列積状態について考える (4)"
emoji: "⛓"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "PyTorch"]
published: true
---

# 目的

[行列積状態について考える (3)](/derwind/articles/dwd-matrix-product03) の続きとして、ニューラルネットワークが Tensor-Train っぽいコンポーネントを持つ場合に学習プロセスを実行してみたい、ということをやってみる。

以下で扱う実装に特に深い意味はないが、訓練が思ったような形で動作するのかを知りたいという感じである。

# 行列の TT-分解

[行列積状態について考える (3)](/derwind/articles/dwd-matrix-product03) より、行列 $A = ( A(i_1,i_2) )_{1 \leq i_1 \leq r_1, 1 \leq i_2 \leq r_2}$ について、

$$
\begin{align*}
A(i_1,i_2) = \sum_{\alpha_0,\alpha_1,\alpha_2} G_1(\alpha_0,i_1,\alpha_1) G_2(\alpha_1,i_2,\alpha_2)
\end{align*}
$$

で $1 \leq i_k \leq n_k$、$\alpha_k \in \{1, \cdots, r_k\}$ であったが、境界条件 $\alpha_0 = \alpha_2 = 1$ であるので、これは単に $n_1 \times r_1$-行列と $r_1 \times n_2$-行列の積に過ぎない。とは言え、これくらい簡単なほうが動作を見やすいのでこれで試してみる。

# 実装

PyTorch を使う。動作検証用には、[MNIST の訓練コード](https://github.com/pytorch/examples/blob/main/mnist/main.py)をベースに動きを見る。

なお、PyTorch は以下のバージョンを用いた。

```sh
$ pip list | grep torch
torch                    2.1.1+cu118
torchaudio               2.1.1+cu118
torchvision              0.16.1+cu118
```

## 必要なモジュールの import

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
```

で必要なモジュールを import する。

## Tensor-Train っぽい雰囲気のネットワークの定義

次に、何となく行列を Tensor-Train 分解したっぽい雰囲気のネットワークを作成する。

```python
class TTNet(nn.Module):
    def __init__(self, bond_dim=50):
        super().__init__()
        self.tt1 = nn.parameter.Parameter(torch.rand(28 * 28, bond_dim))
        self.tt2 = nn.parameter.Parameter(torch.rand(bond_dim, 10))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.einsum("ni,ia,aj->nj", x, self.tt1, self.tt2)
        output = F.log_softmax(x, dim=1)
        return output

    def parameters(self, recurse: bool = True):
        return [self.tt1, self.tt2]
```

※ **ここから下はほぼほぼサンプルコードのまま**なので、脳内で実装できてしまう程度と思う。

## 訓練およびテストループ

ほぼサンプルコードのままで実装する。

```python
def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

## パラメータ設定

ほぼサンプルコードのままで実装する。

```python
batch_size = 64
test_batch_size = 1000
# epochs = 14
epochs = 2
lr = 1.0
gamma = 0.7
no_cuda = False
no_mps = False
dry_run = False
seed = 1
# log_interval = 10
log_interval = 200
save_model = False
```

# 訓練実行

`TTNet` インスタンスを作って、普通に訓練を回す。

```python
torch.manual_seed(seed)

use_cuda = not no_cuda and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('~/.pytorch/MNIST', train=True, download=True,
                   transform=transform)
dataset2 = datasets.MNIST('~/.pytorch/MNIST', train=False,
                   transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = TTNet().to(device)
optimizer = optim.Adam(model.parameters())

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch, log_interval, dry_run)
    test(model, device, test_loader)

if save_model:
    torch.save(model.state_dict(), "mnist_nn.pt")
```

```
Train Epoch: 1 [0/60000 (0%)]	Loss: 96.103439
Train Epoch: 1 [12800/60000 (21%)]	Loss: 3.628413
Train Epoch: 1 [25600/60000 (43%)]	Loss: 1.324179
Train Epoch: 1 [38400/60000 (64%)]	Loss: 1.486694
Train Epoch: 1 [51200/60000 (85%)]	Loss: 1.519152

Test set: Average loss: 1.0253, Accuracy: 8630/10000 (86%)

Train Epoch: 2 [0/60000 (0%)]	Loss: 1.644991
Train Epoch: 2 [12800/60000 (21%)]	Loss: 0.752575
Train Epoch: 2 [25600/60000 (43%)]	Loss: 1.245802
Train Epoch: 2 [38400/60000 (64%)]	Loss: 0.660840
Train Epoch: 2 [51200/60000 (85%)]	Loss: 1.099504

Test set: Average loss: 0.7930, Accuracy: 8832/10000 (88%)
```

特に意味はないネットワークだけど、最初に損失が一気に減って、2 epoch 目で僅かに精度が上がったので、訓練は回せているようだ。

# `torch.einsum` の `backward` ?

`torch.einsum` で誤差逆伝播動くの？って気持ちは少しあったのだが、[Automatic differentation for pytorch einsum](https://discuss.pytorch.org/t/automatic-differentation-for-pytorch-einsum/112504) を見た感じではうまく計算できるようなので、気にせずに実装して、実際動いているみたいだと感じた。

ただ、あまりに古い PyTorch の場合には [Einsum problem in Pytorch 0.4](https://discuss.pytorch.org/t/einsum-problem-in-pytorch-0-4/17877) にあるように不具合もあったようだ。

# まとめ

すごく雑な実装ではあったが、何かしら TT-分解が挟まったようなニューラルネットワークができた場合に、普通に書いてやることで訓練を実行できることが確認できたように思う[^1]。

[^1]: 例えば訓練済みのニューラルネットワークがある場合に、その中の全結合層を何かしらの方法で TT-分解した場合に、ファインチューニングできるだろうか？ということを想定している。
