---
title: "行列積状態について考える (11) — NumPy を使わずに 50 量子ビットのもつれ状態を計算"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "numpy", "TensorNetwork"]
published: false
---

# 目的

[行列積状態について考える (9) — 100 量子ビットのもつれ状態](/derwind/articles/dwd-matrix-product09) で大量の量子ビットの行列積状態 (MPS) のもつれ状態を見た。このような量子ビット数の MPS シミュレーションにおいて、[NumPy もどきを作る (1)](/derwind/articles/dwd-mynumpy-01) で作った自作品で NumPy を置き換えたらどれくらいパフォーマンスに影響が出るのであろう？という、どうでも良いことの検証をしてみる。

とは言え、今回は非常につまらない実装の詳細によって 50 量子ビットで確認する。縮約計算用のインデックスを 50 個までしか使えないという実装上の制限を作ってしまったためで、拡張してもどうせ滅多に使わないからとりあえずはそのままで、と考えたためである。

# MPS ベース量子回路シミュレータ

[行列積状態について考える (10) — 50 量子ビットの期待値計算](/derwind/articles/dwd-matrix-product10) で用意した `TT_SVD_Vidal` その他の実装をライブラリ化したものを用いる。これは https://github.com/derwind/ttz で管理している。また、ブランチで「NumPy もどき」に置き換えている。
この弊害として $Rx(\theta)$ のような複素行列を伴う計算が実行できなくなっている。$Ry(\theta)$ は実行列で使えるので簡単な量子 2 値分類器が実装できそうな気はするが、まだ試していない。

# MPS + NumPy もどきでもつれ状態を作成する

Google Colab を用いた。

```sh
%%bash

pip install -qU "git+https://github.com/derwind/mynumpy.git@0.7"
pip install -qU "git+https://github.com/derwind/ttz.git@0.2a0.dev0"
pip uninstall -y numpy
```

で各種実装完了済みのブランチをインストールできる。念のために NumPy をアンインストールしてしまう。

> Found existing installation: numpy 1.25.2
> Uninstalling numpy-1.25.2:
>   Successfully uninstalled numpy-1.25.2

これによって、

```sh
! pip list | grep numpy
```

> mynumpy                          0.7

自作のものしか見えない状態になった。

## もつれ状態を作成して確率振幅を求める

以下のようなシミュレーションを実行し、$\ket{0}^{\otimes 50}$ と $\ket{1}^{\otimes 50}$ の確率振幅を求める。これらの絶対値の 2 乗が足して 1 になれば、他の計算基底の係数は 0 である。

```python
%%time

from ttz.mps import MPS


num_qubits = 50
mps = MPS(num_qubits)
mps.x(0)
mps.h(0)
for i in range(num_qubits - 1):
    mps.cx(i, i + 1)

amp_000 = mps.amplitude("0" * num_qubits).item()
amp_111 = mps.amplitude("1" * num_qubits).item()
print(amp_000, amp_111)
```

> (0.7071067811865475+0j) (-0.7071067811865475+0j)
> CPU times: user 176 ms, sys: 1.76 ms, total: 177 ms
> Wall time: 179 ms

このことから、

$$
\begin{align*}
\ket{\psi} = \frac{1}{\sqrt{2}} \ket{0}^{\otimes 50} - \frac{1}{\sqrt{2}} \ket{1}^{\otimes 50}
\end{align*}
$$

となっていることが分かるのだが、179 ms で計算が完了するとは思わなかった。
よくよく考えると、このケースでは MPS のノード間のもつれ量は 2 であり、小さな SVD を 50 回弱行うだけなので計算がそれほどかからないのは理解できるのだが、数秒くらいかかるだろうか？と思っていたので驚いた。完全に素の Python だけでこの計算ができているのである。

# まとめ

ただの興味本位で自作ツールを連携させて MPS シミュレーションを実行しただけだが、思った以上にちゃんと動いてくれたので良かった。実際には、特異値行列の特異値の値が複数個一致しているケースでの SVD 計算にバグがあって暫く難航したのだが、NumPy での動作と比較することで修正することができた。

何にせよ、百聞は一見に如かずということで、試して実際に見てみると学びがある。
