---
title: "Blueqat で遊んでみる (1)"
emoji: "🐱"
type: "tech"
topics: ["blueqat", "Qiskit", "量子コンピュータ", "ポエム", "Python"]
published: false
---

# 目的

「量子コンピュータ」を勉強するにあたって流石に Qiskit べったりで勉強するのも宜しくないので、ディープラーニングを勉強する時に TensorFlow と PyTorch で勉強したように[^1]、blueqat SDK も使ってみたい。

[^1]: 結局は PyTorch ばかり使っている。

# blueqat SDK とは？

https://github.com/Blueqat/Blueqat によると

> A Quantum Computing SDK

とのこと。[blueqat cloudで始める無料量子コンピュータプログラミング01（1量子ビットの計算）](https://blueqat.com/yuichiro_minato2/6d3cf2e7-4fdd-4aa5-b211-0e039ded6967) の通りに blueqat cloud にサインアップした「Notebook」を起動したら良い。Jupyter Lab が起動するのでお馴染みの UI である・・・。なお、カーネルは Python 3.8.8 が使えた。

# 題材

Qiskit texobook の [Introduction](https://qiskit.org/textbook/ch-states/introduction.html) の量子回路を題材にする。X ゲートと CX ゲートと Toffoli (CCX) ゲートがあって良さそうだったので。

# 実装例

## Qiskit で書いてみる

そのまま書き下すだけだが、例えば以下のように書ける。

```python
from qiskit import QuantumCircuit, execute, Aer

qc = QuantumCircuit(4)

qc.x(0)
qc.x(1)
qc.barrier()
qc.cx(1, 2)
qc.cx(0, 2)
qc.ccx(0, 1, 3)
qc.measure_all()

backend = Aer.get_backend('aer_simulator')
results = execute(qc, backend=backend, shots=1024).result()
answer = results.get_counts()

print(answer)
```
{'1011': 1024}

シミュレータなので、手計算で求めた状態ベクトルに対応する確率振幅がノイズなしで得られる。

## blueqat で書いてみる

売りの 1 つだと思うがメソッドチェーンを用いてゲートの適用を 1 行で書けるのと、もろもろ省略できるので以下のようになる:

```python
from blueqat import Circuit

c = Circuit()
c = c.x[0].x[1].cx[1,2].cx[0,2].ccx[0,1,3]

results = c.m[:].run(shots=1024)
print(results)
```
Counter({'1101': 1024})

少し注意が必要なのは、Qiskit では量子ビット `q0` の観測値が右のほうに現れるようであるが、blueqat では左に現れるようであるということである。本質的な差ではないので、そういうものと思うしかないだろう。

# まとめ

そのままなので特にまとめるような内容もないが、感想を 1 つあげると blueqat で書くとコンパクトになるなということである。blueqat のほうはドキュメントは英語の状態のものが多く、また公式ドキュメントくらいしか参考文献がなさそうだけど、それはそれで使いながら学んだら良いのかもしれない。
