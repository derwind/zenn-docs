---
title: "Qiskit で遊んでみる (3)"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: true
---

# 目的

よく「状態ベクトルシミュレータは数十量子ビット程度しか扱えない」という表現を聞く。ずっと深く考えずに「そんなもんなんだ」と思っていたが、散歩中に閃いたので記事にしたい。

実機の量子コンピュータでは測定をせずに量子状態をとらえることはできない。つまり「状態ベクトル」というものはシミュレータ固有のものであるのだから、「扱えない」というのは当然シミュレータを実行している古典コンピュータからくる制約である。とすれば、メモリであろう。これを確認したい。

# 実験

## 使用する量子回路

基本的な回路として以下の形のものを使う。

```python
from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(1)
backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend)
```

少し新しめの書き方をすれば以下のようになるが、どちらでも今回の実験に差はない。とにかく回路をシミュレータ上で実行することで状態ベクトルが保存されてメモリを消費する。

```python
qc = QuantumCircuit(1)
sim = Aer.get_backend('aer_simulator')
qc.save_statevector()
job = sim.run(qc)
```

## 使用メモリ量計測実験

以下のように小さな回路を実行して、次に大きな回路を実行する。絶対にこうすべきかと言われると悩むが、動作としてこれで Linux の `free` コマンドで見た使用メモリ量と大体一致する値が `mem_info` に格納されたので今回はこれで良しとする。

```python
import psutil
import time

mem_info = []
# 最大 30 量子ビットまでの回路を作成する
for n_qubits in range(1, 30+1):
    # 小さい回路を走らせてメモリを解放させてメモリ使用量を元に戻す。
    qc = QuantumCircuit(1)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    time.sleep(1) # メモリ量の安定を待つために一瞬眠らせる
    mem = psutil.virtual_memory()
    used = mem.used

    qc = QuantumCircuit(n_qubits)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    time.sleep(1) # メモリ量の安定を待つために一瞬眠らせる
    mem = psutil.virtual_memory()
    mem_info.append((n_qubits, mem.used - used))
```

結果を眺めてみると以下のようになっている。今回の実験では 14 量子ビットまではあまり大きなメモリの消費をしないことから目立った形ではなかったが 15 量子ビット以上は明らかにメモリをガンガン消費し出していたのでこれを見る。

```
mem_info
```
...
(15, 32768),
(16, 249856),
(17, 516096),
(18, 2838528),
(19, 6819840),
(20, 14905344),
(21, 32505856),
(22, 65544192),
(23, 133668864),
(24, 265822208),
(25, 536481792),
(26, 1074380800),
(27, 2150572032),
(28, 4302127104),
(29, 8603254784),
(30, 17212567552)]

# 考察

実測値は明らかに倍々ゲームでメモリを消費しているので、やるまでもないのだが念の為線形回帰にかけてみる。

```python
import numpy as np
import matplotlib.pyplot as plt

# 扱いやすいように numpy.ndarray に変換する。
# 18 量子ビット以上のケースからグラフが明らかに綺麗だったのでそこを見る。
mem_info = np.array(mem_info[17:])
# 後で対数をとるので、1 未満の値を 1 に切り上げておく。
mem_info[mem_info[:,1] < 1] = 1

X = mem_info[:, 0].reshape(-1, 1) # 後で scikit-learn を使うのでその調整。
y = mem_info[:, 1]
```

ここまでで “教師データ” の作成は終わりである。念の為可視化するといかにも指数関数と思われるグラフが出てくる。

```python
plt.scatter(X, y)
plt.show()
```

![](/images/dwd-qiskit03/001.png)

$n$ 量子ビットの状態ベクトルは $2^n$ 次元のベクトルとして表されることから、概ねこれに定数を掛けたようなグラフであろうと想像できる。よって、底 2 の対数グラフをとると以下のようにビンゴとなる。

```python
y_ = np.log2(y)

plt.scatter(X, y_)
plt.show()
```

![](/images/dwd-qiskit03/002.png)

scikit-learn で線形回帰を実行すると、以下のようにかなり良い精度でフィットする。

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X, y_)
reg.score(X, y_)
```
0.9991929550998879

回帰係数を見ると

```python
reg.coef_, reg.intercept_
```
(array([1.03193091]), 3.147254057778717)

のようになっていた。

要するに、$n$ を量子ビット数として、消費メモリ量を $y$ バイトとすると、$\log_2 y = 1.03 n + 3.14$ くらいである。ところで、$n \simeq 20$ くらいでは $1.03 n \simeq n + 1$ くらいになっている。$2^{3.14} \simeq 9$ である。よって、雑であるが $y = 18 \cdot 2^n$ くらいのメモリ使用量になっている。

# まとめ

何故状態ベクトルをシミュレータで扱う場合に数十量子ビットしか扱えないかがこれで理解できた。今回の実験では 26GB のメモリを搭載した VM を使用したが、31 量子ビットで超過したことが想像されるので偶然ではあるがギリギリであった。普通に用意できる実験環境だとこれくらいが限界であろうと思われる。

他の制限としてはよく「あまり深い回路は実機では扱えない」というものも聞く。こちらはシミュレータの古典コンピュータ上での制約ではなく、恐らく NISQ デバイスにおけるゲート適用時のノイズの累積に起因するハードウェアの制約と思われるが次回以降で調査してみたい。
