---
title: "cuQuantum で遊んでみる (1) — GTC2022 より"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "poem", "Python"]
published: true
---

# 目的

GTC2022 で [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) について簡単に学んだので少し記事にまとめたいというもの。

具体的には [Scaling Quantum Circuit Simulations with cuQuantum for Quantum Algorithms [A41102]](https://www.nvidia.com/gtc/session-catalog/#/session/1655237858109001ej5A) を視聴した。

雑記みたいなものとして書いているので、途中計算とは無茶苦茶であまり意味合いまで深掘りしていないがそれは今後考えることにする。また、オマケとして大量の量子ビットを持つ回路の計算を軽く見てみる。

# cuQuantum って何？

文献 [CQ] より:

> **NVIDIA cuQuantum** is an SDK of optimized libraries and tools for accelerating quantum computing workflows. With NVIDIA Tensor Core GPUs, developers can use cuQuantum to speed up quantum circuit simulations based on state vector and tensor network methods by orders of magnitude. (**NVIDIA cuQuantum**は、量子コンピューティングのワークフローを加速するために最適化されたライブラリとツールのSDKです。NVIDIA Tensor Core GPU を使用することで、開発者は cuQuantum を使用して、状態ベクトルおよびテンソルネットワーク手法に基づく量子回路シミュレーションを桁違いに高速化することができます。)

まぁ、量子コンピュータの凄い爆速の GPU シミュレータということ。テンソルネットワークについては例えば文献 [TN] 参照。

# cuQuantum をインストールしよう

手元の環境が CUDA 11.0 なので、それに合わせた記述にする:

```sh
pip install cuquantum cuquantum-python cupy-cuda110
```

で良さそう。因みに以下で触れる内容は、`cuquantum-python` 22.7.0 以降の機能[^1]を使うのだが、これらは Python 3.8+ でないと使えない。一方、Google Colaboratory の Jupyter カーネルは 3.7.12 のようで `cuquantum-python` 22.5.0 までしか入らない。つまり API を叩けないので、今回は Colab を使うことは諦める。

なお、実験環境は NVIDIA の T4 を使っており、そこまで大それたことはしていない。なお、バックエンドには `CuPy` を用いるので、以下で出てくるテンソルの具体的な型は [cupy.ndarray](https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html) である。

[^1]: 具体的には [cuquantum.CircuitToEinsum](https://docs.nvidia.com/cuda/cuquantum/python/api/generated/cuquantum.CircuitToEinsum.html) を使う。

# 今から何をするか？

Qiskit の量子回路をテンソルネットワークの形に落とし込んで縮約計算して、回路の出力状態に関する確率振幅を求めてみる。

## メリット

- 状態ベクトルを保持しないので、$n$ 量子ビットにおける長さ $2^n$ のベクトルを保持しながら行列計算をする必要がない。つまりメモリが効率的。

## デメリット

- ただのスパコン的な世界の計算であって量子力学の恩恵を得られないので、例えば重ね合わせとかが活用できず、全確率振幅を一気に取得できない。たぶん。

# 早速簡単なサンプルで見てみる

まずはお約束で必要なものを import する。

```python
from qiskit import QuantumCircuit, QuantumRegister
from cuquantum import CircuitToEinsum, contract
import numpy as np
```

そして量子回路を構築して `cuQuantum` に放り込むと、

```python
qc = QuantumCircuit(1)
qc.x(0)
converter = CircuitToEinsum(qc)
expr, operands = converter.amplitude(bitstring='1')
print(expr)
for op in operands:
    print(op)
    print()
print(contract(expr, *operands))
```

以下のような出力を得る。

```
a,ba,b->
[1.+0.j 0.+0.j]

[[0.+0.j 1.+0.j]
 [1.+0.j 0.+0.j]]

[0.+0.j 1.+0.j]

(1+0j)
```

縮約計算 `contract(expr, *operands)` の結果は `(1+0j)` つまり 1 だという結果だが、これはどういうことであろうか？

## 手計算してみよう

`operands[0]`, `operands[1]`, `operands[2]` を数式で書くと以下である:

$$
\begin{align*}
\psi = \begin{pmatrix} 1 \\ 0 \end{pmatrix},\ \xi = \begin{pmatrix} 0 \\ 1 \end{pmatrix},\ X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
\end{align*}
$$

`expr = 'a,ba,b->'` であったので、数式として書くと、

$$
\begin{align*}
\psi_{a} X_{ba} \xi_{b} &= \sum_{a,b} \psi_{a} X_{ba} \xi_{b} \\
&= (\xi_0\ \ \xi_1) \begin{pmatrix} x_{00} & x_{01} \\ x_{10} & x_{11} \end{pmatrix} \begin{pmatrix} \psi_0 \\ \psi_1 \end{pmatrix} \\
& = (0\ \ 1) \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = 1
\end{align*}
$$

である。もうちょっとそれらしく書くと $\braket{\xi|X|\psi}$ を計算していることになる。よく教科書とかには、「$X$ ゲートだけの量子回路があるとして、状態 $\ket{1}$ の確率振幅を求めると $p_{X\ket{0}}(\ket{1}) = 1$ になります」とか書いてある計算そのものである。

そんなに怖いことをしているわけではなさそうだと言う事がわかった。次はもう少し複雑な回路を考えてみよう。

# エンタングル状態を考えてみる

いわゆる Bell 状態は以下のような回路で作られる:

```python
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
```

![](/images/dwd-cuquantum01/001.png)

これを `cuQuantum` に入れてみよう:

```python
converter = CircuitToEinsum(qc, dtype='complex128', backend='cupy')
for bitstring in ['00', '01', '10', '11']:
    expr, operands = converter.amplitude(bitstring=bitstring)
    amplitude = contract(expr, *operands)
    print(amplitude)
```

```
(0.7071067811865475+0j)
0j
0j
(0.7071067811865475+0j)
```

となった。何が言いたいかと言うと、状態 $\ket{00}$, $\ket{01}$, $\ket{10}$, $\ket{11}$ の確率振幅を計算することで、回路の出力が $\frac{1}{\sqrt{2}}(\ket{00} + \ket{11})$ になっていることが分かりましたよと言いたいのである。

## またまた手計算してみよう

今回は状態 $\ket{11}$ の確率振幅だけ求めてみる。

```python
expr, operands = converter.amplitude(bitstring='11')
print(expr)
for op in operands:
    print(op)
    print()
```

```
a,b,ca,debc,e,d->
[1.+0.j 0.+0.j]

[1.+0.j 0.+0.j]

[[ 0.70710678+0.j  0.70710678+0.j]
 [ 0.70710678+0.j -0.70710678+0.j]]

[[[[1.+0.j 0.+0.j]
   [0.+0.j 0.+0.j]]

  [[0.+0.j 0.+0.j]
   [0.+0.j 1.+0.j]]]
```

かなり気が滅入る結果が出てきた。また数式で書いてみよう:

$$
\begin{align*}
\psi &= \begin{pmatrix} 1 \\ 0 \end{pmatrix},\ \varphi = \begin{pmatrix} 1 \\ 0 \end{pmatrix},\ H = (h_{ij}) = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \\
S &= (s_{ij}^{k\ell}) = \Bigg[ \left[ \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix} \right], \left[ \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} \right] \Bigg], \\
\xi &= \begin{pmatrix} 0 \\ 1 \end{pmatrix},\ \eta = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
\end{align*}
$$

あまり良い記号とも言えないが、このような記号にしてみた。`expr = 'a,b,ca,debc,e,d->'` なのでこの計算を数式で書くと以下のようになる:

$$
\begin{align*}
\psi_a \varphi_b h_{ca} s_{bc}^{de} \xi_e \eta_d &= \sum_{a,b,c,d,e} \psi_a \varphi_b h_{ca} s_{bc}^{de} \xi_e \eta_d \\
&= \sum_{a,b,c} \psi_a \varphi_b h_{ca} \sum_{de} s_{bc}^{de} \xi_e \eta_d \\
&= \! \sum_{a,b,c} \psi_a \varphi_b h_{ca} \! \left[ \! \begin{pmatrix} 1 \!&\! 0 \\ 0 \!&\! 0 \end{pmatrix} \! \eta_0 \xi_0 \!+\! \begin{pmatrix} 0 \!&\! 0 \\ 0 \!&\! 1 \end{pmatrix} \! \eta_0 \xi_1 \!+\! \begin{pmatrix} 0 \!&\! 0 \\ 1 \!&\! 0 \end{pmatrix} \! \eta_1 \xi_0 \!+\! \begin{pmatrix} 0 \!&\! 1 \\ 0 \!&\! 0 \end{pmatrix} \! \eta_1 \xi_1 \! \right] \\
&= \sum_{a,b,c} \psi_a \varphi_b h_{ca} \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} = \sum_{b,c} \varphi_b \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} \sum_{a} h_{ca} \psi_a = \frac{1}{\sqrt{2}}
\end{align*}
$$

という感じである。CX ゲートが何でこうなっちゃったの？というのは今はまだすっきり書けなくてただ計算を追っただけであるが、結果的には $\braket{\varphi| 0}\braket{1|H|0}$ を計算しているような感じになってしまった[^2]。ちゃんと追いかけていないのだが、文献 [BQ] における **CNOT のテンソル分解**が上で書いた 4 階のテンソル $S$ として出てきていると思うのだが、宿題ということにして今回は濁しておく・・・。

[^2]: 他の状態の確率振幅を求める場合には、内側の $\ket{0}\bra{1}$ の部分が変わってくるはず。

ここで言いたいことは、エンタングルメントとかの複雑な量子回路計算も、**ただのテンソル積の計算になってしまった**ということである。

# 大量の量子ビットの回路

以前の記事 [Qiskit で遊んでみる (3)](/derwind/articles/dwd-qiskit03) では、

> 何故状態ベクトルをシミュレータで扱う場合に数十量子ビットしか扱えないかがこれで理解できた。今回の実験では 26GB のメモリを搭載した VM を使用したが、31 量子ビットで超過したことが想像されるので偶然ではあるがギリギリであった。普通に用意できる実験環境だとこれくらいが限界であろうと思われる。

といったことを書いて締め括っていた。今回は何と 200 量子ビットに挑戦してみたい[^3]。こんな量子ビット数だと以前の記事の通りの設定で確率振幅を求めようとしたら完全に破綻していたはずである。

[^3]: 特に深い意味はなくて、前回の記事より十分に多い量子ビット数という程度。

```python
n_qubits = 200

qr = QuantumRegister(n_qubits, 'q')
qc = QuantumCircuit(qr)
qc.h(qr[:]);
```

回路図は縦に長くて、H ゲートが各入力量子ビットの後に置かれているという素っ気ないものなので割愛する。と言うより縦長過ぎてスクリーンショットを撮る気が起こらない。そして、

```python
converter = CircuitToEinsum(qc, dtype='complex128', backend='cupy')
bitstring = '0' * n_qubits
expr, operands = converter.amplitude(bitstring=bitstring)
amplitude = contract(expr, *operands)
```

で、状態 $\ket{000\cdots 000}$ に対する確率振幅が求まっている・・・はずである。

```python
amplitude = contract(expr, *operands)
print(np.isclose(amplitude, np.sqrt(1/2**n_qubits)))
print(amplitude)
```

```
True
(7.888609052209983e-31+0j)
```

ということで、見事に確率振幅 $\sqrt{\frac{1}{2^{200}}}$ が求まった。

# まとめ

量子回路計算について、テンソルネットワークの形で GPU 上でテンソル計算ができる cuQuantum を使ってみた。これだけだと何でテンソルネットワークが出てくるのか？とか、何が嬉しいのか？という部分がまったく記述できていないのだが、この記事は「cuQuantum を使う」ことだけをスコープとしているので、それ以外のことについてはまた別途考えることにする・・・。とりあえず今回は、**200 量子ビットの回路で確率振幅が求めら得れて嬉しい！** で締め括る・・・。

なお同じようなことであれば、以前に触れた [行列積について考える](/derwind/articles/dwd-matrix-product) の行列積に概念に基づき、[Matrix product state simulation method](https://qiskit.org/documentation/tutorials/simulators/7_matrix_product_state_method.html) を参考に Qiskit の MPS シミュレータを使って以下のようにすれば測定はできるのだが、$2^{200}$ ショットなどの規模で打たないとならなくなるので、ここから確率振幅を求めるのは現実的ではなさそうに感じるがどうなのだろうか・・・。

```python
qc.measure_all()
simulator = AerSimulator(method='matrix_product_state')

tcirc = transpile(qc, simulator)
result = simulator.run(tcirc).result()
counts = result.get_counts(0)
```

# 参考文献

[CQ] [cuQuantum](https://developer.nvidia.com/cuquantum-sdk)
[TN] [Tensor Network](https://tensornetwork.org/)
[BQ] 湊雄一郎, [CNOTゲートの分解](https://blueqat.com/yuichiro_minato2/c6a42d7a-2bd6-43a7-915d-1b90688c474c), 2022
