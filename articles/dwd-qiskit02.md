---
title: "Qiskit で遊んでみる (2)"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: false
---

# 目的

Qiskit の transpile という関数が分かりにくかったので調べてみたい。

[Introduction to Qiskit](https://qiskit.org/documentation/intro_tutorial1.html#introduction-to-qiskit) などを見ると、

```python
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator

simulator = QasmSimulator()
circuit = QuantumCircuit(2, 2)
...
compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
```

といった記述をよく見るが、この `transpile` が何をやっているのかが気になる。  
API ドキュメントで [qiskit.compiler.transpile](https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html) を調べると

> Transpile one or more circuits, according to some desired transpilation targets.

ということで何かしら “トランスパイル” するのは読み取れるがそれ以上が分からない。なので実験してみよう。

以降、基本的な import は済ませてある前提とする:

```python
from qiskit import QuantumCircuit, Aer, transpile, IBMQ
from qiskit.visualization import plot_gate_map
from qiskit.tools import job_monitor
```

# 各種回路とトランスパイルの結果

## ベル状態を作る回路

$\frac{1}{\sqrt{2}}(\ket{00} + \ket{11})$ を作る回路をトランスパイルしてみたい。[qiskit/compiler/transpiler.py#L657](https://github.com/Qiskit/qiskit-terra/blob/0.20.2/qiskit/compiler/transpiler.py#L657) に倣って、$U_3$ ゲートと $CX$ ゲートを使ったトランスパイルを実行してみる。

```python
sim = Aer.get_backend('aer_simulator')
qc = QuantumCircuit(2)
qc.h(0)
qc.x(1)
qc.cx(0, 1)
qc.measure_all()
qc = transpile(qc, sim, basis_gates=['u3', 'cx'])
qc.draw()
```

![](/images/dwd-qiskit02/001.png)

という感じで素直な結果が得られた。

## スワップを実行する回路

```python
qc = QuantumCircuit(2)
qc.swap(0, 1)
qc.measure_all()
sim = Aer.get_backend('aer_simulator')
qc = transpile(qc, sim, basis_gates=['u3', 'cx'])
qc.draw()
```

![](/images/dwd-qiskit02/002.png)

一見「これは何だ？」という結果が得られたが、有名な回路の等価性 $SWAP_{0,1} = CX_{0,1} CX_{1,0} CX_{0,1}$ が出てきている形である[^1]。この表現を後で使うので覚えておきたい。

[^1]: [More Circuit Identities](https://qiskit.org/textbook/ch-gates/more-circuit-identities.html) が参考になると思われる。

## アダマールゲートを 2 回実行する回路

```python
qc = QuantumCircuit(1)
qc.h(0)
qc.h(0)
qc.measure_all()
sim = Aer.get_backend('aer_simulator')
transpiled_qc = transpile(qc, sim)
transpiled_qc.draw('mpl')
```

![](/images/dwd-qiskit02/003.png)

ゲートが消えてしまった！これはアダマールゲート $H$ はエルミートかつユニタリであることから $H^2 = I$ となるためである。[^2]

[^2]: 今回 `basis_gates=['u3', 'cx']` を指定していないが、指定すると恒等ゲート $I$ を $U_3$ で表現してしまって見た目が悪いので敢えて省いた。

何となく “トランスパイル” の意味するところが見えてきたような気がする。続けよう。

## CX を 3 回実行する回路

### シミュレータの場合

```python
qc = QuantumCircuit(3)
qc.x(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 0)
qc.measure_all()
sim = Aer.get_backend('aer_simulator')
transpiled_qc = transpile(qc, sim, basis_gates=['u3', 'cx'])
transpiled_qc.draw('mpl')
```

![](/images/dwd-qiskit02/004.png)

特に今までと違わない “普通の” 表現が得られた。測定を

```python
result = sim.run(transpiled_qc).result()
print(result.get_counts())
```

で実行すると、`{'110': 1024}` つまり $\ket{110}$ が得られるが、理想的なシミュレータなのでこれも想定通りだ。

### 実機の場合

これが実機だと多少事情が異なってくる。今回 `ibmq_belem` を用いてみたい。[^3]

[^3]: 量子ビットの接続トポロジーと、本記事執筆時点での待ち行列の長さでこのデバイスを選んだ。

```python
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_belem')

plot_gate_map(backend, plot_directed=True)
```

![](/images/dwd-qiskit02/005.png)

という接続トポロジーで量子ビットが繋がっている。今回用意した量子回路では、CX のうちのどれか 1 つは隣接量子ビット同士にならないようにしている。さて、トランスパイルするとどうなるだろうか？

なお、`ibmq_belem` はプロパティを見ると `Basis gates: CX, ID, RZ, SX, X` なので、これを指定することにする。

```python
qc = QuantumCircuit(3)
qc.x(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 0)
qc.measure_all()
transpiled_qc = transpile(qc, backend, basis_gates=['cx', 'id', 'rz', 'sx', 'x'])
transpiled_qc.draw('mpl')
```

![](/images/dwd-qiskit02/006.png)

シミュレータの場合とは大きく異なる見た目の複雑な回路が得られた。  
赤枠で囲んだ部分は先程見た「SWAP ゲート」に相当する。$q_2$ と $q_0$ を隣接するように配置できなかったが、`ibmq_belem` の接続トポロジー上は隣接した量子ビットでないと CX を実行できないので、量子ビットを SWAP して移動させているのである。これで $q_0$ と $q_1$ の情報を入れ換えたので、測定の際も $q_1$ の横棒の先に古典ビット $c_0$ が、$q_0$ の横棒の先に古典ビット $c_1$ が控えている。

念のため測定してみよう。

```python
job = backend.run(transpiled_qc)
```

で実験ジョブを投入する。一般には待ち状態があるので即座には実行できずキューイングされる。

```
...
Job Status: job is queued (2)
...
Job Status: job has successfully run
```

でジョブの実行が完了した。

```python
result = job.result()
plot_histogram(result.get_counts())
```

![](/images/dwd-qiskit02/007.png)

NISQ デバイスなのでノイズがあり、理論上は観測されない状態も混じるが、期待する $\ket{110}$ が最頻値で観測されていることが分かる。

# まとめ

トランスパイルでは回路を実行するデバイスの状況に合わせて適切な回路へと組み換えられることが分かった。シミュレータの場合だと理想的で何でもありなのでオリジナルの回路とほぼ同じものが出てくるが、実機の場合だと接続トポロジーの制約があるので、時に予想したよりも複雑な回路になることがあるし、結果として回路の深さも変化してくる。

簡単な遊びではあったがちゃんと確認できて良かった。
