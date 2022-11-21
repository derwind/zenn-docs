---
title: "Qiskit で遊んでみる (8) — Qiskit Aer GPU"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "poem", "Python"]
published: true
---

# 目的

[Qiskit で遊んでみる (7) — Qiskit Aer GPU](/derwind/articles/dwd-qiskit08) に引き続き、Ubuntu 環境で [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) の GPU 対応ビルド、とりわけ [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) 対応をビルドしたものを評価したい。

コンテンツはこういう場合に個人的によく使っているグローバーのアルゴリズムの回路を使うことにし、[cuQuantum で遊んでみる (2) — グローバー探索アルゴリズム](/derwind/articles/dwd-cuquantum02) をベースにしたい。

# 実験内容

- 実験環境は NVIDIA T4 x1 のマシンとする。
- 18/23/25 量子ビットの 3 パターンでグローバーのアルゴリズムを解かせて、wall time を比較したい。
- CPU についてはある程度の量子ビット数からは目に見えて時間がかかるので、18 量子ビットの時のみ試す。

# 18 量子ビットの回路

```python
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate
%matplotlib inline

def revserse_phase(qc: QuantumCircuit, state: str):
    qubits = []
    for i, digit in enumerate(state[::-1]):
        if digit == '0':
            qubits.append(i)
    if qubits:
        qc.x(qubits)
    # MCZ start
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits)))
    qc.h(n_qubits - 1)
    # MCZ end
    if qubits:
        qc.x(qubits)

def define_oracle(solutions):
    # Create the oracle with two solutions: |101> and |111>
    qreg = QuantumRegister(n_qubits, 'qr')
    oracle = QuantumCircuit(qreg)

    for sol in solutions:
        revserse_phase(oracle, sol)

    return oracle

def define_diffuser(n_qubits):
    qreg = QuantumRegister(n_qubits, 'qr')
    diffuser = QuantumCircuit(qreg)
    diffuser.h(qreg[:])
    diffuser.x(qreg[:])
    # MCZ start (HXH = Z)
    diffuser.h(qreg[n_qubits - 1])
    diffuser.append(MCXGate(n_qubits - 1), list(range(n_qubits)))
    diffuser.h(qreg[n_qubits - 1])
    # MCZ end
    diffuser.x(qreg[:])
    diffuser.h(qreg[:])

    return diffuser

solutions = ['101100111000111011', '110001110011000111']
assert len(solutions[0]) == len(solutions[1])
n_qubits = len(solutions[0])
print(f'{n_qubits=}')

oracle = define_oracle(solutions)
#oracle.draw('mpl')
```

> n_qubits=18

```python
diffuser = define_diffuser(n_qubits)
#diffuser.draw('mpl')

N = 2**n_qubits
angle = np.arcsin(np.sqrt(len(solutions) / N))
counts = int((np.pi/2 - angle) / (2*angle) + 0.5)
#print(f'{angle=}, {np.pi/2=}, {counts=}')

qreg = QuantumRegister(n_qubits, 'qr')
grover = QuantumCircuit(qreg)
# initialize |s>
grover.h(qreg[:])
for _ in range(counts):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)
#grover.draw('mpl')

print(len(grover))
```

> 31542

```python
from qiskit import transpile
from qiskit.tools.visualization import plot_histogram
from qiskit_aer import AerSimulator

qc = grover.copy()
qc.measure_all()
```

```python
sim_cpu = AerSimulator(method='statevector', device='CPU')
```

```python
%%time
result_cpu = sim_cpu.run(qc).result()
```

> CPU times: user 59.7 s, sys: 318 ms, total: 1min
> Wall time: 30.8 s

```python
counts = result_cpu.get_counts()
print(counts)
```

> {'101100111000111011': 506, '110001110011000111': 518}

```python
sim_gpu = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=False)
```

```python
%%time
result_gpu = sim_gpu.run(qc).result()
```

> CPU times: user 1.15 s, sys: 390 ms, total: 1.54 s
> Wall time: 1.28 s

```python
counts = result_gpu.get_counts()
print(counts)
```

```python
sim_cuq = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
```

> {'110001110011000111': 513, '101100111000111011': 511}

```python
%%time
result_cuq = sim_cuq.run(qc).result()
```

> CPU times: user 1.58 s, sys: 52 ms, total: 1.64 s
> Wall time: 1.63 s

```python
counts = result_cuq.get_counts()
print(counts)
```

> {'110001110011000111': 510, '101100111000111011': 514}

ということで Wall time について結果をまとめると以下のようになっており、CPU >> cuQuantum > GPU という並びであった。[Qiskit で遊んでみる (7) — Qiskit Aer GPU](/derwind/articles/dwd-qiskit08) の回路では cuQuantum が最速だったが、回路との相性があるのだろうか？

|CPU|GPU|cuQuantum|
|:--:|:--:|:--:|
|30.8 s|**1.28 s**|1.63 s|

以降、CPU はもう試すだけ時間の無駄なので試さない。

# 23 量子ビットの回路

```python
solutions = ['10011011000000000011111', '11101110000000000011111']
n_qubits = len(solutions[0])
print(f'{n_qubits=}')
```

> n_qubits=23

```python
print(len(grover))
```

> 242831

```python
%%time
result_gpu = sim_gpu.run(qc).result()
```

> CPU times: user 49.2 s, sys: 57.7 s, total: 1min 46s
> Wall time: 1min 44s

```python
counts = result_gpu.get_counts()
print(counts)
```

> {'10011011000000000011111': 501, '11101110000000000011111': 523}

```python
%%time
result_cuq = sim_cuq.run(qc).result()
```

> CPU times: user 2min 52s, sys: 5.07 s, total: 2min 57s
> Wall time: 2min 57s

```python
counts = result_cuq.get_counts()
print(counts)
```

> {'11101110000000000011111': 523, '10011011000000000011111': 501}

で今回も、cuQuantum > GPU という結果。ユーザー CPU 時間について cuQuantum 版の方が長いように見えるのでここの問題だろうか？

|GPU|cuQuantum|
|:--:|:--:|
|**1min 44s**|2min 57s|

# 25 量子ビットの回路

```python
solutions = ['1001101100000000000011111', '1110111000000000000011111']
n_qubits = len(solutions[0])
print(f'{n_qubits=}')
```

> n_qubits=25

```python
print(len(grover))
```

> 537097 [^1]

[^1]: **「わたしの戦闘力は530000です」**

```python
%%time
result_gpu = sim_gpu.run(qc).result()
```

> CPU times: user 6min 40s, sys: 7min 51s, total: 14min 32s
> Wall time: 14min 31s

```python
counts = result_gpu.get_counts()
print(counts)
```

> {'1110111000000000000011111': 489, '1001101100000000000011111': 535}

```python
%%time
result_cuq = sim_cuq.run(qc).result()
```

> CPU times: user 24min 27s, sys: 26.1 s, total: 24min 53s
> Wall time: 24min 49s

```python
counts = result_cuq.get_counts()
print(counts)
```

> {'1001101100000000000011111': 507, '1110111000000000000011111': 517}

で今回も、cuQuantum > GPU という結果。

|GPU|cuQuantum|
|:--:|:--:|
|**14min 31s**|24min 49s|

# まとめ

表を全部まとめると次のようになる:

|n_qubits|CPU|GPU|cuQuantum|
|:--:|:--:|:--:|:--:|
|18|30.8 s|**1.28 s**|1.63 s|
|23|N/A|**1min 44s**|2min 57s|
|25|N/A|**14min 31s**|24min 49s|

詳細は理由はよく分からないが、CPU よりも GPU と cuQuantum は圧倒的に速いのは確定として、[Qiskit で遊んでみる (7) — Qiskit Aer GPU](/derwind/articles/dwd-qiskit08) では GPU > cuQuantum であった結果が今回は逆転して cuQuantum > GPU になっていた。

この辺は回路との相性があるのかもしれないし、単純に Qiskit Aer と cuQuantum の口が噛み合いにくいのかもしれないし、合わせ込みが不十分なのかもしれないが、詳細は分からない[^2]。ひょっとしたらビルド時に与えるオプションの不備があったのかもしれない。時間が経ってユーザーが増えてきて、事例が増えてきたりするとまた変わってくるのかもしれない。

[^2]: と言うか実装を見たら良いのだが、たぶん分からないし調査していないということ。

当面は小さめの回路で実験して、一番パフォーマンスがでそうな組み合わせを見極めてから大規模な回路にすると良いのかなと思う。

少し気になることとしては、[Google Cirq + cuQuantum で遊んでみる (2) — グローバー探索アルゴリズム](/derwind/articles/dwd-cirq-qsim02) では 25 量子ビットのケースで同等の回路で深さが 35377 だったので、今回何故 530000 になっているのかがよく分からない。異なる SDK を使ったので回路の実装をミスっているかもしれない。マルチ制御ゲート周りは結構違いそうなのでそこに何かあるかもしれない。とにかく明らかに状況が違うので、Cirq + qsim + cuQuantum の時の結果と単純な比較はできそうにない。

この辺も複数の SDK で対象となる回路を組んでみて特性を比較すると良いかもしれない。

人から話を聞くだけだと「GPU だと速い」とか「cuQuantum だと速い」とか聞くけれども、どういう実行環境で何量子ビットから GPU の優位性が出てくるかは自分で調べないとならない。また、速いと言ってもどういった実行環境で具体的にどちらがどれくらい速いかも自分で調べないとならない。
色々と考えさせられる結果になったが、自分で解いてみたい課題を設定して自分で実行してみると、掴めるものや見えてくるものがあると感じた。
