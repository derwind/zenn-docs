---
title: "Google Cirq + cuQuantum で遊んでみる (2) — グローバー探索アルゴリズム"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Cirq", "NVIDIA", "cuQuantum", "poem", "Python"]
published: true
---

# 目的

[Google Cirq で遊んでみる (1) — グローバー探索アルゴリズム](/derwind/articles/dwd-cuquantum01) に引き続き、Google Cirq を使ってグローバー探索アルゴリズムを実行してみたい。のだが、今回はさらに `cuQuantum` を用いてみたい。

# cuQuantum 対応 qsim のビルド

`cuQuantum` 対応の `qsim` を使いたい場合、自分で `qsim` をビルドしないとならない様子。
[GPU-based quantum simulation on Google Cloud](https://quantumai.google/qsim/tutorials/gcp_gpu) に従う。

## 想定環境

- Ubuntu 18.04
- Python 3.8
- CUDA 11.2
- NVIDIA T4

## 開発環境セットアップ

```sh
sudo apt install -y nvidia-cuda-toolkit
sudo apt install -y python3.8-dev
sudo apt install -y cmake && pip3 install pybind11
```

## cuQuantum のインストール

```sh
wget https://developer.download.nvidia.com/compute/cuquantum/22.07.1/local_installers/cuquantum-local-repo-ubuntu1804-22.07.1_1.0-1_amd64.deb
sudo dpkg -i cuquantum-local-repo-ubuntu1804-22.07.1_1.0-1_amd64.deb
sudo cp /var/cuquantum-local-repo-ubuntu1804-22.07.1/cuquantum-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuquantum cuquantum-dev cuquantum-doc
```

## qsim インストール

[Installation and Compilation](https://docs.nvidia.com/cuda/cuquantum/custatevec/getting_started.html#installation-and-compilation) に従う。
おおよそ 8 分くらいで終わると思う。

```
git clone https://github.com/quantumlib/qsim.git
export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH} };export CUQUANTUM_DIR="/opt/nvidia/cuquantum" && cd qsim &&  make clean all && pip install .
```

## セットアップの確認

```sh
python -c "import qsimcirq; print(qsimcirq.qsim_gpu)"
```

> <module 'qsimcirq.qsim_cuda' from '/usr/local/lib/python3.8/dist-packages/qsimcirq/qsim_cuda.cpython-38-x86_64-linux-gnu.so'>

こういう出力が得られれば正常にセットアップされたと考えて良さそう。

# 実験用回路の実装

[Google Cirq で遊んでみる (1) — グローバー探索アルゴリズム](/derwind/articles/dwd-cuquantum01) と同様なので細かい説明は省く。

```python
import numpy as np
import cirq
from cirq.contrib.svg import SVGCircuit
%matplotlib inline

def revserse_phase(circuit, qubits, state: str):
    n_qubits = len(qubits)
    qr = []
    for i, digit in enumerate(state):
        if digit == '0':
            qr.append(i)

    circuit.append([
        cirq.Moment(*[cirq.X(qubits[i]) for i in qr]),
        # MCZ start (HXH = Z)
        cirq.Z.controlled(n_qubits-1).on(
            *[qubits[i] for i in range(1, n_qubits)], qubits[0]),
        # MCZ end
        cirq.Moment(*[cirq.X(qubits[i]) for i in qr])
    ])

def define_oracle(n_qubits, solutions):
    # Create the oracle
    qubits = cirq.LineQubit.range(n_qubits)
    oracle = cirq.Circuit()

    for sol in solutions:
        revserse_phase(oracle, qubits, sol)

    return oracle

def define_diffuser(n_qubits):
    qubits = cirq.LineQubit.range(n_qubits)
    diffuser = cirq.Circuit()
    diffuser.append([
        cirq.Moment(*[cirq.H(qubits[i]) for i in range(n_qubits)]),
        cirq.Moment(*[cirq.X(qubits[i]) for i in range(n_qubits)]),
        # MCZ start (HXH = Z)
        cirq.Z.controlled(n_qubits-1).on(
            *[qubits[i] for i in range(1, n_qubits)], qubits[0]),
        # MCZ end
        cirq.Moment(*[cirq.X(qubits[i]) for i in range(n_qubits)]),
        cirq.Moment(*[cirq.H(qubits[i]) for i in range(n_qubits)])
    ])

    return diffuser
```

# 8 量子ビットに挑戦！

特に深い意味はないのだが、測定結果の可視化が潰れない程度の量子ビット数が 8 だった。

```python
n_qubits = 8
solutions = ['10011011', '11101110']

oracle = define_oracle(n_qubits, solutions)
SVGCircuit(oracle)
```

![](/images/dwd-cirq-qsim02/001.png)

```python
diffuser = define_diffuser(n_qubits)
SVGCircuit(diffuser)
```

![](/images/dwd-cirq-qsim02/002.png)

```python
N = 2**n_qubits
angle = np.arcsin(np.sqrt(len(solutions) / N))
iterations = int((np.pi/2 - angle) / (2*angle) + 0.5)
print(f'{angle=}, {np.pi/2=}, {iterations=}')
```

> angle=0.08850384314401546, np.pi/2=1.5707963267948966, iterations=8

```python
qubits = cirq.LineQubit.range(n_qubits)
grover = cirq.Circuit()
# initialize |s>
grover.append([
    cirq.Moment(*[cirq.H(qubits[i]) for i in range(n_qubits)])
])
for _ in range(iterations):
    grover += oracle
    grover += diffuser
SVGCircuit(grover)
```

![](/images/dwd-cirq-qsim02/003.png)

これは以下に見るように回路の深さが深いので途中で切れている。スクロールさせてまでキャプチャする意味も感じなかったので画面に表示されている部分だけ記念にキャプチャした。

```python
print(len(grover))
```

> 89

さて、次は `qsim` で `cuQuantum` を有効にする。[Optional: Use the NVIDIA cuQuantum SDK](https://quantumai.google/qsim/tutorials/gcp_gpu#optional_use_the_nvidia_cuquantum_sdk) によると

```python
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
```

を使うことで `cuQuantum` が有効になるらしい。因みに、`cuQuantum` 有効版として `qsim` をビルドしていない場合には実行時例外が発生する。

```python
import qsimcirq
import matplotlib.pyplot as plt

def binary_labels(num_qubits):
    return [bin(x)[2:].zfill(num_qubits) for x in range(2 ** num_qubits)]

qubits = cirq.LineQubit.range(n_qubits)
grover.append(cirq.measure(qubits[:]))

gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)
result = simulator.run(grover, repetitions=1000)
```

1 秒もかからずに計算は完了した。

```python
# 確率として見たい場合には以下のようにすれば良い。
#result = cirq.get_state_histogram(result)
#result = result / np.sum(result)
_ = cirq.plot_state_histogram(
    result, plt.subplot(), title = 'Measurement results',
    xlabel = 'State', ylabel = 'Count',
    tick_label=binary_labels(n_qubits))
plt.xticks(rotation=70)
plt.show()
```

![](/images/dwd-cirq-qsim02/004.png)

```python
counts = cirq.get_state_histogram(result)
print(solutions)
print(counts[[int(sol, 2) for sol in solutions]])
```

> ['10011011', '11101110']
> [524. 474.]

ということで期待する結果に。
念のために確率振幅も求めてみる。

```python
gpu_options = qsimcirq.QSimOptions(use_gpu=True, gpu_mode=1)
simulator = qsimcirq.QSimSimulator(qsim_options=gpu_options)
results = simulator.compute_amplitudes(grover, bitstrings=[int(sol, 2) for sol in solutions])

print(f'qsim results: {results}')
```

> qsim results: [(0.7055538296699524+0j), (0.705554187297821+0j)]

も一瞬で計算は終わって期待通り。

# 25 量子ビットに挑戦！

とりあえずやってみる。

```python
n_qubits = 25
solutions = ['100110110000000000011111', '111011100000000000011111']
```

でやってみた。

```python
N = 2**n_qubits
angle = np.arcsin(np.sqrt(len(solutions) / N))
iterations = int((np.pi/2 - angle) / (2*angle) + 0.5)
print(f'{angle=}, {np.pi/2=}, {iterations=}')
```

> angle=0.0002441406274253193, np.pi/2=1.5707963267948966, iterations=3216

嫌な予感・・・。

```python
print(len(grover))
```

> 35377

流石に・・・深い。

```python
counts = cirq.get_state_histogram(result)
print(solutions)
print(counts[[int(sol, 2) for sol in solutions]])
```

> ['1001101100000000000011111', '1110111000000000000011111']
> [483. 517.]

ということで期待する結果は得られていた。2 回計算してみたけど、1 回目は 5 分 16 秒で、2 回目は 5 分 10 秒かかった。

ただ、残念なことに、このやり方で `cuQuantum` が利用されているのかはよく分からない。

# まとめ

一応 `cuQuantum` 対応の `qsim` がビルドできたように思う。そして、API からも有効化したシミュレータを作成できているようには思う・・・。が、本当に `cuQuantum` を有効に活用したパスを通っているのか？とか、どういう使い方をした時に `cuQuantum` の `cuStateVec` と `cuTensorNet` がどう使われているのかが分からないことに気付いた。もっと調査を続けないとうまく活用できそうにない。
