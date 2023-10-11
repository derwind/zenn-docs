---
title: "Qiskit で遊んでみる (22) — Google Colab 上で Qiskit Aer GPU"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "cuQuantum", "poem"]
published: true
---

# 目的

cuQuantum 対応の Qiskit Aer を毎回ビルドするのは大変なので、Google Colab 上でお手軽に使いましょうという内容。実は自分では特に何もやってないのだが、情報が埋もれているので[^1]、それを掘り返して少し試してみるという感じになる。

[^1]: というよりはテスト版みたいなものなので、公式には告知されていない。

# 現在の PyPI 版にはちょっとした問題がある

[[qiskit-aer-gpu] ImportError: libcustatevec.so.1](https://github.com/Qiskit/qiskit-aer/issues/1874) という Issue を大分前に起票したのだが、デフォルトでは [qiskit-aer-gpu 0.12.2](https://pypi.org/project/qiskit-aer-gpu/0.12.2/) から必要なライブラリが見えておらずパスを通さないとならない状態である。これを Google Colab 上で実行するのは難しいので、テスト版の `qiskit-aer-gpu 0.13.1` を使う。

# 実際に試す

上記 Issue の本文を丹念に読むと、幾つか Jupyter ノートブックへのリンクがあり、要するに以下を実行すれば Google Colab 上で GPU 対応の [qiskit-aer-gpu 0.13.1](https://test.pypi.org/project/qiskit-aer-gpu/0.13.1/) が実行できるようになる。テスト版なので多少問題はあるかもしれないが、とりあえず使ってみる。

```python
%%bash
pip install -q "qiskit==0.44.2"
pip install -q pylatexenc
pip uninstall -y qiskit-aer
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 cuquantum-cu12
pip install -i https://test.pypi.org/simple/ "qiskit-aer-gpu==0.13.1"
ldd /usr/local/lib/python3.10/dist-packages/qiskit_aer/backends/controller_wrappers.cpython-310-x86_64-linux-gnu.so
```

上記は 2 分 3 秒かかった。

# 評価

幾つか試して問題なさそうであることは確認したが、ここでは上記 Jupyter ノートブックから抜粋したい。

## cuStateVec

```python
from qiskit import *
from qiskit.circuit.library import *
from qiskit.providers.aer import *
import sys

sim = AerSimulator(method='statevector')
sim_gpu = AerSimulator(method='statevector', device='GPU')

shots = 100
depth=10
qubits = 25
block_bits = 25

circuit = transpile(QuantumVolume(qubits, depth, seed=0),
                    backend=sim,
                    optimization_level=0)
circuit.measure_all()

result_base = execute(circuit,sim,shots=shots,seed_simulator=12345).result()

#print(result_base)
#if result_base.to_dict()['metadata']['mpi_rank'] == 0:
print(result_base.to_dict()['backend_name'])
print(result_base.to_dict()['results'][0]['time_taken'])
print(sorted(result_base.to_dict()['results'][0]['data']['counts'].items(),key=lambda x:x[0]))

result_gpu = execute(circuit,sim_gpu,shots=shots,seed_simulator=12345,blocking_qubits=block_bits, cuStateVec_enable=True).result()

#print(result_gpu)
#if result_gpu.to_dict()['metadata']['mpi_rank'] == 0:
print(result_gpu.to_dict()['backend_name'])
print(result_gpu.to_dict()['results'][0]['time_taken'])
print(sorted(result_gpu.to_dict()['results'][0]['data']['counts'].items(),key=lambda x:x[0]))
```

> aer_simulator_statevector
> 36.944698124
> [('0x1004c0e', 1), ('0x10de628', 1), ('0x11c3543', 1), ...]
> aer_simulator_statevector_gpu
> 1.107945392
> [('0x1004c0e', 1), ('0x10de628', 1), ('0x11c3543', 1), ...]

これは 38 秒くらいかかった。

## cuTensorNet

```python
from qiskit import *
from qiskit.circuit.library import *
from qiskit.providers.aer import *
import sys

sim = AerSimulator(method='statevector')
sim_gpu = AerSimulator(method='tensor_network', device='GPU')

shots = 100
depth=10
qubits = 10
block_bits = 10

circuit = transpile(QuantumVolume(qubits, depth, seed=0),
                    backend=sim,
                    optimization_level=0)
circuit.measure_all()

result_base = execute(circuit,sim,shots=shots,seed_simulator=12345).result()

#print(result_base)
#if result_base.to_dict()['metadata']['mpi_rank'] == 0:
print(result_base.to_dict()['backend_name'])
print(result_base.to_dict()['results'][0]['time_taken'])
print(sorted(result_base.to_dict()['results'][0]['data']['counts'].items(),key=lambda x:x[0]))

result_gpu = execute(circuit,sim_gpu,shots=shots,seed_simulator=12345,blocking_qubits=block_bits).result()

#print(result_gpu)
#if result_gpu.to_dict()['metadata']['mpi_rank'] == 0:
print(result_gpu.to_dict()['backend_name'])
print(result_gpu.to_dict()['results'][0]['time_taken'])
print(sorted(result_gpu.to_dict()['results'][0]['data']['counts'].items(),key=lambda x:x[0]))
```

> aer_simulator_statevector
> 0.002062768
> [('0x10', 1), ('0x104', 1), ('0x108', 1), ...]
> aer_simulator_tensor_network_gpu
> 3.209672026
> [('0x10', 1), ('0x104', 1), ('0x108', 1), ...]

これは 3 秒くらいかかった。

# まとめ

Qiskit Aer 0.13 の正式版がいつになるのか分からないので、テスト版を用いて GPU シミュレーションを Google Colab 上でお手軽に実行してみた。動かした感じでは普通のことをやる分には問題なさそうである。
