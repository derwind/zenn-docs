---
title: "cuQuantum で遊んでみる (3) — 期待値計算"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "poem", "Python"]
published: false
---

# 目的

引き続き `cuQuantum` で遊んでみるのだが、今回は `Qiskit` も `Cirq` も使わずに直接 `cuQuantum` の Python ラッパを叩いてみたい。

# ハミルトニアン

期待値を計算する対象のハミルトニアンはお馴染みの Pauli $Z$ 演算子とする。$\ket{0}$ と $\ket{1}$ で期待値を計算してみたい。

![](/images/dwd-cuquantum03/001.png)

# 早速計算

[expectation_pauli.py](https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/custatevec/expectation_pauli.py) を参考にするのだが、なかなか C++ なので一部 Python っぽく書き換えて使う。

```python
import numpy as np
import cupy as cp

import cuquantum
from cuquantum import custatevec as cusv

nIndexBits = 1
nSvSize    = 2**nIndexBits

Z = cusv.Pauli.Z

expectation_values0 = np.empty(1, dtype=np.float64)
expectation_values1 = np.empty(1, dtype=np.float64)

ZERO = cp.asarray([1, 0], dtype=np.complex64)
ONE  = cp.asarray([0, 1], dtype=np.complex64)

####################################################################################

handle = cusv.create()

# apply Pauli Z operator to |0> and |1>
cusv.compute_expectations_on_pauli_basis(
    handle, ZERO.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits,
    expectation_values0.ctypes.data, [[Z]], 1, [[0]], [1])

cusv.compute_expectations_on_pauli_basis(
    handle, ONE.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits,
    expectation_values1.ctypes.data, [[Z]], 1, [[0]], [1])

# destroy handle
cusv.destroy(handle)

print(expectation_values0, expectation_values1)
```

> [1.] [-1.]

ということで、$\braket{0|Z|0} = 1$ と $\braket{1|Z|1} = -1$ が無事求まった。

# まとめ

もうちょっと試したコードもあるが見ていて煩雑に感じたので、一番基本のパターンで手短な記事を残しておいたほうが後々便利かな？と思ったのでそうした。

特別な理由がないなら、まずは `Qiskit` や `Cirq` のシミュレータ経由で実行するほうが圧倒的に楽そうには感じる。

# おまけ

引用先のコードで $-0.14$ の期待値を求めるほうの内容は、$\braket{\psi|Y \otimes X \otimes I|\psi}$ を求めているらしい。やや雑だが以下のコードと比較すると良さそうである。

```python
sv = np.array([
    [0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
     0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j]], dtype=np.complex64)

I = np.array([
    [1, 0],
    [0, 1]
], dtype=np.complex64)
X = np.array([
    [0, 1],
    [1, 0]
], dtype=np.complex64)
Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=np.complex64)
H = np.kron(np.kron(Y, X), I)
print(sv.conj()@H@sv.reshape(-1, 1))
```

> [[-0.14+5.1409006e-09j]]
