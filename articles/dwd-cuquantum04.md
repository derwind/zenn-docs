---
title: "cuQuantum で遊んでみる (4) — VQE"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "poem", "Python"]
published: true
---

# 目的

[cuQuantum で遊んでみる (2) — グローバー探索アルゴリズム](/derwind/articles/dwd-cuquantum02) でさらっと `cuQuantum` の `cuTensorNet` に触って、[cuQuantum で遊んでみる (3) — 期待値計算](/derwind/articles/dwd-cuquantum03) で比較的低レベルの `cuStateVec` の API で期待値計算を行った。

今回は、`cuTensorNet` で VQE を実行してみたいと思う。公式のサンプル [qiskit_advanced.ipynb](https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/cutensornet/circuit_converter/qiskit_advanced.ipynb) では QAOA を解説しているが、ちょっと今回は頑張って、参考にしながら VQE をしてみる形だ。

# お題

あまりちゃんとした VQE の知識がないので、[Qiskit におけるアルゴリズム入門](https://qiskit.org/documentation/locale/ja_JP/tutorials/algorithms/01_algorithms_introduction.html) の VQE を拝借したい。

$$
\begin{align*}
\mathcal{H} = -1.052 I \otimes I + 0.398 I \otimes Z - 0.398 Z \otimes I - 0.011 Z \otimes Z + 0.181 X \otimes X
\end{align*}
$$

といった感じのハミルトニアンの期待値を最適化して、基底状態の固有値を求めることになる。

> 原子間距離 0.735A の H2 分子のハミルトニアン

と書いてあるのでそういうことらしい。オングストロームが懐かしすぎるが、長さはまったく覚えていない。

Qiskit の場合だと、こういうハミルトニアンは `SparsePauliOp` で定義すれば良くて、エルミート演算子が得られる。ところで、`QuantumCircuit` に適用する演算子はユニタリ演算子でないとダメなので、そのままでは組み合わせられない。普通は ansatz を準備する量子回路とハミルトニアンをセットで Qiskit の API に渡すことで期待値計算してくれるので、何も考えることはないのだが、今回これを無理やり `cuTensorNet` でやってみたいのだ。

色々考えてまったく良い方法が思いつかなかったので、

- ansatz: `QuantumCircuit` からテンソルを作成
- ハミルトニアン: `SparsePauliOp` からテンソルを作成

して $\braket{\psi(\theta) | \mathcal{H} | \psi(\theta)}$ を計算することにした。

# ゴール

ハミルトニアンの基底状態の固有値

> 'eigenvalue': -1.857275020719397,

を求めることになる。

# やってみよう

とりあえず必要なモジュールを import する:

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from cuquantum import CircuitToEinsum, contract
import numpy as np
import cupy as cp
from scipy.optimize import minimize
```

続いて $\ket{\psi(\theta)}$ と $\bra{\psi(\theta)}$ を作る:

```python
def create_ansatz(num_qubits):
    ansatz = TwoLocal(num_qubits, 'ry', 'cz')
    return ansatz, ansatz.inverse()
```

ハミルトニアンを定義する:

```python
# -1.0 * II + 0.39  * IZ - 0.39 * ZI - 0.01 * ZZ + 0.18 * XX
hamiltonian = SparsePauliOp.from_list([
    ('II', -1.052373245772859),
    ('IZ', 0.39793742484318045),
    ('ZI', -0.39793742484318045),
    ('ZZ', -0.01128010425623538),
    ('XX', 0.18093119978423156)
])
hamiltonian = cp.array(hamiltonian.to_matrix().reshape(2, 2, 2, 2))
```

この辺がかなり強引で、`SparsePauliOp` で定義したハミルトニアンから 4x4 行列を生成して、更に 4 階のテンソルにしてしまうのだ。今回、`cuTensorNet` のバックエンドにデフォルトの `CuPy` を使っているので、`CuPy` の意味での多次元配列を用いているが、バックエンドには `PyTorch` や、そして恐らく `JAX` も使用できるので、その辺は好きにやれば良い。

期待値を計算する関数を定義する:

```python
num_qubits = 2
ansatz, ansatz_dagger = create_ansatz(num_qubits)

def expectation(theta, ansatz, ansatz_dagger, hamiltonian):
    assert len(theta) == 4 * ansatz.num_qubits
    # ansatz |ψ(θ)> にパラメータを設定
    ansatz = ansatz.bind_parameters(theta)
    ansatz_dagger = ansatz_dagger.bind_parameters(theta)

    converter = CircuitToEinsum(ansatz)
    expr, operands = converter.state_vector()
    # テンソルの縮約計算を実行
    vec = contract(expr, *operands)

    # ansatz <ψ(θ)| にパラメータを設定
    converter = CircuitToEinsum(ansatz_dagger)
    expr, operands = converter.amplitude('0' * ansatz.num_qubits)
    # 'a,b,cb,da,efcd...->' ==> 'cb,da,efcd...->ab'
    out, expr = expr[:ansatz.num_qubits*2].replace(',', ''), \
                expr[ansatz.num_qubits*2:]
    expr += out
    # テンソルの縮約計算を実行
    vec_dagger = contract(expr, *(operands[ansatz.num_qubits:]))

    # 期待値を求めるテンソル同士を縮約計算する
    val = contract('ab,cdba,dc->', vec, hamiltonian, vec_dagger)
    return float(val.real)
```

かなり変なことをやっていて、$\ket{\psi(\theta)}$ はハミルトニアンのテンソルと縮約するために $\bra{00}$ での測定をさせずに状態ベクトルで止めている。一方で、$\bra{\psi(\theta)}$ は終端を $\bra{00}$ で閉じたいので `amplitude('00')` としている。なおかつ、入力はハミルトニアンから伸びてくる足なので、$\ket{00}$ ではない。なので、入力部分を削って、逆に出力に添え字 `ab` を回している。

この関数は [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) に渡すことになるのでインターフェイスを合わせている。

# 実験

```python
theta = np.random.random(4 * num_qubits)
args = (ansatz, ansatz_dagger, hamiltonian)

result = minimize(expectation, theta, args=args method='SLSQP')
print(f'optimal_value: {result.fun}')
```

> optimal_value: -1.857275014742714

ということで、相当無茶苦茶なことをしたが、一応欲しい答えは得られたように思う。

# まとめ

本当はどうするのが良かったのか分からないが、とりあえず今回は無理やり計算で H2 分子の基底状態の固有値を求めてみた。きっともっとうまい方法があるに違いない。

ところで最近「NVIDIA CUDA Quantum」という SDK がリリースされたようで？、[Variational Quantum Eigensolver](https://nvidia.github.io/cuda-quantum/latest/using/python.html#variational-quantum-eigensolver) を見ると、同 SDK の中で閉じた形で VQE ができるようだ。`cuQuantum` も同じような感じにしてもらえたり、或は `cuQuantum` と「NVIDIA CUDA Quantum」の間で行き来できても便利かもしれない。
