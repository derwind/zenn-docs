---
title: "cuQuantum で遊んでみる (5) — VQE その 2"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "poem", "Python"]
published: true
---

# 目的

[cuQuantum で遊んでみる (4) — VQE](/derwind/articles/dwd-cuquantum04) で無理やり VQE をしてみたが、もうちょっと小綺麗なことがしたいので、`cuQuantum` の API で期待値計算をさせたい。

一応前回の無理やり版も実行して比較したい。

# お題

$\mathcal{H} = Z \otimes X$ の期待値計算を最小化したい。要するに、$\ket{\psi} = \ket{1} \otimes \ket{+} = \frac{1}{\sqrt{2}}(\ket{10} + \ket{11})$ 或は $\ket{\psi^\prime} = \ket{0} \otimes \ket{-} = \frac{1}{\sqrt{2}}(\ket{00} - \ket{01})$ を PQC で近似したい。

![](/images/dwd-cuquantum05/001.png)

# 無理やり版

必要なモジュールを import:

```python
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from cuquantum import CircuitToEinsum, contract
import numpy as np
import cupy as cp
from scipy.optimize import minimize
```

そして、**前回と同様に無理やり実装**をする:

```
def create_ansatz(num_qubits):
    ansatz = TwoLocal(num_qubits, 'ry', 'cz')
    return ansatz, ansatz.inverse()

hamiltonian = SparsePauliOp.from_list([
    ('ZX', 1.0)
])
hamiltonian = cp.array(hamiltonian.to_matrix().reshape(2, 2, 2, 2))

num_qubits = 2
ansatz, ansatz_dagger = create_ansatz(num_qubits)

def expectation(theta, ansatz, ansatz_dagger, hamiltonian):
    assert len(theta) == 4 * ansatz.num_qubits
    ansatz = ansatz.bind_parameters(theta)
    ansatz_dagger = ansatz_dagger.bind_parameters(theta)

    converter = CircuitToEinsum(ansatz)
    expr, operands = converter.state_vector()
    vec = contract(expr, *operands)

    converter = CircuitToEinsum(ansatz_dagger)
    expr, operands = converter.amplitude('0' * ansatz.num_qubits)
    # 'a,b,cb,da,efcd...->' ==> 'cb,da,efcd...->ab'
    out, expr = expr[:ansatz.num_qubits*2].replace(',', ''), \
                expr[ansatz.num_qubits*2:]
    expr += out
    vec_dagger = contract(expr, *(operands[ansatz.num_qubits:]))

    val = contract('ab,cdba,dc->', vec, hamiltonian, vec_dagger)
    return float(val.real)
```

**実験**:

```python
theta = np.random.random(4 * num_qubits)
args = (ansatz, ansatz_dagger, hamiltonian)

result = minimize(expectation, theta, args=args, method='Powell')
print(f'optimal_value: {result.fun}')
print(f'x: {result.x}')
```

> optimal_value: -1.0000000000000007
> x: [-2.16629773  0.87642794  0.48274178  0.58879033  0.17536848  0.69861945
  0.40712598  0.19401039]

**結果**:

```python
qc = TwoLocal(2, 'ry', 'cz').decompose()
qc = qc.bind_parameters(result.x)
qc.measure_all()
backend = Aer.get_backend('aer_simulator')
counts = backend.run(qc).result().get_counts()
plot_histogram(counts)
```

![](/images/dwd-cuquantum05/002.png)

なんだかとても微妙な結果のように見える。が、よく考えると面白い結果だったのであえて掲載した。これは概ね

$$
\begin{align*}
\ket{\psi} = \sqrt{\frac{3}{5}} \frac{\ket{00} - \ket{01}}{\sqrt{2}} + \sqrt{\frac{2}{5}} \frac{\ket{10} + \ket{11}}{\sqrt{2}} = \sqrt{\frac{3}{5}} \ket{0-} + \sqrt{\frac{2}{5}} \ket{1+}
\end{align*}
$$

が出ていることを意味している。これは後で検証しよう。念のため計算すると

$$
\newcommand \parenthetical[1] { \left( #1 \right) }
\begin{align*}
\braket{\psi | Z \!\otimes\! X | \psi} &= \parenthetical{ \sqrt{\frac{3}{5}} \bra{0-} + \sqrt{\frac{2}{5}} \bra{1+} } \! (Z \!\otimes\! X) \! \parenthetical{ \sqrt{\frac{3}{5}} \ket{0-} + \sqrt{\frac{2}{5}} \ket{1+} } \\
& = \parenthetical{ \sqrt{\frac{3}{5}} \bra{0-} + \sqrt{\frac{2}{5}} \bra{1+} } \! \parenthetical{ -\sqrt{\frac{3}{5}} \ket{0-} - \sqrt{\frac{2}{5}} \ket{1+} } = - \frac{3}{5} - \frac{2}{5} = -1
\end{align*}
$$

となるのである。と言っても測定では位相の情報が落ちているので、状態ベクトルで求めてみる。

```python
qc.remove_final_measurements()
sv = Statevector(qc)
sv.draw('latex')
```

$$
\begin{align*}
0.5492966902 \ket{00} - 0.5492966858 \ket{01} + 0.4452787292 \ket{10} + 0.4452787335 \ket{11}
\end{align*}
$$

となる。要するに、ハミルトニアンの固有値 -1 が縮退しているので、固有ベクトル同士の重ね合わせが求まったということである。

実は VQE を実行するごとにこの辺の配合率はふらふらしており、安定しなかった。次の「cuQuantum API 版」では「当初これが出てくることを期待していた」という結果が出た。こちらも VQE 実行ごとに結果がふらふらしているので、単に今回はそういう結果だったというだけである。



# cuQuantum API 版

**期待値計算実装**:

ハミルトニアンが Pauli 演算子であれば `expectation` メソッドが使えるので期待値計算が大幅に楽になる。

```python
num_qubits = 2
ansatz, _ = create_ansatz(num_qubits)

def expectation_cutn(theta, ansatz, hamiltonian: str):
    assert len(theta) == 4 * ansatz.num_qubits
    ansatz = ansatz.bind_parameters(theta)

    converter = CircuitToEinsum(ansatz)
    expr, operands = converter.expectation(hamiltonian)
    val = contract(expr, *operands)

    return float(val.real)
```

**実験**:

```python
theta = np.random.random(4 * num_qubits)
args = (ansatz, hamiltonian)

hamiltonian = 'XZ'

result = minimize(expectation_cutn, theta, args=args, method='Powell')
print(f'optimal_value: {result.fun}')
print(f'x: {result.x}')
```

> optimal_value: -1.0000000000000004
> x: [2.37477257 0.14751172 1.11527912 0.41963828 0.3659612  0.52473461
 0.84868946 0.13640107]

**結果**:

```python
qc = TwoLocal(2, 'ry', 'cz').decompose()
qc = qc.bind_parameters(result.x)
qc.measure_all()
backend = Aer.get_backend('aer_simulator')
counts = backend.run(qc).result().get_counts()
plot_histogram(counts)
```

![](/images/dwd-cuquantum05/003.png)

これは概ね、

$$
\begin{align*}
\ket{\psi} = \frac{1}{\sqrt{2}}(\ket{00} - \ket{01}) = \ket{0-}
\end{align*}
$$

が出ていることを意味している。状態ベクトルを念のため計算すると

```python
qc.remove_final_measurements()
sv = Statevector(qc)
sv.draw('latex')
```

$$
\begin{align*}
-0.7053123039 \ket{00} + 0.7053123164 \ket{01} - 0.0503442633 \ket{10} - 0.0503442692 \ket{11}
\end{align*}
$$

となる。

# まとめ

今回試したどちらの方法でもハミルトニアンの最小の固有値 -1 は求まった。が、Pauli 演算子なら期待値計算が cuQuantum API 一発で済んで楽だよねという結論。

なお、最適化手法についてはどれが良いかは分からないが、‘SLSQP’ はわりと縮退した状態がまざっている事が多めのような気がした。‘Powell’ と ‘COBYLA’ は単一の固有ベクトルがわりとハッキリ出やすかったように感じた。あまり本質的ではないし、ただの偶然かもしれないが。

ハミルトニアンを作るとき、Qiskit API の `SparsePauliOp` で定義する場合と、cuQuantum API の `expectation` で与える場合で順序が逆なので、この辺は注意が必要かなという感想。
