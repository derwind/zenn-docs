---
title: "Qiskit で遊んでみる (9) — Shor の符号"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "QiskitAer", "poem", "Python"]
published: true
---

# 目的

量子回路のエラー訂正に「Shor の符号」というものがあるらしいので、効果を体感してみたい。体感なのでつまり、数式で追いかけるのはやめて、適当に雑音の入る回路をシミュレートしてシミュレータで回して満足するところまでを実行する。

# 眺める回路

[量子コンピューティング 基本アルゴリズムから量子機械学習まで](https://www.ohmsha.co.jp/book/9784274226212/) か [Quantum error correction](https://en.wikipedia.org/wiki/Quantum_error_correction) で見る事ができる回路で確認する。

最終的には以下を実装する:

![](/images/dwd-qiskit09/006.png)

ちゃんと実装できれば、1 回のビット反転、1 回の位相反転、或はその両方が同時に起こるケースを検出してエラー訂正できる、らしい。

# 回路を作る

とりあえず、必要なモジュールを import する:

```python
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Kraus
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error
```

## ビット反転の回路とエラー訂正回路

そのまま実装する:

```python
bit_flip_code_circuit = QuantumCircuit(3)
bit_flip_code_circuit.cx(0, 1)
bit_flip_code_circuit.cx(0, 2)

bit_flip_decode_circuit = QuantumCircuit(3)
bit_flip_decode_circuit.cx(0, 1)
bit_flip_decode_circuit.cx(0, 2)
bit_flip_decode_circuit.ccx(1, 2, 0)

display(bit_flip_code_circuit.draw())
display(bit_flip_decode_circuit.draw())
```

![](/images/dwd-qiskit09/001.png)

## 位相反転の回路とエラー訂正回路

そのまま実装する:

```python
bit_flip_code_circuit = QuantumCircuit(3)
bit_flip_code_circuit.cx(0, 1)
bit_flip_code_circuit.cx(0, 2)

bit_flip_decode_circuit = QuantumCircuit(3)
bit_flip_decode_circuit.cx(0, 1)
bit_flip_decode_circuit.cx(0, 2)
bit_flip_decode_circuit.ccx(1, 2, 0)

display(bit_flip_code_circuit.draw())
display(bit_flip_decode_circuit.draw())
```

![](/images/dwd-qiskit09/002.png)

## Shor の符号の回路とエラー訂正回路

上記で用意したパーツを用いて、そのまま実装する:

```python
sign_flip_code_circuit = QuantumCircuit(3)
sign_flip_code_circuit.cx(0, 1)
sign_flip_code_circuit.cx(0, 2)
for i in range(3):
    sign_flip_code_circuit.h(i)

sign_flip_decode_circuit = QuantumCircuit(3)
for i in range(3):
    sign_flip_decode_circuit.h(i)
sign_flip_decode_circuit.cx(0, 1)
sign_flip_decode_circuit.cx(0, 2)
sign_flip_decode_circuit.ccx(1, 2, 0)

display(sign_flip_code_circuit.draw())
display(sign_flip_decode_circuit.draw())
```

![](/images/dwd-qiskit09/003.png)
![](/images/dwd-qiskit09/004.png)

## 1 回のビット反転、1 回の位相反転が発生し得る量子雑音の回路

「量子チャネル」という呼び方で良いのか分かっていない部分もあるのだが、とりあえず今回はそう呼ぶことにして「0 番目にビット反転、3 番目に位相反転」の量子雑音が 10% 程度入る回路を実装する:

```python
quantum_channel = QuantumCircuit(9)
quantum_channel.x(0)
quantum_channel.z(3)
p_error = 0.9
error = pauli_error([('X', p_error), ('I', 1 - p_error)])
quantum_channel.append(Kraus(error), [0])
error = pauli_error([('Z', p_error), ('I', 1 - p_error)])
quantum_channel.append(Kraus(error), [3])
display(quantum_channel.draw())
```

![](/images/dwd-qiskit09/005.png)

一見 X ゲートと Z ゲートを適用する回路に見えるが、これらが 90% しくじる設定にしているので、ほぼほぼ恒等ゲートという気持ちである。

## 全部くっつける

Shor の符号の回路 + 量子雑音の回路 + エラー訂正の回路の順番に組み合わせていく:

```python
circuit = shor_code_circuit.copy()
circuit.barrier()
circuit.compose(quantum_channel, range(9), inplace=True)
circuit.barrier()
circuit.compose(shor_decode_circuit, range(9), inplace=True)
circuit.add_register(ClassicalRegister(1, 'cr'))
circuit.measure([0], [0])

display(circuit.draw())
```

![](/images/dwd-qiskit09/006.png)

# 実験

雑音のせいで一般に混合状態になるかもしれないので本当は密度演算子での時間発展を考えるべきであろうが、エラー訂正されて終状態は純粋状態のはず、と信じて検証してみる。

凄い雑な検証だが「50% の確率で $\ket{0}$ か $\ket{1}$ で初期化された回路が適当に雑音にさらされて測定される。この時ちゃんとエラー訂正が機能すれば始状態だけが観測されているはずなのでそれを確認している。1000 ショットで試していて、始状態と同じものが 1000 回観測されるはず」というのを 100 回試している。

以下を実行しても見かけ上何も起こらないのが期待値であり、実際何度か試してもセルの実行で待たされた以外は何も起こらなかった。

```python
import random

sim = AerSimulator()

for _ in range(100):
    circuit2 = circuit.copy()
    shots = 1000
    expected = 0

    if random.random() >= 0.5:
        init_circuit = QuantumCircuit(9)
        init_circuit.x(0)
        circuit2 = init_circuit.compose(circuit2, range(9))
        expected = 1

    result = sim.run(circuit2, shots=shots).result()
    counts = result.get_counts()
    assert str(expected) in counts and counts[str(expected)] == shots
```

# まとめ

もの凄い駆け足で Shor の符号を見た。僅かな雑音に耐えるために 9 個の物理量子ビットを用いて 1 つの論理量子ビットを構成した形である。Hamming 距離が 3 のケースなので、これ以上の雑音は正しくエラー訂正できない。

エラー訂正を専門にしているわけではないのであまりに乱雑な感想だが、NISQ デバイスのエラー訂正厳しそうだな、物理量子ビットが増えてもエラーフリーな論理量子ビットの個数って幾らにできるのだろう・・・と思った。

これまた詳しくないので乱暴なことしか書けないが、しばらくはエラー緩和 (error mitigation) で頑張るほうが物理量子ビット数を活かせて良いのかな・・・？とも思う。分からない。

# 参考文献

- [量子コンピューティング 基本アルゴリズムから量子機械学習まで](https://www.ohmsha.co.jp/book/9784274226212/)
- [Quantum error correction](https://en.wikipedia.org/wiki/Quantum_error_correction)
