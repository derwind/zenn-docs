---
title: "Qiskit で遊んでみる (10) — スタビライザ符号"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "QiskitAer", "poem", "Python"]
published: false
---

# 目的

スタビライザ符号についてちょっと勉強してみたので、何も分かってないけどコードを書いてみる。

# スタビライザ符号概要

以下に淡々とスタビライザ符号の概要を列挙する。見ても何も嬉しくはないただのメモ書きである。後で行う検証の実装の上では特に気にしなくても良い程度のものである。

- エラー訂正の仕組みを持った論理量子ビット
- その量子ビット間にはたらく論理量子ゲート

を作りたい。そのために以下のような事を考えるようなものらしい。

- **スタビライザ群** $\mathcal{S}$ なるものが量子状態のなすヒルベルト空間 $\mathcal{H}$ に作用してると考えて、$\mathcal{H} = V \oplus V^\perp$ と直交分解する。ここで、$V$ は $\mathcal{S}$-不変な (スタビライズされた) 部分空間とする。この $V$ を**スタビライザ状態**と呼ぶ。
- $V$ の中で論理量子ビットを実装すれば、計算中にエラーが生じた時 $V^\perp$ の元に化けるので、$\mathcal{S}$-不変でなくなってエラー検出できる。($\mathcal{S}$ の元を通して測定するとエラーを検出できる)
- このようにスタビライザ状態を一種のコードと見て論理的な量子ビットを実装することを**スタビライザ符号**と呼び、こういう枠組みのことをスタビライザ形式 (stabilizer formalism) と呼ぶ。

具体的には、3 量子ビットのスタビライザ状態を用いる場合、論理量子状態 $\ket{0_L}$ を $\ket{000}$ で実装し、論理量子状態 $\ket{1_L}$ を $\ket{111}$ で実装することができる。

また、エラー訂正の仕組みを保ったまま論理量子ゲートを適用したいので、

- 論理量子ゲートに採用される演算子はスタビライザ状態をスタビライザ状態にうつすようなものであることが望まれる。このような性質を持った演算子を**クリフォード演算子**と呼び、これらは群の構造を持っている。これをクリフォード群と呼ぶ。

通常の量子回路の場合、ある状態から別の任意の量子状態を近似するためのゲートセットの存在と構成が望まれる。そういうゲートセットが存在することを**普遍性**と呼び、例えば、1 量子ビットのゲート $H$ と $T$ および 2 量子ビットのゲート $CX$ の組み合わせはこの普遍性を持つゲートセットになっている。とすれば

- $H$, $T$, $CX$ がクリフォードゲートであれば、エラー訂正の仕組みとゲートセットの普遍性が両立して幸せになれる

となるが、これは否定的で $T$ がクリフォード演算子では**ない**。魔法状態という仕組みを使えば実質的なクリフォード演算子にできるが、魔法状態の作成の計算コストが大きすぎて現実的ではない。

# 論理量子ビットの実装

前述のような理論を述べたところであまり嬉しいわけでもないので、具体例だけ見る。

3 量子ビットのケースがわりと使い勝手が良いのでこのケースだけ見る。このケースではスタビライザ群として $\mathcal{S}_3 = \{ III, ZZI, IZZ, ZIZ \}$ がとれる。これに対応するヒルベルト空間 $\mathbb{C}^{2^3}$ の中の $\mathcal{S}_3$-不変な部分空間、即ちスタビライザ状態は $V = \mathrm{span}\{ \ket{000}, \ket{111} \}$ となる。

ということで、この空間 $V$ をコードの空間に見立てて論理量子ビットを実装すれば良いのだが、概要で述べたように $\ket{0_L}$ を $\ket{000}$ で実装し、論理量子状態 $\ket{1_L}$ を $\ket{111}$ で実装すると何となく分かりやすい気がする。

今回 3 量子ビットだけ使うので、[Qiskit で遊んでみる (9) — Shor の符号#ビット反転の回路とエラー訂正回路](/derwind/articles/dwd-qiskit09#ビット反転の回路とエラー訂正回路) で見た「ビット反転」のエラー訂正の仕組みの考えと組み合わせる形で、以下のように符号回路を実装する。

- $\ket{0_L} = \ket{000}$ の準備回路

![](/images/dwd-qiskit10/001.png)

- $\ket{1_L} = \ket{111}$ の準備回路

![](/images/dwd-qiskit10/002.png)

# 量子回路を実装していく

何はともあれ、必要なモジュールを import する

```python
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister
from typing import List, Tuple, Dict, Sequence, Union, Optional
```

今回、論理量子ビットを使うので、ちょっとでも小綺麗にしたくて僅かに抽象化する[^1]:

[^1]: 雑な実装だし長いので見なくて良い。辻褄が合うようにしているだけである。

```python
class LogicalQubit:
    def __init__(self, enc_circuit):
        self._circuit = enc_circuit.copy()

    @property
    def circuit(self):
        return self._circuit

class LogicalClbit:
    def __init__(self, dec_circuit):
        self._circuit = dec_circuit.copy()

    @property
    def circuit(self):
        return self._circuit

class LogicalQuantumCircuit:
    def __init__(self, qregs: int):
        self.n_code = 3 # width of coding circuits
        self._circuit = None
        self._qubits = []
        self._clbits = []
        self.add_register(qregs)
        self.encoded = False

    def add_register(self, qregs: int):
        self._circuit = QuantumCircuit(qregs*self.n_code)
        enc_circuit, _ = self.make_bit_flip_code_circuits()

        for i in range(qregs):
            qubit = LogicalQubit(enc_circuit)
            self._qubits.append(qubit)

    def compose(self, other: QuantumCircuit, qubits: Optional[Sequence[int]] = None):
        self.circuit.compose(other, qubits, inplace=True)

    def encode(self):
        for i, qubit in enumerate(self.qubits):
            self.circuit.compose(qubit.circuit, range(i*self.n_code, (i+1)*self.n_code), inplace=True)
        self.encoded = True

    def barrier(self):
        self.circuit.barrier()

    def h(self, qubit: int):
        self.circuit.h(qubit*self.n_code)

    def i(self, qubit: int):
        pass

    def id(self, qubit: int):
        self.i(qubit)

    def x(self, qubit: int):
        self.circuit.x(qubit*self.n_code)

    def cx(self, control_qubit: int, target_qubit: int):
        self.circuit.cx(control_qubit*self.n_code, target_qubit*self.n_code)

    def measure_all(self):
        if self.encoded:
            _, dec_circuit = self.make_bit_flip_code_circuits()
            for i in range(len(self.qubits)):
                clbit = LogicalClbit(dec_circuit)
                self.circuit.compose(clbit.circuit, range(i*self.n_code, (i+1)*self.n_code), inplace=True)
                self._clbits.append(clbit)
        self.circuit.barrier()
        self.circuit.add_register(ClassicalRegister(len(self.qubits*self.n_code), 'c'))
        qubits = clbits = [i*self.n_code for i in range(len(self.qubits))]
        self.circuit.measure(qubits, clbits)

    def draw(
        self,
        output: Optional[str] = None,
        scale: Optional[float] = None,
        style: Optional[Union[dict, str]] = None
    ):
        return self.circuit.draw(output=output, scale=scale, style=style)

    @property
    def circuit(self):
        return self._circuit

    @property
    def qubits(self) -> List[LogicalQubit]:
        return self._qubits

    @property
    def clbits(self) -> List[LogicalClbit]:
        return self._clbits

    def make_bit_flip_code_circuits(self):
        enc_circuit = QuantumCircuit(self.n_code)
        enc_circuit.cx(0, 1)
        enc_circuit.cx(0, 2)

        dec_circuit = QuantumCircuit(self.n_code)
        dec_circuit.cx(0, 1)
        dec_circuit.cx(0, 2)
        dec_circuit.ccx(1, 2, 0)

        return enc_circuit, dec_circuit

    def __len__(self):
        return len(self.circuit)

def counts2counts(counts: qiskit.result.counts.Counts, n_code):
    def key2key(key):
        key = key[::-1]
        return ''.join([key[i*n_code] for i in range(len(key)//n_code)])[::-1]

    c = counts
    d = {}
    for k,v in counts.items():
        new_key = key2key(k)
        d.setdefault(new_key, 0)
        d[new_key] += v
    return qiskit.result.counts.Counts(d, time_taken=c.time_taken)
```

## 普通の量子もつれ回路を見てみる

この `LogicalQuantumCircuit` を使うともつれ回路はどのように見えるのであろうか？
まずは論理量子ビット + ビット反転エラー訂正用のエンコードを無効化してみよう:

```python
circuit = LogicalQuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
circuit.draw()
```

![](/images/dwd-qiskit10/003.png)

明らかにいつもの見なれた回路に余計なレーンが追加されているだけである。
次にエンコードを有効化してみよう:

```python
circuit = LogicalQuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.barrier()
circuit.encode()
circuit.barrier()
circuit.measure_all()
circuit.draw()
```

![](/images/dwd-qiskit10/004.png)

ちょっと見た目がつらくなったが、[Qiskit で遊んでみる (9) — Shor の符号#ビット反転の回路とエラー訂正回路](/derwind/articles/dwd-qiskit09#ビット反転の回路とエラー訂正回路) でのビット反転エラーの訂正回路を繋いだ上で測定しているだけである。

## 雑音チャネルを追加する

論理量子ビットに雑音を乗せる。今回は激しくもビット反転が 50% の確率でかかるような雑音チャネルを考える。

```python
def make_quantum_channel(n_qubits):
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import pauli_error

    p_error = 0.5
    quantum_channel = QuantumCircuit(n_qubits)
    for i in range(len(circuit.qubits)):
        quantum_channel.x(i)
        error = pauli_error([('X', p_error), ('I', 1 - p_error)])
        quantum_channel.append(Kraus(error), [i])

    return quantum_channel
```

## エラー訂正なしの場合

```python
circuit = LogicalQuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.barrier()
quantum_channel = make_quantum_channel(len(circuit.qubits))
circuit.compose(quantum_channel, [i*circuit.n_code for i in range(len(circuit.qubits))])
circuit.measure_all()
circuit.draw()
```

![](/images/dwd-qiskit10/005.png)

結果は分かりきっているが一応測定して可視化する:

```python
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

sim = AerSimulator()
result = sim.run(circuit.circuit).result()
counts = result.get_counts()
plot_histogram(counts2counts(counts, circuit.n_code), figsize=(6,4))
```

![](/images/dwd-qiskit10/006.png)

期待通りに混合状態となり、酷い事になってしまった。

## エラー訂正ありの場合

```python
circuit = LogicalQuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.barrier()
circuit.encode()
circuit.barrier()
quantum_channel = make_quantum_channel(len(circuit.qubits))
circuit.compose(quantum_channel, [i*circuit.n_code for i in range(len(circuit.qubits))])
circuit.barrier()
circuit.measure_all()
circuit.draw(scale=.75)
```

![](/images/dwd-qiskit10/007.png)

この回路を先ほどと同様に測定すると以下のようになる:

![](/images/dwd-qiskit10/008.png)

エラー訂正されてちゃんともつれ状態が観測されている。
ただ、素朴だった時の回路を再掲すると、エラーがあってそれを訂正する回路はこんなに単純な問題設定でも複雑怪奇になってしまった・・・

**素朴だった時の回路再掲**:

![](/images/dwd-qiskit10/003.png)

# まとめ

実際書いていてややこしかったのだが、以下のようなことをしたことになる。

- 3 量子ビットでのスタビライザ符号を実装した。
- 特にビット反転のエラー訂正の枠組みと共存する形での実装をした。
- 論理量子ゲートとしてクリフォードゲート $H$ と $CX$ を用いて、スタビライザ状態を維持したままの量子もつれ状態を作成し、これをスタビライザ符号化した。
- エラー訂正の回路を適用して測定することで、仮にスタビライザ状態が損なわれても自動で修復されることを見た。
- 結果、ちゃんと量子もつれ状態が観測された。

概要で触れたように、$T$ はクリフォードゲートになっていないので、スタビライザ符号の枠組みの中で普遍性は達成できていない。これについての「魔法状態」の話は明らかに本記事のレベルを逸脱するのでここでは触れない。

将来画期的な事実が見つかって、普遍性とエラー訂正の仕組みが容易に両立し、かつ計算コストも十分低く実現できることを期待したい。

# 参考文献

- [量子コンピュータと量子通信III －量子通信・情報処理と誤り訂正－](https://shop.ohmsha.co.jp/shop/shopdetail.html?brandcode=000000006439)
- [量子コンピューティング  基本アルゴリズムから量子機械学習まで](https://www.ohmsha.co.jp/book/9784274226212/)
- [スタビライザーによる量子状態の記述](https://whyitsso.net/physics/quantum_mechanics/stabilizer.html)
- [Proving Universality - Qiskit textbook](https://qiskit.org/textbook/ch-gates/proving-universality.html)
- [Clifford gates](https://en.wikipedia.org/wiki/Clifford_gates)
