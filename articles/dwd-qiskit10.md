---
title: "Qiskit で遊んでみる (10) — スタビライザ符号"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "QiskitAer", "poem", "Python"]
published: true
---

# 目的

スタビライザ符号についてちょっと勉強してみたので、何も分かってないけどコードを書いてみる。

- なお、書いたコードについてはとてもじゃないけど修正できる気はしない。何となく動いている状態。
- 直下の「スタビライザ符号概要」も書きたいことは書いているのだけど、長文すぎて正直口頭ではとても説明できない。
- とにかく忘れないうちに書き出してしまおう・・・という感じ。

# スタビライザ符号概要

以下に淡々とスタビライザ符号の概要を列挙する。見ても何も嬉しくはないただのメモ書きである。後で行う検証の実装の上では特に気にしなくても良い程度のものである。

- エラー訂正の仕組みを持った論理量子ビット
- その量子ビット間にはたらく論理量子ゲート

を作りたい。そのために以下のような事を考えるようなものらしい。

- **スタビライザ群** $\mathcal{S}$ なるものが量子状態のなすヒルベルト空間 $\mathcal{H}$ に作用してると考えて、$\mathcal{H} = V \oplus V^\perp$ と直交分解する。ここで、$V$ は $\mathcal{S}$-不変な (スタビライズされた) 部分空間とする。この $V$ を**スタビライザ状態**と呼ぶ。
- $V$ の中で論理量子ビットを実装すれば、計算中にエラーが生じた時 $V^\perp$ の元に化けるので、$\mathcal{S}$-不変でなくなってエラー検出できる。($\mathcal{S}$ の元を通して測定するとエラーを検出できる)
- このようにスタビライザ状態を一種のコードと見て論理的な量子ビットを実装することを**スタビライザ符号**と呼び、こういう枠組みのことをスタビライザ形式 (stabilizer formalism) と呼ぶ。

具体的には、3 量子ビットのスタビライザ状態を用いる場合、論理量子状態 $\ket{\bar{0}}$ を $\ket{000}$ で実装し、論理量子状態 $\ket{\bar{1}}$ を $\ket{111}$ で実装することができる。

また、エラー訂正の仕組みを保ったまま論理量子ゲートを適用したいので、

- 論理量子ゲートに採用される演算子はスタビライザ状態をスタビライザ状態にうつすようなものであることが望まれる。このような性質を持った演算子を**クリフォード演算子**と呼び、これらは群の構造を持っている。これをクリフォード群と呼ぶ。
- 論理量子ゲートとしての、$X$, $Z$, $H$, $CX$ を仮に $\bar{X}$, $\bar{Z}$, $\bar{H}$, $C\bar{X}$ と書く事にする。

通常の量子回路の場合、ある状態から別の任意の量子状態を近似するためのゲートセットの存在と構成が望まれる。そういうゲートセットが存在することを**普遍性**と呼び、例えば、1 量子ビットのゲート $H$ と $T$ および 2 量子ビットのゲート $CX$ の組み合わせはこの普遍性を持つゲートセットになっている。とすれば

- $H$, $T$, $CX$ がクリフォードゲートであれば、エラー訂正の仕組みとゲートセットの普遍性が両立して幸せになれる

となるが、これは否定的で $T$ がクリフォード演算子では**ない**。魔法状態という仕組みを使えば実質的なクリフォード演算子にできるが、魔法状態の作成の計算コストが大きすぎて現実的ではない。

# 論理量子ビットの実装

前述のような理論を述べたところであまり嬉しいわけでもないので、具体例だけ見る。

3 量子ビットのケースがわりと使い勝手が良いのでこのケースだけ見る。このケースではスタビライザ群として $\mathcal{S}_3 = \{ III, ZZI, IZZ, ZIZ \}$ がとれる。特に生成元は \{ ZZI, ZIZ \}$ である。これに対応するヒルベルト空間 $\mathbb{C}^{2^3}$ の中の $\mathcal{S}_3$-不変な部分空間、即ちスタビライザ状態は $V = \mathrm{span}\{ \ket{000}, \ket{111} \}$ となる。

ということで、この空間 $V$ をコードの空間に見立てて論理量子ビットを実装すれば良いのだが、概要で述べたように $\ket{\bar{0}}$ を $\ket{000}$ で実装し、論理量子状態 $\ket{\bar{1}}$ を $\ket{111}$ で実装すると何となく分かりやすい気がする。

今回 3 量子ビットだけ使うので、[Qiskit で遊んでみる (9) — Shor の符号#ビット反転の回路とエラー訂正回路](/derwind/articles/dwd-qiskit09#ビット反転の回路とエラー訂正回路) で見た「ビット反転」のエラー訂正の仕組みの考えと組み合わせる形で、以下のように符号回路を実装する。

- $\ket{\bar{0}} = \ket{000}$ の準備回路

![](/images/dwd-qiskit10/001.png)

- $\ket{\bar{1}} = \ket{111}$ の準備回路

![](/images/dwd-qiskit10/002.png)

# 論理量子ゲート

何故と言われても困るが、手計算で色々確認した限りでは以下のように論理ゲートを設定するとユニタリゲートとなり、また以下のように直感的に嬉しい気がする性質を持っていたのでこれらを採用した。

- $\bar{X} \bar{X} = \bar{Z} \bar{Z} = \bar{H} \bar{H} = \bar{I}$
- $\bar{H} \bar{X} \bar{H}  = \bar{Z}$
- $\bar{H} \bar{Z} \bar{H}  = \bar{X}$
- $\bar{H} \ket{\bar{0}} = \frac{1}{\sqrt{2}}(\ket{\bar{0}} + \ket{\bar{1}})$
- $\bar{H} \ket{\bar{1}} = \frac{1}{\sqrt{2}}(\ket{\bar{0}} - \ket{\bar{1}})$
- $C\bar{X}_{0,1} \bar{H}_0 \ket{\bar{0}} \ket{\bar{0}} = \frac{1}{\sqrt{2}}(\ket{\bar{0}} \ket{\bar{0}} + \ket{\bar{1}} \ket{\bar{1}} )$

これらの関係を満たすユニタリゲートの定義は以下である。基本的に行列の要素を対角成分に沿ってコピーする形での拡張であり、符号が反転する要素が対角成分に来ている場合、真ん中でぶつかるまでお互いに端からコピーしていく感じである:

$$
\begin{align*}
\bar{I} = \begin{pmatrix}
    1 & 0 & \ldots & 0 \\
    0 & 1 & \ldots & 0 \\
    0 & 0 & \ddots & 0 \\
    0 & 0 & \ldots & 1
\end{pmatrix}
\end{align*}
$$

$$
\begin{align*}
\bar{X} = \begin{pmatrix}
    0 & \ldots & 0 & 1 \\
    0 & \ldots & 1 & 0 \\
    0 & \ddots & 0 & 0 \\
    1 & \ldots & 0 & 0
\end{pmatrix}
\end{align*}
$$

$$
\begin{align*}
\bar{Z} = \begin{pmatrix}
    1 & 0 & \ldots & 0 & 0 \\
    0 & 1 & \ldots & 0 & 0 \\
    0 & 0 & \ddots & 0 & 0 \\
    0 & 0 & \ddots & -1 & 0 \\
    0 & 0 & \ldots & 0 & -1
\end{pmatrix}
\end{align*}
$$

$$
\begin{align*}
\bar{H} = \frac{1}{\sqrt{2}}(\bar{X} + \bar{Z})
\end{align*}
$$

$$
\begin{align*}
\mathrm{ZEROS} &= \mathrm{diag}[1, 1, \cdots, 1, 0, 0, \cdots, 0] \\
\mathrm{ONES} &= \mathrm{diag}[0, 0, \cdots, 0, 1, 1, \cdots, 1]
\end{align*}
$$

$$
\begin{align*}
C\bar{X} &= I \otimes \cdots \otimes I \otimes \bar{X} \otimes I \otimes \cdots \otimes I \otimes \mathrm{ONES} \otimes \cdots \otimes I \\
&+ I \otimes \cdots \otimes I \otimes \bar{I} \otimes I \otimes \cdots \otimes I \otimes \mathrm{ZEROS} \otimes \cdots \otimes I
\end{align*}
$$

# 量子回路を実装していく

何はともあれ、必要なモジュールを import する

```python
import qiskit
from qiskit import QuantumCircuit, ClassicalRegister
import qiskit.opflow as opflow
from qiskit.quantum_info.operators import Operator
import numpy as np
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
```

次に上のほうで書いた論理量子ゲートを実装する。行列の計算が面倒臭過ぎたので、`qiskit.opflow` を用いたが、それでも見通しは悪い:

```python
class LogicalGates:
    def __init__(self, n_code):
        self._I = None
        self._X = None
        self._Z = None
        self._extended_Zero_Zero = opflow.MatrixOp(np.diag([1]*2**(n_code-1)+[0]*2**(n_code-1)))
        self._extended_One_One = opflow.MatrixOp(np.diag([0]*2**(n_code-1)+[1]*2**(n_code-1)))

        self.n_code = n_code
        self._initialize(self.n_code)

    @property
    def I(self):
        return Operator(self._I.to_matrix())

    @property
    def X(self):
        return Operator(self._X.to_matrix())

    @property
    def Z(self):
        return Operator(self._Z.to_matrix())

    @property
    def H(self):
        return Operator((self._X.to_matrix() + self._Z.to_matrix())/np.sqrt(2))

    def CX(self, c, t, n_qubits):
        def _op_at(op1, loc1, op2, loc2, n_qubits):
            assert loc1 != loc2
            if loc1 == n_qubits-1:
                result = op1
                for i in reversed(range(n_qubits-1)):
                    if i == loc2:
                        result = result^op2
                    else:
                        result = result^self.I
            elif loc2 == n_qubits-1:
                result = op2
                for i in reversed(range(n_qubits-1)):
                    if i == loc1:
                        result = result^op1
                    else:
                        result = result^self.I
            else:
                result = opflow.I
                for i in reversed(range(n_qubits)):
                    if i == loc1:
                        result = result^op1
                    elif i == loc2:
                        result = result^op2
                    else:
                        result = result^self.I
            return result

        mat = _op_at(self._X, t, self._extended_One_One,   c, n_qubits).to_matrix() + \
              _op_at(self._I, t, self._extended_Zero_Zero, c, n_qubits).to_matrix()
        return Operator(mat)

    def _initialize(self, n_code):
        i = opflow.I
        for _ in range(n_code-1):
            i = opflow.I^i
        self._I = i

        x = opflow.X
        for _ in range(n_code-1):
            x = opflow.X^x
        self._X = x

        z = opflow.Z
        for _ in range(n_code-1):
            z = opflow.I^z
        self._Z = z
```

最後に論理量子ビットと論理量子ゲートを用いた論理量子回路を実装するが、あまりに酷いものになった:

```python
class LogicalQuantumCircuit:
    def __init__(self, qregs: int, n_code: int=3, encode: bool=False):
        self.n_code = n_code # length of code bits
        self.gates = LogicalGates(self.n_code)
        self._circuit = None
        self._qubits = []
        self.add_register(qregs)
        self.encoded = False
        self.error_correcting_circuit = None
        if encode:
            self._encode()

    def add_register(self, qregs: int):
        self._circuit = QuantumCircuit(qregs*self.n_code)
        enc_circuit, _ = self.make_bit_flip_code_circuits()

        for i in range(qregs):
            qubit = LogicalQubit(enc_circuit)
            self._qubits.append(qubit)

    def compose(self, other: QuantumCircuit, qubits: Optional[Sequence[int]] = None):
        self.circuit.compose(other, qubits, inplace=True)

    def _encode(self):
        for i, qubit in enumerate(self.qubits):
            self.circuit.compose(qubit.circuit, range(i*self.n_code, (i+1)*self.n_code), inplace=True)
        self.encoded = True

    ## decorators ##

    def _insert_error_correcting_circuit(f):
        def wrapper(*args):
            self = args[0]
            if self.error_correcting_circuit is not None:
                ec_circuit = self.error_correcting_circuit
                self.error_correcting_circuit = None
                for i in range(len(self.qubits)):
                    # correct errors
                    self.circuit.compose(ec_circuit, range(i*self.n_code, (i+1)*self.n_code), inplace=True)
                    # and encode again
                    enc_circuit, _ = self.make_bit_flip_code_circuits(True)
                    self.circuit.compose(enc_circuit, range(i*self.n_code, (i+1)*self.n_code), inplace=True)
            f(*args)
        return wrapper

    def _insert_error_correcting_circuit_but_no_encode(f):
        def wrapper(*args):
            self = args[0]
            if self.error_correcting_circuit is not None:
                ec_circuit = self.error_correcting_circuit
                self.error_correcting_circuit = None
                for i in range(len(self.qubits)):
                    # correct errors
                    self.circuit.compose(ec_circuit, range(i*self.n_code, (i+1)*self.n_code), inplace=True)
            f(*args)
        return wrapper

    def _save_error_correcting_circuit(f):
        def wrapper(*args):
            self = args[0]
            if self.encoded:
                # For, X, Z, H, CX
                # generators of stabilizer group for |000>, |111> are ZZI and ZIZ.
                # Clifford gates X, Z, H, CX map ZZI and ZIZ to themselves.
                # So error correcting circuits don't be affected.
                _, self.error_correcting_circuit = self.make_bit_flip_code_circuits()
            return f(*args)
        return wrapper

    def _insert_barrier(f):
        def wrapper(*args):
            self = args[0]
            self.barrier()
            return f(*args)
        return wrapper

    ################

    def barrier(self):
        self.circuit.barrier()

    @_insert_error_correcting_circuit
    @_save_error_correcting_circuit
    def h(self, qubit: int):
        self.circuit.append(self.gates.H, range(qubit*self.n_code, (qubit+1)*self.n_code))

    @_insert_error_correcting_circuit
    @_save_error_correcting_circuit
    def i(self, qubit: int):
        pass

    def id(self, qubit: int):
        self.i(qubit)

    @_insert_error_correcting_circuit
    @_save_error_correcting_circuit
    def x(self, qubit: int):
        self.circuit.append(self.gates.X, range(qubit*self.n_code, (qubit+1)*self.n_code))

    @_insert_error_correcting_circuit
    @_save_error_correcting_circuit
    def cx(self, control_qubit: int, target_qubit: int):
        self.circuit.append(self.gates.CX(control_qubit, target_qubit, self.num_qubits), range(self.num_qubits*self.n_code))

    @_insert_barrier
    @_insert_error_correcting_circuit_but_no_encode
    def measure_all(self):
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
    def num_qubits(self) -> int:
        return len(self.qubits)

    def make_bit_flip_code_circuits(self, reset=False):
        if self.n_code == 1:
            return QuantumCircuit(1), QuantumCircuit(1)
        elif self.n_code == 3:
            enc_circuit = QuantumCircuit(self.n_code)
            if reset:
                enc_circuit.reset(1)
                enc_circuit.reset(2)
            enc_circuit.cx(0, 1)
            enc_circuit.cx(0, 2)

            dec_circuit = QuantumCircuit(self.n_code)
            dec_circuit.cx(0, 1)
            dec_circuit.cx(0, 2)
            dec_circuit.ccx(1, 2, 0)

            return enc_circuit, dec_circuit
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.circuit)

def counts2counts(counts: qiskit.result.counts.Counts, n_code: Union[int,LogicalQuantumCircuit]):
    if isinstance(n_code, LogicalQuantumCircuit):
        n_code = n_code.n_code
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
circuit = LogicalQuantumCircuit(2, encode=False)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()
circuit.draw()
```

![](/images/dwd-qiskit10/003.png)

明らかにいつもの見なれた回路に余計なレーンが追加されているだけである。試しに測定しても普通の結果である。

```python
from qiskit_aer import AerSimulator

sim = AerSimulator()
result = sim.run(circuit.circuit).result()
counts = result.get_counts()
print(counts2counts(counts, circuit))
```

> {'00': 550, '11': 474}


次にエンコードを有効化してみよう:

```python
circuit = LogicalQuantumCircuit(2, encode=True)
circuit.h(0)
circuit.barrier()
circuit.cx(0, 1)
circuit.measure_all()
circuit.draw()
```

![](/images/dwd-qiskit10/004.png)

かなり見た目がつらくなったが、[Qiskit で遊んでみる (9) — Shor の符号#ビット反転の回路とエラー訂正回路](/derwind/articles/dwd-qiskit09#ビット反転の回路とエラー訂正回路) でのビット反転エラーの訂正回路を繋いでエラーを訂正しつつ、次のゲートに入る前にもう一度綺麗にエンコードする・・・という事を繰り返して測定しているだけである。ancilla 的なやつを一旦リセットしないと綺麗に再エンコードできないのでそうしたが、何かそれは違う気がする。宿題とする。

## 雑音チャネルの回路を見てみる

論理量子ビットに雑音を乗せる。今回は激しくもビット反転が 50% の確率でかかるような雑音チャネルを考える。

```python
def make_noisy_channel(circuit: LogicalQuantumCircuit, locs=[]):
    from qiskit.quantum_info import Kraus
    from qiskit_aer.noise import pauli_error

    p_error = 0.5
    quantum_channel = QuantumCircuit(circuit.num_qubits*circuit.n_code)
    for i in range(circuit.num_qubits):
        loc = i*circuit.n_code
        if len(locs) > i:
            loc += locs[i]
        quantum_channel.x(loc)
        error = pauli_error([('X', p_error), ('I', 1 - p_error)])
        quantum_channel.append(Kraus(error), [loc])

    return quantum_channel

def append_noisy_channel(circuit: LogicalQuantumCircuit, locs=[]):
    channel = make_noisy_channel(circuit, locs=locs)
    circuit.circuit.compose(channel, range(circuit.num_qubits*circuit.n_code), inplace=True)
```

# 実験

以下の仮定をする:

- 雑音はゲート間でのみ発生する。
- ゲート間においては雑音は高々 1 物理量子ビットにしか影響を与えない。
- 符号化回路およびエラー訂正回路における雑音は無視できるものとする。

## エンコードなしエラー訂正なしの場合

```python
circuit = LogicalQuantumCircuit(2, encode=False)
circuit.h(0)
circuit.barrier()
append_noisy_channel(circuit, [0, 1])
circuit.barrier()
circuit.cx(0, 1)
circuit.barrier()
append_noisy_channel(circuit, [2, 0])
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

## エンコードありエラー訂正ありの場合

```python
circuit = LogicalQuantumCircuit(2, encode=True)
circuit.h(0)
circuit.barrier()
append_noisy_channel(circuit, [0, 1])
circuit.barrier()
circuit.cx(0, 1)
circuit.barrier()
append_noisy_channel(circuit, [2, 0])
circuit.measure_all()
circuit.draw()
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
- 論理量子ゲートとしてクリフォードゲート $\bar{H}$ と $C\bar{X}$ を用いて、スタビライザ状態を維持したままの量子もつれ状態を作成し、これをスタビライザ符号化した。
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
