---
title: "Qiskit で遊んでみる (20) — Qiskit Optimization での QAOA"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "量子機械学習", "機械学習", "poem"]
published: true
---

# 目的

この記事は大分昔に書いた[記事](https://blueqat.com/derwind/e792298f-8b9d-428b-b84e-b57a52546a1b) がベースだったりする[^1]。

[^1]: 自分の中では大分前なのだが、どうやら 4 ヶ月程度しか経ってないらしい・・・。

この時はちょっと Qiskit の古い API を使ったのだが、今回はそれも含めて最新の API にまでアップデートしたい。**なお、実用性は狙いではない。**

量子コンピューティングの基礎を手を使って学ぶ上での良書に「[IBM Quantumで学ぶ量子コンピュータ](https://www.shuwasystem.co.jp/book/9784798062808.html)」があるが、同書 pp.203-210 に掲載された QAOA のコードは新しい Qiskit では動かない。[Qiskit Aqua](https://github.com/qiskit-community/qiskit-aqua) という Deprecated になってしまったパッケージを使っているからである。このコードをQiskit 0.44.0 以降で動くところまでアップデートしたい。

また、大規模疑似量子アニーリング SDK の [TYTAN SDK](https://github.com/tytansdk/tytan) もあるので、併せて活用する[^2]。

[^2]: 本記事を書いた時点ではよく分かっていなかったが、この 4 ヶ月で勉強して基本的なところを習得した。

# まずは TYTAN SDK で解いてみる

ここからは書籍が手元にある前提とし、記号類も極力書籍に合わせる。まずは TYTAN SDK をインストールする。

```sh
! pip install -q git+https://github.com/tytansdk/tytan.git
```

チュートリアルは [tytan_tutorial](https://github.com/tytansdk/tytan_tutorial) に色々あるので、適宜参照されたい。

必要なモジュールを import する。

```python
from __future__ import annotations

from tytan import symbols_list, Compile, sampler
```

QUBO の定式化を実装する。

```python
q = symbols_list(4, "q{}")
C = (q[0] + q[1] + q[2]) ** 2 + \
    q[3] ** 2 + \
    (q[0] + q[1] + q[2]) ** 2 + \
    q[0] ** 2 + \
    q[3] ** 2 + \
    (q[1] + q[2]) ** 2 + \
    q[0] ** 2 + \
    q[3] ** 2 + \
    (q[1] + q[2] + q[3]) ** 2

Q, offset = Compile(C).get_qubo()
print(Q)
print(offset)
```

## 制約条件なしで解く

サンプラで QUBO 最適化を実行する。

```python
sampler.SASampler()

result = solver.run(Q, shots=100)
print(result)
```

> [[{'q0': 0, 'q1': 0, 'q2': 0, 'q3': 0}, 0.0, 100]]

となって、実は解けない。制約条件が必要なのである。後半で扱う QAOA では $XY$ ミキサーというものがこれを担当する。

## 制約条件を付加して解く

重みは適当なのだが、car1 が候補 1 と候補 2 を同時に通る、または両方とも通らないような解もでてきてしまうので、必ず片方通るように強制する。

```python
# 車ごとに片方の道だけ通って欲しい。
C_car_unique_path = (2*q[0] - 1) * (2*q[1] - 1) + (2*q[2] - 1) * (2*q[3] - 1)

weight = 3

C_total = C + weight * C_car_unique_path
```

そして、制約条件を適当に重み付けして付加したコスト関数から QUBO を作って最適化をすると、

```python
Q, offset = Compile(C_total).get_qubo()

sampler.SASampler()

result = solver.run(Q, shots=100)
print(result)
```

> [[{'q0': 1, 'q1': 0, 'q2': 0, 'q3': 1}, -4.0, 81], [{'q0': 0, 'q1': 0, 'q2': 1, 'q3': 0}, -2.0, 12], [{'q0': 0, 'q1': 1, 'q2': 0, 'q3': 0}, -2.0, 7]]

のように解ける。100 回のサンプリングのうち 81 回を占める解は `q0=1`, `q1=0`, `q2=0`, `q3=1` である。

と言う事で、書籍と同様に **car1は候補1を、car2は候補2を通ると良い** ということが分かった。なお、重みをより重くするなどすると、少し結果に影響があるので試してみるのも面白いと思う。

# Qiskit (QAOA) で解いてみる

ここからが本題の「書籍のコードのアップデート」になる。まずは必要なパッケージをインストールする。諸事情というか Qiskit Optimization 0.6.0 を使うのだが、正式にはリリースされていないバージョンなので、GitHub からインストールする。

## QUBO の準備

```sh
! pip install qiskit pylatexenc
! pip uninstall -y qiskit-optimization
! pip install git+https://github.com/qiskit-community/qiskit-optimization.git
```

必要なモジュールを import する。

```python
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
```

折角 TYTAN SDK をインストールしているので、これを使って QUBO の線形項と二次項の係数の計算をさぼりたいため、ヘルパー関数を定義する。

```python
LinearInfo = dict[str, int | float]
QuadraticInfo = dict[tuple[str, str], int | float]

def make_linear_and_quadratic(Q: dict) -> tuple[LinearInfo, QuadraticInfo]:
    linear: LinearInfo = {}
    quadratic: QuadraticInfo = {}
    for key, coeff in Q.items():
        key0, key1 = key
        if key0 == key1:
            linear[key0] = coeff
        else:
            quadratic[(key0, key1)] = coeff
    return linear, quadratic
```

TYTAN SDK で制約なしの QUBO 式 `Q` を計算してこのヘルパー関数に通す。

```python
q = symbols_list(4, "q{}")
C = (q[0] + q[1] + q[2]) ** 2 + \
    q[3] ** 2 + \
    (q[0] + q[1] + q[2]) ** 2 + \
    q[0] ** 2 + \
    q[3] ** 2 + \
    (q[1] + q[2]) ** 2 + \
    q[0] ** 2 + \
    q[3] ** 2 + \
    (q[1] + q[2] + q[3]) ** 2

Q, offset = Compile(C).get_qubo()

linear, quadratic = make_linear_and_quadratic(Q)
print(linear, quadratic)
```

> {'q0': -2.0, 'q1': -2.0, 'q2': -2.0, 'q3': -2.0} {('q1', 'q3'): 2.0, ('q0', 'q2'): 4.0, ('q1', 'q2'): 8.0, ('q2', 'q3'): 14.0, ('q0', 'q1'): 16.0}

[IBM Quantumで学ぶ量子コンピュータ](https://www.shuwasystem.co.jp/book/9784798062808.html) pp.206-207 と見比べると、書籍の `linear` と `quadratic` と同じものが求まっていることが分かると思う。

計算がさぼれたところで、再び書籍の書き方に戻る。

```python
qubo = QuadraticProgram()
qubo.binary_var('q0')
qubo.binary_var('q1')
qubo.binary_var('q2')
qubo.binary_var('q3')

qubo.minimize(linear=linear, quadratic=quadratic)
print(qubo)
```

> minimize 4\*q0\*q1 + 4\*q0\*q2 + 8\*q1\*q2 + 2\*q1\*q3 + 2\*q2\*q3 + 4\*q0 + 4\*q1 + 4\*q2 + 4\*q3 (4 variables, 0 constraints, '')

上記はもうちょっと見やすいかもしれない以下のような出力もできる。

```python
print(qubo.export_as_lp_string())
```

> \ This file has been generated by DOcplex
> \ ENCODING=ISO-8859-1
> \Problem name: CPLEX
> 
> Minimize
>  obj: 4 q0 + 4 q1 + 4 q2 + 4 q3 + [ 8 q0\*q1 + 8 q0\*q2 + 16 q1\*q2 + 4 q1\*q3
>       + 4 q2*q3 ]/2
> Subject To
> 
> Bounds
>  0 <= q0 <= 1
>  0 <= q1 <= 1
>  0 <= q2 <= 1
>  0 <= q3 <= 1
> 
> Binaries
>  q0 q1 q2 q3
> End

## QUBO をイジングハミルトニアンに変換する

イジング変数 $\sigma \in \{-1, +1\}$ とバイナリ変数 $x \in \{0, 1\}$ の間には例えば $\sigma = 1 - 2x$ と言った対応付けができる。これを用いて、バイナリ変数からなる QUBO 式をイジング変数の式に変換した後に、$\sigma$ を Pauli $Z$ ゲートに置換するという “量子化” を行い、イジングハミルトニアンを得る。

この辺は API だけで変換することができ、またこういうものといった感じである。

```python
qubit_op, offset = qubo.to_ising()

print(qubit_op.to_list())
print(f'{offset=}')
```

> [('IIIZ', (-4+0j)), ('IIZI', (-5.5+0j)), ('IZII', (-5.5+0j)), ('ZIII', (-3+0j)), ('IIZZ', (1+0j)), ('IZIZ', (1+0j)), ('IZZI', (2+0j)), ('ZIZI', (0.5+0j)), ('ZZII', (0.5+0j))]
> offset=13.0

## 初期状態の準備

続けて初期状態を準備する。

```python
initial_state_circuit = QuantumCircuit(4)
initial_state_circuit.h(0)
initial_state_circuit.cx(0, 1)
initial_state_circuit.x(0)
initial_state_circuit.h(2)
initial_state_circuit.cx(2, 3)
initial_state_circuit.x(2)

display(initial_state_circuit.draw('mpl'))
```

![](/images/dwd-qiskit20/001.png)

この量子回路は以下のような状態を準備する。

$$
\begin{align*}
\ket{\psi_0} = \frac{1}{2} (\ket{0101} + \ket{0110} + \ket{1001} + \ket{1010})
\end{align*}
$$

## 最適化を実行する

続けて、QAOAを解いてみる。

```python
sampler = Sampler()
optimizer = COBYLA()
step = 1
# XY ミキサーの定義
mixer = SparsePauliOp(["XXII", "YYII", "IIXX", "IIYY"], [1/2, 1/2, 1/2, 1/2])

qaoa = QAOA(
    sampler,
    optimizer,
    reps=step,
    initial_state=initial_state_circuit,
    mixer=mixer,
)
result = qaoa.compute_minimum_eigenvalue(qubit_op)
print(result)
```

> {   'aux_operators_evaluated': None,
>     'best_measurement': {   'bitstring': '1001',
>                             'probability': 0.8981221499465242,
>                             'state': 9,
>                             'value': (-5+0j)},

「1001」という TYTAN SDK でも見えた列が見えているので、何かしら解けていそうな感じである。念のために `AerSimulator` で状態の分布を確認してみよう。

```python
opt_circuit = result.optimal_circuit.bind_parameters(result.optimal_parameters)

sim = AerSimulator()
t_qc = transpile(opt_circuit, backend=sim)
counts = sim.run(t_qc).result().get_counts()
print(counts)
```

> {'0101': 8, '1010': 81, '0110': 30, '1001': 905}

ということで、疑似量子アニーリング版と同様に **car1は候補1を、car2は候補2を通ると良い** らしいことが分かった。こちらもミキサー（ミキシングハミルトニアン）を渡さない場合にどうなるか考えてみて、書籍と照らし合わせると良いと思う。なお、疑似量子アニーリングにしても QAOA にしてもメタヒューリスティックなアルゴリズムなので、厳密解が得られるとは限らない・・・ことに注意する必要がある[^3]。

[^3]: 実際上記を何度も実行すると、`counts` の内容は結構ばらけていることが分かる。`step` を増やすと状況はマシになるかもしれないが、計算時間がそれなりにかかるようになる。`p=3` ぐらいにするとかなり良い感じで解が得られる。

# まとめ

単に書籍をそのまま書き直しただけだが、大体これでコードは動く状態にアップデートできると思う。念の為に Qiskit のパッケージの各バージョンを出力しておく。

QAOA に興味がわいた場合、

- [Factoring integers with sublinear resources on a superconducting quantum processor](https://arxiv.org/abs/2212.12372)

を読んでみるのも面白いかもしれない。

```python
from qiskit import __qiskit_version__

__qiskit_version__
```

> {'qiskit-terra': '0.25.0', 'qiskit': '0.44.0', 'qiskit-aer': '0.12.2', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.6.0', 'qiskit-machine-learning': '0.6.1'}
