---
title: "Qiskit で遊んでみる (19) — VQE を今北産業する"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "量子機械学習", "機械学習", "poem"]
published: true
---

# 目的

NISQ 量子コンピュータの代表的なアルゴリズムの 1 つに VQE (Variational Quantum Eigensolver) というものがある。名前も凄そうだが、解説が圧倒される難しいものが多い。これについて [今北産業](https://dic.nicovideo.jp/a/%E4%BB%8A%E5%8C%97%E7%94%A3%E6%A5%AD) の精神で迫ってみたい。

# 結論

VQE とは要するに以下 (3 行) である:

```python
qc = QuantumCircuit(1)
qc.ry(Parameter("θ"), 0)
print(VQE(Estimator(), qc, SPSA()).compute_minimum_eigenvalue(Pauli("Z")).optimal_parameters)
```

> {Parameter(θ): 3.141592653589793}

# 詳細

VQE の話は概ね以下のような解説を伴う。

- 量子化学計算
    - Full CI
    - 量子位相推定
- シュレーディンガー方程式とハミルトニアン
- 第二量子化
- ハミルトニアンの基底状態の固有値

ところがこの辺は、問題の定義やどうしてこういう数理モデルを選択するのか？といった部分の話である。よって、VQE のメインルーチンからは切り離すことが可能である。

VQE のメインルーチンは用は深層学習の最適化ルーチンと本質的には同じで、「与えられたコスト関数の最小値を求める数値計算」である。

冒頭で書いたコードはあまりに乱暴だが、もう少し丁寧に書くと以下のようになる。

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import SPSA


qc = QuantumCircuit(1)
qc.ry(Parameter("θ"), 0)

estimator = Estimator()  # コスト関数を定義するための道具
operator = Pauli("Z")  # 解析対象
optimizer = SPSA()  # オプティマイザ

vqe = VQE(estimator, qc, optimizer)  # train loop を回す訓練器
result = vqe.compute_minimum_eigenvalue(operator)

print(result.optimal_parameters)
```

数式的には、

$$
\begin{align*}
f(\theta) &= (1 \quad 0) \begin{pmatrix}
  \cos(\frac{\theta}{2}) & \sin(\frac{\theta}{2}) \\
  - \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
\end{pmatrix}
\begin{pmatrix}
  1 & 0 \\
  0 & -1
\end{pmatrix}
\begin{pmatrix}
  \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) \\
  \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
\end{pmatrix}
\begin{pmatrix}
1 \\
0
\end{pmatrix} \\
&= (1 \quad 0) \begin{pmatrix}
  \cos(\frac{\theta}{2}) & \sin(\frac{\theta}{2}) \\
  - \sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2})
\end{pmatrix}
\begin{pmatrix}
  \cos(\frac{\theta}{2}) \\
  -\sin(\frac{\theta}{2})
\end{pmatrix} \\
&= \cos \theta
\end{align*}
$$

をコスト関数として、これが最小になるように、$\theta$ を最適化することになる。自明であるが $\theta = \pm \pi$ で最小値 $-1$ をとる。よって、冒頭に戻ると、

> {Parameter(θ): 3.141592653589793}

が得られることにつながる。

コスト関数 $f(\theta) = \cos \theta$ が与えられたところから話を始めると早い[^1]。

[^1]: ここまでは数理モデルを作る話なので。

- 適当に $\theta$ の初期値を決める
- 適当なオプティマイザで $\theta$ を逐次的に更新する
- 損失関数値が変動しなくなれば最適化完了

というどこかで見たような話になる。

NISQ 量子コンピュータの状況に適合するオプティマイザということで SPSA (Simultaneous Perturbation Stochastic Approximation) を用いているが、信頼できる勾配が計算できる理想的な状況を仮定するなら Adam で最適化しても問題ない。**完全に深層学習の最適化手法** である。

# まとめ

肉付けするとやっぱり何か小難しくなるが、極力肉を削って VQE のメインルーチンに迫ってみた。

量子機械学習にも色々種類はあるが、上記のようなものは要は「**量子力学的な観点で作ったコスト関数** を、**深層学習で見られるような最適化手法で最適化する** こと」ということである。

量子化学計算の場合はモロに量子力学的な問題であるが、巡回セールスマン問題など経路最適化問題などは特に量子力学的なわけではない。ところが、問題の定式化 (数式化) については量子計算の枠組みに乗せることが可能である。**量子計算にメリットを見出すのであれば** 定式化を量子計算の流儀で行い、深層学習と同様の最適化で数値解を求めることができる。
