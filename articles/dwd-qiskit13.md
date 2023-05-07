---
title: "Qiskit で遊んでみる (13) — 量子位相推定"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem"]
published: false
---

# 目的

[量子フーリエ変換を眺める](/derwind/articles/dwd-qft-numpy) で量子フーリエ変換に触れたので、その応用として量子位相推定についてまとめておきたい。

簡単な近似計算を CPU 上で行い、より精度を高めた近似を GPU 上で行ってみたい。

# 量子位相推定とは

[量子位相推定](https://ja.learn.qiskit.org/course/ch-algorithms/quantum-phase-estimation) に細かい説明があるので、二番煎じになるのだが、「与えられたユニタリ行列 $U$ とその (正規化された) 固有ベクトル $\ket{\psi}$ だけが分かっている場合に対応する固有値を求めるアルゴリズム」である。「位相なのに固有値？」という気持ちはするが、ユニタリ行列の場合、固有値の絶対値は 1 になるので[^1]、すべての固有値は $\exp(2 \pi i \theta)$ の表示で書ける。このアルゴリズムは位相 $\theta$ の部分を推定するアルゴリズムである。

[^1]: $U \ket{\psi} = \lambda \ket{\psi}$ なので、$\bra{\psi} U^\dagger = \lambda^* \bra{\psi}$ である。積をとると $1 = \braket{\psi | U^\dagger U | \psi} = |\lambda|^2 \braket{\psi | \psi}$ となるので、$|\lambda| = 1$ である。

# 量子フーリエ変換おさらい

量子フーリエ変換をモジュールとして利用するのでここでおさらいをする。

[量子フーリエ変換を眺める](/derwind/articles/dwd-qft-numpy) では省いたのだが、Qiskit textbook の [量子フーリエ変換](https://ja.learn.qiskit.org/course/ch-algorithms/quantum-fourier-transform) を見ていくと、「4. 量子フーリエ変換」において、QFT の具体的なテンソル積分解の式が掲載されている。なお、右端が 0 版目の量子ビットで、左端が $n-1$ 版目の量子ビットになっている。この式を踏まえつつ、入力の非負の整数 $x$ に対し、$n$ 桁の 2 進数展開を考えるものとし、QFT の式を書き下すと以下のようになる:

$$
\begin{align*}
\operatorname{QFT}(\ket{x}) &= \frac{1}{\sqrt{N}} (\ket{0} + e^{\frac{2\pi i}{2^1} x} \ket{1}) \otimes (\ket{0} + e^{\frac{2\pi i}{2^2} x} \ket{1}) \otimes \cdots \otimes (\ket{0} + e^{\frac{2\pi i}{2^n} x} \ket{1}) \\
&= \bigotimes_{k=1}^n \frac{1}{\sqrt{2}} (\ket{0} + e^{\frac{2\pi i}{2^k} x} \ket{1}) \\
&= \bigotimes_{k=1}^n P_{n-k} \left( \frac{2\pi}{2^k} x \right) \ket{+}^{\otimes n}
\tag{1}
\end{align*}
$$
ここで $P$ は位相ゲートであり、$P_i\ (0 \leq i < n)$ は $i$ 番目の量子ビットに作用する。

# やりたいこと (量子位相推定のアルゴリズム概略)

量子位相推定のキーとなるテクニックの 1 つは位相キックバックである。量子ビット $q$ と補助量子ビットとしてユニタリ $U$ の固有ベクトル $\ket{\psi}$ を準備したものを用意して、この 2 者間に制御ユニタリゲートを適用すると、位相キックバックの形で $q$ 側に固有値の位相 $\theta$ の情報が

$$
\begin{align*}
P_{n-k} \left( \frac{2 \pi}{2^k} \ket{\theta} \right) \ket{+}
\tag{2}
\end{align*}
$$

として現れる。これが要点のほとんどすべてである。

---

(2) 式を (1) 式と見比べると、QFT で現れる一部分になっていることが分かる。よって、$n$ 個の量子ビットで同様のことをしてテンソル積をとると、$QFT(\theta)$ と等価な状態が生成できることになる。後は逆 QFT 回路を作用させると $\theta$ が求まるというわけである。

まとめると:

1. **位相キックバックを用いて $P_{n-k} \left( \frac{2\pi}{2^k} \theta \right) \ket{+}$ を出力する回路を作り出す。**
2. **この回路を並列に $n$ 個並べてテンソル積の形で $\bigotimes_{k=1}^n P_{n-k} \left( \frac{2\pi}{2^k} \theta \right) \ket{+}^{\otimes n}$ を出力する回路にする。**
3. **逆量子フーリエ変換 $\operatorname{QFT}^\dagger$ を作用させて $\ket{\theta}$ を測定する。**

となる。

実際には少々これは正しくなくて、$\theta$ そのものではなく**整数値** $2^n \theta$ が求まることになる。これを $2^n$ で割る事で、有限 2 進小数 $0.b_{n-1} b_{n-2} \cdots b_1 b_0\ {}_\text{(bin)}$ の形で $\theta$ が推定される。2 進数の意味で推定精度の桁を 1 段階上げるためには 1 量子ビット追加する必要がある。

なお、**量子ビット数に対して必要なゲート数は $\mathcal{O}(2^n)$ + 逆 QFT の回路の長さくらい**になるので、回路深度についての注意は必要である。

# やってみる (数式)

まずは数式の形で追いかける。唐突だが、$\ket{+}\ket{\psi}$ に制御 $U$ ゲートを作用させることを考える:

$$
\begin{align*}
CU \ket{+}\ket{\psi} &= \frac{1}{\sqrt{2}} (\ket{0}\ket{\psi} + \ket{1} U\ket{\psi}) \\
& = \frac{1}{\sqrt{2}}(\ket{0}\ket{\psi} + \ket{1} e^{2 \pi i \theta} \ket{\psi}) = (P \left( 2 \pi \theta \right) \ket{+}) \ket{\psi}
\end{align*}
$$

となって、位相ゲートを $\ket{+}$ に作用させたかのような結果が得られる (位相キックバック)。

これを $j$ 回繰り返すと

$$
\begin{align*}
(CU)^j \ket{+}\ket{\psi} = CU^j \ket{+}\ket{\psi} = (P \left( 2 \pi j \theta \right) \ket{+}) \ket{\psi}
\tag{3}
\end{align*}
$$

となる。ここで (1) 式と比較すると、$2 \pi j \theta$ ではなく、$\frac{2 \pi}{2^k} \theta$ であってくれたら・・・という状況に気づく。と言う事で変数変換をする。

$$
\begin{align*}
2 \pi j \theta = \frac{2 \pi}{2^n} j 2^n \theta = \frac{2 \pi}{2^k} \varphi
\end{align*}
$$
ここで、$j = 2^{n-k}$、$\varphi = 2^n \theta$ とおいた。

この変数変換を踏まえて (3) 式を書き直すと

$$
\begin{align*}
(CU)^{2^{n-k}} \ket{+}\ket{\psi} = \left( P \left( \frac{2 \pi}{2^k} \varphi \right) \ket{+}  \right) \ket{\psi} \quad (1 \leq k \leq n)
\tag{3'}
\end{align*}
$$

となる。ほぼ QFT の形が得られた。

---

(1) 式と (3') 式を見比べると、量子ビットを $n$ 個並べて $\bigotimes_{k=1}^n$ を適用すれば良いことが分かる。つまり、$q_n$ に $\ket{\psi}$ が用意されているとして、

$$
\begin{align*}
\bigotimes_{k=1}^n (CU_{n-k,n})^{2^{n-k}} \ket{+}^{\otimes n} \ket{\psi} &= \bigotimes_{k=1}^n P_{n-k} \left( \frac{2 \pi}{2^k} \varphi \right) \ket{+}^{\otimes n} \ket{\psi} \\
&= \operatorname{QFT} (\varphi) \ket{\psi}
\tag{4}
\end{align*}
$$

である。

---

(4) 式に逆 QFT を適用して計算基底で測定することで、$\varphi$ の近似値として、$\sum_{\ell=0}^{n-1} 2^\ell b_\ell$ が求まる。ここで、$b_\ell \in \{0, 1\}$ である。気持ちとしてはこれが $\varphi$ であるので、

$$
\begin{align*}
2^n \theta = \varphi \approx \sum_{\ell=0}^{n-1} 2^\ell b_\ell
\end{align*}
$$

であり、両辺を $2^n$ で割って、

$$
\begin{align*}
\theta = \frac{\varphi}{2^n} \approx \sum_{\ell=0}^{n-1} \frac{b_\ell}{2^{n-\ell}} = 0.b_{n-1} b_{n-2} \cdots b_1 b_0 \ {}_\text{(bin)}
\end{align*}
$$

という、$\theta$ の近似的な 2 進小数展開が得られることになる。

以上が数式を用いた量子位相推定の解説となる。

# やってみる (実装)

(4) 式を実装して、逆 QFT の回路を接続すれば実装は完了である。つまり、以下のようにすれば良い:

```python
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
from qiskit.circuit.library import QFT
from qiskit.extensions import UnitaryGate
from qiskit_aer import AerSimulator

# 数式でいうユニタリ行列 U をランダムに作る。
unitary = qi.random_unitary(2, seed=1234)
unitary
```

> Operator([[-0.65182701+0.35104045j, -0.06872086+0.66870741j],
>           [ 0.30111103-0.60101938j,  0.3613751 +0.64615469j]],
>          input_dims=(2,), output_dims=(2,))

再現性のために、`seed` を固定してランダムなユニタリ行列を作った。続けて、

```python
# U の固有値と固有ベクトルを得る。
w, vec = np.linalg.eig(unitary.data)
w = w[0]
vec = vec[:, 0]
# np.allclose(unitary.data @ vec, w * vec) # => True

arg = np.log(w)
arg = arg.imag * 1.j
print(arg)
theta = (arg / (2*np.pi) * (-1j)).real
assert 0 <= theta < 1
print(theta)
```

> 2.878968619668155j
> 0.4582020868266377

最終的には量子位相推定では後者の `0.4582020868266377` の近似値が求まることになる。

```python
# ユニタリ行列をゲートにする。
unitary_gate = UnitaryGate(unitary, label='random_unitary')
unitary_gate.name = 'random_unitary'
```

```python
%%time

n_qubits = 4

qc = QuantumCircuit(n_qubits + 1)
for i in range(n_qubits):
    qc.h(i)

# 固有ベクトルを補助量子ビットの部分に用意する。
qc.initialize(vec, n_qubits)

for i in range(n_qubits):
    for _ in range(2**i):
        # ユニタリゲートを制御ユニタリゲートとして回路に追加する。
        qc.append(unitary_gate.control(), [i, n_qubits])
    qc.barrier()

iqft_circuit = QFT(n_qubits).inverse()
qc = qc.compose(iqft_circuit, list(range(n_qubits)))
qc.measure_all()
qc.draw(scale=0.4, fold=75)
```

![](/images/dwd-qiskit13/001.png)

以上で量子回路の実装は完了である。先に触れた通り、回路の深さは大凡 $2^\text{n\_qubits}$ + 逆 QFT 回路の深さになっている。

---

測定を行って $\varphi$ を推定しよう。今回は状態ベクトルで直接求める。

まず、シミュレータが解釈できるゲートになるまで `.decompose` を適用する。何回適用すれば良いかよく分からないが、今回のケースでは 6 回だった。

```python
%%time

qc1 = qc.decompose().decompose().decompose().decompose().decompose().decompose()
```

“測定” を行い、最も多く観測された結果を取得する:

```python
%%time

sim = AerSimulator()
counts = sim.run(qc1).result().get_counts()

counts_items = sorted(counts.items(), key=lambda k_v: -k_v[1])
print(counts_items)
```

> [('00111', 547), ('01000', 166), ('10111', 120), ('00110', 38), ('11000', 27), ('01001', 20), ('00101', 13), ('01011', 13), ('01010', 12), ('00011', 8), ('10100', 7), ('01100', 7), ('11001', 6), ('00100', 6), ('00010', 5), ('10110', 5), ('01110', 4), ('00000', 3), ('10010', 2), ('10011', 2), ('11011', 2), ('11100', 2), ('11010', 2), ('00001', 2), ('10001', 1), ('11111', 1), ('10101', 1), ('11101', 1), ('01101', 1)]

思ったより圧倒的な頻度で答えが得られていないので、ノイズがあると少々怖い。

```python
state, count = counts_items[0]

estimated_theta = int(state, 2) / (2**n_qubits)
print(f'estimated={estimated_theta}, real theta={theta}')
```

> estimated=0.4375, real theta=0.4582020868266377

まぁまぁの精度で $\theta$ の近似値が得られたのではないだろうか？

# やってみる (GPU 計算)

ここまでは CPU 上での計算で、わりとすぐに完了する。が、精度面では恐らく不満も残るものであると思う。そこで精度をあげて、12 量子ビットでシミュレーションを行ってみたい。この規模になると CPU だとしんどいと思われるので、GPU を使う。[Qiskit で遊んでみる (7) — Qiskit Aer GPU](https://zenn.dev/derwind/articles/dwd-qiskit07) に書いたような手順で Colab 上で GPU 対応 Qiskit Aer を実行できる。なお、この記事は現時点では古くなっているので、適宜読み替える必要がある。

適当にセルを抜粋して実行時間を掲載すると以下のような結果であった:

...

```python
%%time

n_qubits = 12

qc = QuantumCircuit(n_qubits + 1)
for i in range(n_qubits):
    qc.h(i)
qc.initialize(vec, n_qubits)

for i in range(n_qubits):
    for _ in range(2**i):
        qc.append(unitary_gate.control(), [i, n_qubits])
    qc.barrier()
    print(i)

iqft_circuit = QFT(n_qubits).inverse()
qc = qc.compose(iqft_circuit, list(range(n_qubits)))
qc.measure_all()
print(len(qc))
```

> 4135
> CPU times: user 48.9 s, sys: 225 ms, total: 49.1 s
> Wall time: 51.1 s

```python
%%time

qc1 = qc.decompose().decompose().decompose().decompose().decompose().decompose()
print(len(qc1))
```

> 57731
> CPU times: user 1min 27s, sys: 300 ms, total: 1min 27s
> Wall time: 1min 28s

`.decompose` 前は 4235 だった深さが、57731 にまで増えているのもつらいところである。つまり理論上は $\mathcal{O}(2^n)$ の深度でも、シミュレータや実機上で実行できるゲートにまでトランスパイルして分解すると、遥かに深い可能性がある。また、ゲートの深さが指数関数的に増えるので、量子ビット数を 1 つ増やすごとに、量子回路の構築時間が倍々で増えるかもしれない。

```python
%%time

sim = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
counts = sim.run(qc1).result().get_counts()

counts_items = sorted(counts.items(), key=lambda k_v: -k_v[1])
print(counts_items[:5])
```

> [('0011101010101', 719), ('1011101010101', 157), ('0011101010100', 51), ('0011101010110', 17), ('1011101010100', 15)]
> CPU times: user 10min 35s, sys: 2.31 s, total: 10min 37s
> Wall time: 10min 24s

GPU を使ったものの結構時間がかかった。1 つの理由としては、状態ベクトルの GPU シミュレーションは常に GPU を使うわけではなく、CPU に戻って、次のゲートの投入、次のゲートの投入・・・という処理を回す「CPU-GPU ハイブリッド」な感じになるからかもしれない。一瞬 GPU を使って、また一瞬 GPU を使ってと、深層学習に比べて遥かに穏やかな GPU の使い方をしている。

```python
state, count = counts_items[0]

estimated_theta = int(state, 2) / (2**n_qubits)
print(f'estimated={estimated_theta}, real theta={theta}')
```

> estimated=0.458251953125, real theta=0.4582020868266377

4 量子ビットの時よりはかなりマシな精度で推定できた。但し、量子ビット数を増やしても 2 進小数の桁の精度しか上がらないので、10 進小数と比較すると精度の改善が緩やかであることには注意したい。

大雑把に 12 量子ビットでの計算で 15 分程度かかったわけだが、13 量子ビットだと 30 分、14 量子ビットだと 1 時間くらいかかることになるのだろうか？

# まとめ

量子フーリエ変換を応用するアルゴリズムとして量子位相推定を見てみた。量子計算としての計算量には古典計算に対する優位性があると思うが、一方で計算を担う量子回路の構築にかなりのリソースを食うので、この部分は後々には Python のままだと厳しいかなという感触がある。そのうち [Mojo](https://www.modular.com/mojo) みたいな言語で書けるようになるとこの辺も改善するのだろうか？
