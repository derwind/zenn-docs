---
title: "量子フーリエ変換を眺める"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "NumPy", "Qiskit", "poem"]
published: true
---

# 目的

Qiskit textbook の [量子フーリエ変換](https://ja.learn.qiskit.org/course/ch-algorithms/quantum-fourier-transform) を見ると、Bloch 球上のアニメーションが掲載されている。そう言えば以前に勉強した時は「ふ〜ん」で流した気がするので、少し真面目に見てみようという企画。

# 量子フーリエ変換とは？

量子コンピュータの色々な FTQC アルゴリズム、例えば量子位相推定や Shor のアルゴリズムで利用されるモジュールである。形式的には、古典的な離散フーリエ変換と対をなす形で定義される。

## 離散フーリエ変換

$N \in \N$ に対し $x$ を $[0, 1, \cdots, N-1]$ 上の函数とし、$x_j = x(j),\ j \in [0, 1, \cdots, N-1]$ とおく。$\delta_j$ をデルタ測度とし、$\mu = \sum_{j=0}^{N-1} \delta_j$ とおく。この時、離散フーリエ変換は以下のように定義される:

$$
\begin{align*}
y_k = \hat{x}(k) = \mathcal{F}[x](k) =  \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} x_j \omega_N^{jk},\quad k \in [0, 1, \cdots, N-1]
\tag{1}
\end{align*}
$$
ここで、$\omega_N^{jk} = e^{2\pi i \frac{jk}{N}}$ である。

これは積分の形で書くと以下のようになる:

$$
\begin{align*}
\mathcal{F}[x](k) = \frac{1}{\sqrt{N}} \int x(j) \omega_N^{jk} d\mu(j)
\tag{1'}
\end{align*}
$$

離散フーリエ変換によって、$[0, 1, \cdots, N-1]$ 上の函数は $[0, 1, \cdots, N-1]$ 上の函数にうつる。或は、点列 $\{x_j\}$ を点列 $\{y_k\}$ にうつすとも言える。

## 量子フーリエ変換

量子フーリエ変換は、この離散フーリエ変換を正規直交基底、とりわけ計算基底に拡張したものとして定義される。この構成は超函数 (distribution) のフーリエ変換の定義と同様である。

$\mathscr{S}(\R)$ を $\R$ 上の急減少函数の空間とし、$\mathscr{S}^\prime(\R)$ をその上の超函数のなす空間とする。この時、$T \in \mathscr{S}^\prime(\R)$ のフーリエ変換は任意の $\varphi \in \mathscr{S}(\R)$ に対して以下で定義される:

$$
\begin{align*}
\braket{\mathcal{F}[T], \varphi} := \braket{T, \mathcal{F}[\varphi]}
\end{align*}
$$

この構成法と同様にして、量子フーリエ変換は以下で定義される。なお、$\ket{j}$ や $\ket{k}$ を計算基底とする。4 量子ビットのケースでは、$\ket{5}$ は 5 を 2 進数展開した `0101` を使って、$\ket{5} = \ket{0101} = \ket{0} \otimes \ket{1} \otimes \ket{0} \otimes \ket{1}$ と解釈される。

**定義** $\ket{j}$ の量子フーリエ変換は $[0, 1, \cdots, N-1]$ 上の任意の函数 $x$ に対して以下で定義される:

$$
\begin{align*}
\big\langle \mathcal{F}[\ket{j}], x \big\rangle := \big\langle \ket{j}, \mathcal{F}[x] \big\rangle
\end{align*}
$$

ここで、$\mathcal{F}$ は離散フーリエ変換である。

**具体的な計算**

この右辺を積分の順序交換を用いて計算すると、

$$
\begin{align*}
\big\langle \ket{j}, \mathcal{F}[x] \big\rangle &= \frac{1}{\sqrt{N}} \int \ket{j} d\mu(j) \int x(k) \omega_N^{jk} d\mu(k) \\
&= \frac{1}{\sqrt{N}} \int x(k) d\mu(k) \int \omega_N^{jk} \ket{j} d\mu(j) \\
&= \int x(k) \left[ \frac{1}{\sqrt{N}} \int \omega_N^{jk} \ket{j} d\mu(j) \right] d\mu(k)
\end{align*}
$$

となる。この事から量子フーリエ変換は

$$
\begin{align*}
\ket{k} = \ket{\tilde{j}} = \mathcal{F}[\ket{j}] &= \frac{1}{\sqrt{N}} \int \omega_N^{jk} \ket{j} d\mu(j)\\
&= \frac{1}{\sqrt{N}} \sum_{j=0}^{N-1} \omega_N^{jk} \ket{j}
\tag{2}
\end{align*}
$$

と書ける。この内容は [量子フーリエ変換](https://ja.learn.qiskit.org/course/ch-algorithms/quantum-fourier-transform) と符合している。

# フーリエ基底での計算

[量子フーリエ変換](https://ja.learn.qiskit.org/course/ch-algorithms/quantum-fourier-transform) の「2.1 フーリエ基底での計算」をもうちょっと具体的に計算して眺めたいというのがここからのテーマである。今回は量子回路を使わずに `NumPy` の計算だけで済ませる。

## 準備

幾つか準備が必要である。今回複数量子ビットのケースを見たいが、量子フーリエ変換後は複合システムの状態ベクトルが現れる。一方、Qiskit textbook のアニメーションは個別の量子ビットに対する部分システムを描画している。このためには複合システムを密度行列で表した上で部分トレースをとって部分システムの状態を割り出す必要がある。

詳細は [量子コンピュータと量子通信I](https://shop.ohmsha.co.jp/shopdetail/000000006441/) などに譲るとして、システム A とシステム B からなる複合システムの密度行列が $\rho^{AB}$ であり、これが $\rho^{AB} = \sum_i \rho_i \otimes \sigma_i$ のように書けているとする。この時、1 つのコンポーネントに注目するとして $\rho \otimes \sigma$ からシステム A の状態 $\rho^A$ の割り出しは以下の計算によって達成される:

$$
\begin{align*}
\rho^A = \operatorname{tr}_B (\rho \otimes \sigma) = \rho \operatorname{tr} (\sigma)
\tag{3}
\end{align*}
$$

要するに、全体で見ると $\sum_i \rho_i \operatorname{tr} (\sigma_i)$ を計算すれば良い。

以下、`NumPy` で実装をするが、ストーリーは以下のようになっている:

1. 計算基底に量子フーリエ変換を適用して状態ベクトルを得る。
2. 状態ベクトルから密度行列を作る。
3. 密度行列に部分トレースを適用して、部分システムとしての各量子ビットの状態を決定する。

## NumPy 実装

密度行列を作るまでの関数らの定義を一気に掲載する:

```python
import numpy as np

I = np.eye(2)
ZERO = np.array([1., 0.])
ONE = np.array([0., 1.])

# bin_num は |5> であれば '0101' を期待している。
def to_statevec(bin_num: str, n_qubits: int = 4) -> int | np.ndarray:
    vec = 1
    for c in bin_num:
        v = ZERO if c == '0' else ONE
        vec = np.kron(vec, v)
    return vec

# 状態ベクトルから密度行列を計算する。
def density_matrix(sv: np.ndarray) -> np.ndarray:
    return sv.reshape(-1, 1) @ sv.reshape(1, sv.shape[0]).conj()

# https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/states/densitymatrix.py
def dm2sv(dm: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eig(dm)

    psi = evecs[:, np.argmax(evals)]
    if np.isclose(psi.real[0], 0).all():
        if psi[0].imag >= 0:
            psi *= -1.j
        else:
            psi *= 1.j
    return psi
```

最後に密度行列から状態ベクトルをとりだす関数を定義している。一般にはこれは純粋状態に由来する密度行列でないと無理なのだが、今回扱う範囲ではそのようなケースのみなので、思い切り端折って実装している。この実装は Qisikit の [https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/states/densitymatrix.py](https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/quantum_info/states/densitymatrix.py) を拝借した。


また、量子フーリエ変換は定義をそのまま実装して以下のようになる:

```python
def qft(x: int, n_qubits: int = 4):
    N = 2**n_qubits
    result = 0
    for i in range(N):
        bin_num = bin(i)[2:].zfill(n_qubits)
        result += np.exp(2*np.pi*1.j*(i*x)/N)*to_statevec(bin_num, n_qubits)
    return result / np.sqrt(N)
```

部分トレースをとる処理は汚くなったが以下のようにした:

```python
def partial_trace(vec: np.ndarray, target_qubit: int, n_qubits: int = 4):
    def partial_basis_generator(target_qubit: int, n_qubits: int = 4):
        if n_qubits <= 1:
            return

        N = 2**(n_qubits-1)
        for i in range(N):
            bin_num = bin(i)[2:].zfill(n_qubits-1)
            bin_num_former = bin_num[:n_qubits-target_qubit-1]
            bin_num_latter = bin_num[n_qubits-target_qubit-1:]
            sv1 = to_statevec(bin_num_former)
            if isinstance(sv1, np.ndarray):
                sv1 = sv1.reshape(-1, 1)
            sv2 = to_statevec(bin_num_latter)
            if isinstance(sv2, np.ndarray):
                sv2 = sv2.reshape(-1, 1)
            yield np.kron(np.kron(sv1, I), sv2)

    dm = density_matrix(sv)

    result = 0
    for basis in partial_basis_generator(target_qubit, n_qubits):
        result += basis.T.conj() @ dm @ basis
    return result
```

`target_qubit` 番目の量子ビットに対応する部分システムの情報を取り出すものとして実装している。`target_qubit` 番目の量子ビットを除いた計算基底を順次返すジェネレータ `partial_basis_generator` を用意して、(3) 式を参考に以下の数式を実行している。

$$
\begin{align*}
\rho^A &= \sum_i (I_A \otimes \bra{e_i^B}) (\rho \otimes \sigma) (I_A \otimes \ket{e_i^B}) \\
&= \sum_i \rho \otimes \braket{e_i^B |\sigma | e_i^B} \\
&= \rho \sum_i \braket{e_i^B |\sigma | e_i^B} = \rho \operatorname{tr} (\sigma)
\tag{4}
\end{align*}
$$

ここで、`target_qubit` 番目の量子ビットに対応する部分システムを A、それ以外の量子ビットに対応する部分システムを B とし、$\{ \ket{e_i^B} \}$ は `target_qubit` 番目の量子ビットを除いた計算基底全体をわたるとする[^1]。

[^1]: やや苦しい数式であり、厳密ではない。例えば 4 量子ビットのケースで、2 番目の量子ビットに対する状態を取り出したい場合にはこの数式では不完全である。`partial_basis_generator` では 0 番目と 1 番目の量子ビットの計算基底からなるベクトルと、3 番目の量子ビットの計算基底のベクトルの間に単位行列を挟む形でクロネッカー積をとっている。

これで Qiskit textbook のアニメーションを検証する準備ができた。

## 検証

まずは 1 量子ビットの時から見よう。この時、量子フーリエ変換はただのアダマールゲート H であるので、$\ket{0}$ に対しては $\ket{+}$ が、$\ket{1}$ に対しては $\ket{-}$ が得られれば良い。

```python
print(qft(0, 1))
print(qft(1, 1))
```

> [0.70711+0.j 0.70711+0.j]
> [ 0.70711+0.j -0.70711+0.j]

期待する結果、$\operatorname{QFT}(\ket{0}, 1) = \ket{+}$ と $\operatorname{QFT}(\ket{1}, 1) = \ket{-}$ が得られた。なお、表示は見やすさのために、`np.round(..., 5)` したものを掲載している。

次に 4 量子ビットのケースを検証しよう:

```python
n_qubits = 4

for x in range(6+1):
    print(f'{x=}')
    sv = qft(x, n_qubits)
    for i in range(n_qubits):
        v = np.round(dm2sv(partial_trace(sv, i, 4)), 5)
        print(i, v)
    print()
```

> x=0
> 0 [0.70711+0.j 0.70711+0.j]
> 1 [0.70711+0.j 0.70711+0.j]
> 2 [0.70711+0.j 0.70711+0.j]
> 3 [0.70711+0.j 0.70711+0.j]
>
> x=1
> 0 [0.70711+0.j     0.65328+0.2706j]
> 1 [0.5    -0.5j 0.70711+0.j ]
> 2 [0.70711+0.j      0.     +0.70711j]
> 3 [ 0.70711+0.j -0.70711+0.j]
>
> x=2
> 0 [0.70711+0.j  0.5    +0.5j]
> 1 [0.70711+0.j      0.     +0.70711j]
> 2 [ 0.70711+0.j -0.70711+0.j]
> 3 [0.70711+0.j 0.70711-0.j]
>
> x=3
> 0 [0.70711+0.j      0.2706 +0.65328j]
> 1 [ 0.70711+0.j  -0.5    +0.5j]
> 2 [0.70711+0.j      0.     -0.70711j]
> 3 [ 0.70711+0.j -0.70711+0.j]
>
> x=4
> 0 [0.70711+0.j      0.     +0.70711j]
> 1 [ 0.70711+0.j -0.70711+0.j]
> 2 [0.70711+0.j 0.70711-0.j]
> 3 [0.70711+0.j 0.70711-0.j]
>
> x=5
> 0 [-0.2706 -0.65328j  0.70711+0.j     ]
> 1 [ 0.70711+0.j  -0.5    -0.5j]
> 2 [0.70711+0.j      0.     +0.70711j]
> 3 [ 0.70711+0.j -0.70711+0.j]
>
> x=6
> 0 [ 0.70711+0.j  -0.5    +0.5j]
> 1 [0.70711+0.j      0.     -0.70711j]
> 2 [ 0.70711+0.j -0.70711-0.j]
> 3 [0.70711+0.j 0.70711-0.j]

数字なので分かりにくいが、$\ket{\tilde{0}}, \ket{\tilde{1}}, \ket{\tilde{2}}, \cdots$ につれて、3 番目の量子ビットでは $\ket{+}$ と $\ket{-}$ を交互に行ったり来たりしているのが見える。2 版目の量子ビットも $\ket{+}, \ket{i}, \ket{-}, \cdots$ と移動している。他の量子ビットも同様である。

これで、0 番目の量子ビットは $x$ に対してゆっくりと位相が変化し、量子ビットの序数が大きくなるにつれ位相の変化が激しくなることがなんとなく分かった。

# まとめ

Qiskit textbook のアニメーションを `NumPy` で実装しようとすると意外と面倒くさいことが分かったが、実装して確認すると符合する結果が得られた。

フーリエ何某の関連では “遠方は激しく振動する (ことで積分がキャンセルする; 離散的な標語としては級数が収束する)” といった感覚がある。今回はそう単純に当てはめられはしないのだが、何となく序数の大きい量子ビットでの状況は位相の変化が激しいという「これ系のコンテキストでよくある感覚のもの」が出てきているので馴染みやすい。

なお、こんな面倒くさいことをしなくても、Qiskit textbook を読み進めれば分かるように「4. 量子フーリエ変換」の数式を経て、具体的に純粋状態のテンソル積で書き下せることが分かる。
