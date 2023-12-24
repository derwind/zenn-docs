---
title: "行列積状態について考える (6) — Vidal の標準形"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python", "Qiskit"]
published: false
---

# 目的

「Vidal の標準形」と呼ばれる MPS (Matrix Product State; 行列積状態) の一つの表示形式について見る。

# Tensor-Train 表示

正規化された量子の状態ベクトル

$$
\begin{align*}
\ket{\Psi} = \Psi_{000} \ket{000} + \Psi_{001} \ket{001} + \cdots + \Psi_{111} \ket{111}
\end{align*}
$$

を考える。一般に $d$ 量子ビットで考えるとして、係数の確率振幅 $\Psi_{i_1 i_2 \cdots i_d}$ の部分のことを波動関数と呼ぶらしい。[Wikipedia/波動関数](https://ja.wikipedia.org/wiki/%E6%B3%A2%E5%8B%95%E9%96%A2%E6%95%B0) によると、「基底 $\{\ket{\Psi_{i_1 i_2 \cdots i_d}}\}$ 表示での波動関数」と呼ぶのが良いのかもしれない。あまり物理的な解釈に詳しくないので気にしないことにする。

この波動関数を寄せ集めて行列を作り、[行列積状態について考える (3)](/derwind/articles/dwd-matrix-product03) で扱った [Tensor-Train Decomposition, SIAM J. Sci. Comput., 33(5), 2295–2317. (23 pages), I. V. Oseledets](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition) の TT-分解を適用することで、

$$
\begin{align*}
\Psi_{i_1 i_2 \cdots i_d} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} G_1(\alpha_0,i_1,\alpha_1) G_2(\alpha_1,i_2,\alpha_2) \cdots G_d(\alpha_{d-1},i_d,\alpha_d)
\end{align*}
$$

或は、(少し混乱を招く書き方だが) 記号を少し変えて

$$
\begin{align*}
\Psi_{i_1 i_2 \cdots i_d} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} G^{i_1}_{\alpha_1} G^{i_2}_{\alpha_1,\alpha_2} \cdots G^{i_d}_{\alpha_{d-1}}
\end{align*}
$$

と書けることが分かる。これを行列積状態 (MPS) と呼んだ。正確にはその状態の係数なのだが、係数の集合と量子状態が 1 対 1 に対応するので、混同して同じ用語で呼んでも良いらしい。

# Vidal の標準形

[テンソルネットワークの基礎と応用](https://www.saiensu.co.jp/search/?isbn=978-4-7819-1515-9&y=2021) p.83 によると、「正準な行列積波動関数には等価な表現が多数ある」と書かれている。その 1 つに [Efficient classical simulation of slightly entangled quantum computations](https://arxiv.org/abs/quant-ph/0301063) で提示された「Vidal の標準形」というものがあるらしい。これは、

$$
\begin{align*}
\Psi_{i_1 i_2 \cdots i_d} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} \Gamma^{[1] i_1}_{\alpha_1} \lambda^{[1]}_{\alpha_1} \Gamma^{[2] i_2}_{\alpha_1 \alpha_2} \lambda^{[2]}_{\alpha_2} \cdots \Gamma^{[d] i_d}_{\alpha_{d-1}} 
\end{align*}
$$

と書けることを主張するものである。ここで $\Gamma^{[\ell]}$ は両端は 2 階、内側では 3 階のテンソルで、$\lambda^{[\ell]}$ は特異値からなる 1 階のテンソルである。

これは結構視覚的にインパクトが強い表記のように思えるので、趣味の問題だが少し記号を変えて

$$
\begin{align*}
\Psi_{i_1 i_2 \cdots i_d} = \sum_{\alpha_0,\cdots,\alpha_{d-1},\alpha_d} \Gamma_1 (i_1, \alpha_1) \lambda_1 (\alpha_1) \Gamma_2 (\alpha_1, i_2 ,\alpha_2) \lambda_2 (\alpha_2) \cdots \Gamma_d (\alpha_{d-1}, i_d)
\end{align*}
$$

とすると、冒頭の表示のようになって少し気持ちが安らぐ可能性がある。要するに、TT-分解のような表示において、サイト間の特異値をくくり出して陽に見せる表示である。

手計算で幾らかの例を計算してみると一応は「こういうものか・・・」というのは分かるのだが、一般化した実装が容易とも思えない。とにかく自力で実装するのが結構大変そうだったので、今回 Qiskit Aer の実装を眺めたい。

# Qiskit Aer の `matrix_product_state` シミュレーション

実は Qiskit Aer には「Vidal の標準形」を利用した MPS を使ったシミュレーションがあって、以下のようにすれば良い。ここでは例として GHZ 状態を用いる。

```python
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


n_qubits = 3

qc = QuantumCircuit(n_qubits)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.save_matrix_product_state(label="my_mps")
qc.measure_all()

sim = AerSimulator(method="matrix_product_state")
result = sim.run(qc).result()
Gammas, lambdas = result.results[0].data.my_mps
Gamma1, Gamma2, Gamma3 = Gammas
lambda1, lambda2 = lambdas

print(f"{Gammas=}")
print(f"{lambdas=}")
```

```
Gammas=[(array([[1.-0.j, 0.-0.j]]), array([[0.-0.j, 1.-0.j]])), (array([[1.41421356-0.j, 0.        -0.j],
       [0.        -0.j, 0.        -0.j]]), array([[0.        +0.j, 0.        +0.j],
       [0.        +0.j, 1.41421356+0.j]])), (array([[1.-0.j],
       [0.-0.j]]), array([[0.-0.j],
       [1.-0.j]]))]
lambdas=[array([0.70710678, 0.70710678]), array([0.70710678, 0.70710678])]
```

今回、この実装を拝借したいと思う。なお実装は [matrix_product_state_internal.cpp#L1754-L1819](https://github.com/Qiskit/qiskit-aer/blob/0.13.1/src/simulators/matrix_product_state/matrix_product_state_internal.cpp#L1754-L1819) の部分である。

これは C++ で書かれているのだが、AI の力を借りて Python に翻訳しても恐らくあまり Python らしからぬ実装が出て来るであろうから、今回は筋トレを兼ねて自力で Python に翻訳した。

# Vidal の標準形の Python 実装

頑張ってソースコードを書き換えると以下のようになるらしい。もう少しコンパクトにできるかもしれないが、オリジナルの実装を極力残した。

```python
def TT_SVD_Vidal(
    C: np.ndarray, num_qubits: int | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gammas = []
    lambdas = []

    if len(C.shape) == 1:
        C = C.reshape(1, -1)
    if num_qubits is None:
        num_qubits = int(np.log2(np.prod(C.shape)))

    for i in range(num_qubits - 1):
        # step 1
        if i == 0:
            remaining_matrix = C
        else:
            remaining_matrix = S.reshape(-1, 1) * Vh  # np.diag(S) @ Vh
        _, cols = remaining_matrix.shape
        reshaped_matrix = np.concatenate([
            remaining_matrix[:, :cols // 2], remaining_matrix[:, cols // 2:]
        ], axis=0)
        # step 2
        U, S, Vh = np.linalg.svd(reshaped_matrix, full_matrices=False)
        r = len(S) - len(S[S == 0])
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        # step 3
        rows, _ = U.shape
        left_gamma = np.array([U[:rows // 2, :].tolist(), U[rows // 2:, :].tolist()])
        if i != 0:
            for a1 in range(left_gamma[0].shape[0]):
                left_gamma[:, a1, :] /= lambdas[-1][a1]  # [i, a1, a2]
        if i == 0:
            left_gamma = left_gamma.squeeze(axis=1)  # drop a1
        gammas.append(left_gamma)
        lambdas.append(S)
    # step 4
    _, cols = Vh.shape
    right_gamma = np.array([Vh[:, :cols // 2], Vh[:, cols // 2:]])
    right_gamma = right_gamma.squeeze(axis=2)  # drop a2
    gammas.append(right_gamma)

    return gammas, lambdas
```

# 実験

まず、GHZ 状態で試してみる。

```python
ket_ZERO = np.array([1, 0], dtype=float)
ket_ONE = np.array([0, 1], dtype=float)
state_ghz = (state_000 + state_111) / np.sqrt(2)

gammas_ghz, lambdas_ghz = TT_SVD_Vidal(state_ghz)
lambdas_ghz.append(None)
for i, (gamma, lam) in enumerate(zip(gammas_ghz, lambdas_ghz)):
    print(f"gamma_{i}={gamma}")
    if lam is not None:
        print(f"lambda_{i}={lam}")

tensor = np.einsum(
    "ia,a,jab,b,kb->ijk",
    gammas_ghz[0], lambdas_ghz[0], gammas_ghz[1], lambdas_ghz[1], gammas_ghz[2]
).flatten()
print(f"{tensor=}")
print(np.allclose(tensor, state_ghz))
```

```
gamma_0=[[1. 0.]
 [0. 1.]]
lambda_0=[0.70710678 0.70710678]
gamma_1=[[[ 1.41421356  0.        ]
  [ 0.          0.        ]]

 [[ 0.          0.        ]
  [ 0.         -1.41421356]]]
lambda_1=[0.70710678 0.70710678]
gamma_2=[[ 1. -0.]
 [ 0. -1.]]
tensor=array([0.70710678, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.70710678])
True
```

Qiskit Aer のシミュレーションの内容に対応する結果が出力され、縮約をとることで元の状態が復元できた。

次にランダムな状態ベクトルで試してみよう。

```python
from qiskit.quantum_info.random import random_statevector


sv = random_statevector(2**5, seed=1234)
print(sv.data)

gammas, lambdas = TT_SVD_Vidal(sv.data)

tensor = np.einsum(
    "ia,a,jab,b,kbc,c,lcd,d,md->ijklm",
    gammas[0], lambdas[0], gammas[1], lambdas[1], gammas[2],
    lambdas[2], gammas[3], lambdas[3], gammas[4]
).flatten()
print(np.allclose(tensor, sv.data))
```

```
[-0.1768313 -0.05308609j  0.00706735-0.27473268j  0.08168709-0.09664569j  0.01682705-0.05573499j  0.09523223-0.14147162j  0.32118424-0.14667558j -0.16304792+0.09106995j  0.10424328-0.02725673j -0.18370004-0.18740138j  0.03789962-0.1472075j  -0.05649957-0.03303674j  0.14595127+0.12291322j -0.09485034-0.16608936j  0.05727681+0.17531807j -0.13948863-0.05373012j -0.23805623-0.18865786j  0.04793166+0.05657085j  0.19110411+0.15844681j  0.05734748-0.02445505j -0.11049396+0.07153537j  0.02958648-0.03504915j  0.08458498-0.00121041j  0.1313439 +0.18362078j -0.12761052+0.09876508j  0.0767684 -0.132593j    0.03874187+0.30867336j -0.00357393-0.11259216j  0.00145334+0.09350818j -0.07489082+0.05491618j -0.06841686-0.00931013j  0.14677325+0.02232594j  0.02853828-0.01806043j]
True
```

ということで元の状態が復元できた。

# まとめ

Qiskit Aer の実装を拝借することで「Vidal の標準形」と呼ばれる、量子状態のテンソルネットワーク表現について見た。
自分で論文から実装をおこしていないので、こういうものなのか？という気持ちはある。行列を横方向に半分に割って、縦に繋げるといった部分はやや衝撃的な気もするのだが、ちゃんと読み解けていない。
とりあえず現時点では「こういうものらしい」という状態である。

# 参考文献
[O] [I. V. Oseledets, Tensor-Train Decomposition, SIAM J. Sci. Comput., 33(5), 2295–2317. (23 pages), 2011.](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition)
[S] [Ulrich Schollwoeck, The density-matrix renormalization group in the age of matrix product states, arXiv:1008.3477, 2010.](https://arxiv.org/abs/1008.3477)
[V] [Guifre Vidal, Efficient classical simulation of slightly entangled quantum computations, arXiv:quant-ph/0301063, 2003.](https://arxiv.org/abs/quant-ph/0301063)
[N] [西野友年, テンソルネットワークの基礎と応用, サイエンス社, 2021.](https://www.saiensu.co.jp/search/?isbn=978-4-7819-1515-9&y=2021)
