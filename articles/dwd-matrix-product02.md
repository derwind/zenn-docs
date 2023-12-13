---
title: "行列積状態について考える (2)"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["math", "Python"]
published: true
---

# 目的

行列積状態 (Matrix Product State; MPS) について以前 [行列積について考える](/derwind/articles/dwd-matrix-product) を書いたが、結局あまり理解できている気がしないし、そもそも具体的に計算できている気がしないので、論文を読んで考える。

# 論文

[The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477) が有名らしいので、

> 4. Matrix product states (MPS)

を眺めて、今回は量子状態 $\ket{\psi} = \ket{000}$ を MPS に変換してみたい。

2 量子ビットのシステムを使った概要がざっと続いて

> 4.1.3. Decomposition of arbitrary quantum states into MPS

からもっと一般の話になる。

# 論文を見ていく

$\ket{\psi} = \ket{000}$ をターゲットとするで $d=2$, $L=3$ とする。まず式 (30) の形に落とし込もう。すると

$$
\begin{align*}
\ket{\psi} =& 1\cdot\ket{000} + 0\cdot\ket{001} + 0\cdot\ket{010} + 0\cdot\ket{011} \\
&+ 0\cdot\ket{100} + 0\cdot\ket{101} + 0\cdot\ket{110} + 0\cdot\ket{111}
\tag{30}
\end{align*}
$$

ということになるだろう。$\sigma_1, \sigma_2, \sigma_3 \in \{0, 1\}$ として、$\sigma_1 = \sigma_2 = \sigma_3 = 0$ の時だけ $c_{\sigma_1 \sigma_2 \sigma_3}$ は 1 で、それ以外では 0 である。

さて、式 (31) に変形するにあたって、$\sigma_1 \sigma_2 \sigma_3$ というには 2 進数だと考えることにする。例えば、$\sigma_1 = 1, \sigma_2 = 0, \sigma_3 = 0$ は `0b100`、従って 10 進数の `4` と見る。これは、$\sigma_1 \sigma_2 \sigma_3 \simeq 2^2 \sigma_1 + 2^1 \sigma_2 + 2^0 \sigma_3 \in \{0, 1, 2, 3, 4, 5, 6, 7\}$ という同一視をしていることになる。すると、

$$
\begin{align*}
(c_{\sigma_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3}
&\stackrel{\text{vec} \leftrightarrow \text{mat}}{\simeq}
(\Psi_{\sigma_1, (\sigma_2 \sigma_3)})_{\sigma_1,(\sigma_2 \sigma_3)} \\
&\quad = \begin{pmatrix}
c_{0\mathit{00}} & c_{0\mathit{01}} & c_{0\mathit{10}} & c_{0\mathit{11}} \\
c_{1\mathit{00}} & c_{1\mathit{01}} & c_{1\mathit{10}} & c_{1\mathit{11}}
\end{pmatrix} \in \operatorname{Mat}(2, 2^{L - 1})
\tag{31}
\end{align*}
$$

となるそうだ。「$\sigma_2 \sigma_3$」という一纏めのインデックスの値については斜体で記述した。これはコードで書くと、以下のような内容に対応する:

```python
c = np.arange(8)
m = c.reshape(2, 4)
```

$(\Psi_{\sigma_1, (\sigma_2 \sigma_3)})_{\sigma_1,(\sigma_2 \sigma_3)}$ は行列であるので、SVD することができて、

$$
\begin{align*}
(\Psi_{\sigma_1, (\sigma_2 \sigma_3)})_{\sigma_1,(\sigma_2 \sigma_3)} &= U S V^\dagger \\
&= U \begin{pmatrix} S_{00} & \\ & S_{11} \end{pmatrix} V^\dagger \\
& = (\sum_{a_1 = 0}^{r_1 - 1} U_{\sigma_1, a_1} S_{a_1, a_1} (V^\dagger)_{a_1, \sigma_2 \sigma_3})_{\sigma_1,(\sigma_2 \sigma_3)},
\tag{31.5}
\end{align*}
$$

となる。単に行列の各成分について SVD の結果の計算を陽に書いただけである。ここで $r_1$ は特異値からなる対角行列 $S$ のランクである。

$S V^\dagger$ の部分を掛けて、これをベクトルへと `flatten` したい。

$$
\begin{align*}
S V^\dagger = (W_{a_1, \sigma_2 \sigma_3})_{\sigma_1,(\sigma_2 \sigma_3)} \stackrel{\text{mat} \leftrightarrow \text{vec}}{\simeq} (c_{a_1 \sigma_1 \sigma_2})_{\sigma_1 \sigma_2 \sigma_3},
\end{align*}
$$

これはコードで書くと、以下のような感じの内容に対応する:

```python
m = np.zeros((2, 4))
c = m.flatten()
```

これを (31.5) 式に代入して次を得る:

$$
\begin{align*}
\begin{split}
(\Psi_{\sigma_1, (\sigma_2 \sigma_3)})_{\sigma_1,(\sigma_2 \sigma_3)} &= U S V^\dagger \\
&= (\sum_{a_1 = 0}^{r_1 - 1} U_{\sigma_1, a_1} S_{a_1, a_1} (V^\dagger)_{a_1, \sigma_2 \sigma_3})_{\sigma_1,(\sigma_2 \sigma_3)} \\
&\simeq (\sum_{a_1 = 0}^{r_1 - 1} U_{\sigma_1, a_1} c_{a_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3}
\end{split}
\tag{32}
\end{align*}
$$

更に、行列 $(U_{\sigma_1, a_1})_{\sigma_1, a_1}$ もテコ入れする。以下のように行列を行ごとに分解して、行ベクトルの集まりと同一視する:

$$
\begin{align*}
(U_{\sigma_1, a_1})_{\sigma_1, a_1} = \begin{pmatrix}
U_{00} & U_{01} \\
U_{10} & U_{11}
\end{pmatrix}
\simeq \{(U_{00} \ \ U_{01}), (U_{10} \ \ U_{11})\} =: \{ A^0, A^1 \} = \{ (A^{\sigma_1}_{a_1})_{a_1} \}_{\sigma_1 \in \{0, 1\}}
\end{align*}
$$

すると、これを式 (32) に代入して、式 (31) も思い出して

$$
\begin{align*}
\begin{split}
(c_{\sigma_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} &\simeq (\Psi_{\sigma_1, (\sigma_2 \sigma_3)})_{\sigma_1,(\sigma_2 \sigma_3)} \\
&\simeq (\sum_{a_1 = 0}^{r_1 - 1} U_{\sigma_1, a_1} c_{a_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} \\
&\simeq (\sum_{a_1 = 0}^{r_1 - 1} A^{\sigma_1}_{a_1} c_{a_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3}
\end{split}
\tag{32.5}
\end{align*}
$$

を得る。要するに NumPy 的には `reshape` やスライシングを駆使して式を書き直しているだけである。

次にベクトル $(c_{a_1 \sigma_2 \sigma_3})_{a_1 \sigma_2 \sigma_3}$ もテコ入れする。これも `reshape` して次のような行列との同一視を行う:

$$
\begin{align*}
(c_{a_1 \sigma_1 \sigma_2})_{a_1 \sigma_2 \sigma_3} &= (c_{000}, c_{001}, c_{010}, c_{011}, c_{100}, c_{101}, c_{110}, c_{111}) \\
&\simeq \begin{pmatrix}
c_{00\mathit{0}} & c_{00\mathit{1}} \\
c_{01\mathit{0}} & c_{01\mathit{1}} \\
c_{10\mathit{0}} & c_{10\mathit{1}} \\
c_{11\mathit{0}} & c_{11\mathit{1}} \\
\end{pmatrix}
=: (\Psi_{(a_1 \sigma_2), \sigma_3})_{(a_1 \sigma_2), \sigma_3}
\end{align*}
$$

これを式 (32.5) に代入して

$$
\begin{align*}
\begin{split}
(c_{\sigma_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} &\simeq (\sum_{a_1 = 0}^{r_1 - 1} A^{\sigma_1}_{a_1} c_{a_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} \\
&\simeq (\sum_{a_1 = 0}^{r_1 - 1} A^{\sigma_1}_{a_1} \Psi_{(a_1 \sigma_2), \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} 
\end{split}
\tag{33}
\end{align*}
$$

を得る。この行列 $\Psi$ もまた SVD にかけられて次のようになる:

$$
\begin{align*}
(\Psi_{(a_1 \sigma_2), \sigma_3})_{(a_1 \sigma_2), \sigma_3} &= USV^\dagger \\
&= (\sum_{a_2 = 0}^{r_2 - 1} U_{(a_1 \sigma_2), a_2} S_{a_2, a_2} (V^\dagger)_{a_2, \sigma_3})_{(a_1 \sigma_2), \sigma_3},
\end{align*}
$$

ここで、$r_2$ は $S$ のランクである。

これを式 (33) に代入して

$$
\begin{align*}
\begin{split}
(c_{\sigma_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3}
&\simeq
(\Psi_{\sigma_1, (\sigma_2 \sigma_3)})_{\sigma_1, (\sigma_2 \sigma_3)} \\
&\simeq (\sum_{a_1 = 0}^{r_1 - 1} A^{\sigma_1}_{a_1} \sum_{a_2 = 0}^{r_2 - 1} U_{(a_1 \sigma_2), a_2} S_{a_2, a_2} (V^\dagger)_{a_2, \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} \\
&= (\sum_{a_1 = 0}^{r_1 - 1} \sum_{a_2 = 0}^{r_2 - 1} A^{\sigma_1}_{a_1} U_{(a_1 \sigma_2), a_2} S_{a_2, a_2} (V^\dagger)_{a_2, \sigma_3})_{\sigma_1 \sigma_2 \sigma_3}
\end{split}
\tag{33.5}
\end{align*}
$$

というベクトルの等式を得る。

次にまた $U$ と $SV^\dagger$ を別々にテコ入れする。

まず $U$ のほうであるが、これは少しトリッキーで添え字に使われている記号 $a_1$, $\sigma_2$, $a_2$ のうち $\sigma_2$ が最初の添え字になるような並び替えを行う。要は NumPy の言葉で言うと `transpose` で軸の並び替えを行う。

$$
\begin{align*}
(U_{(a_1 \sigma_2), a_2})_{(a_1 \sigma_2), a_2} &= \begin{pmatrix}
U_{00 \mathit{0}} & U_{00 \mathit{1}} \\
U_{01 \mathit{0}} & U_{01 \mathit{1}} \\
U_{10 \mathit{0}} & U_{10 \mathit{1}} \\
U_{11 \mathit{0}} & U_{11 \mathit{1}} \\
\end{pmatrix} \\
&\simeq \left\{
\begin{pmatrix}
U_{00 \mathit{0}} & U_{00 \mathit{1}} \\
U_{10 \mathit{0}} & U_{10 \mathit{1}}
\end{pmatrix}
,
\begin{pmatrix}
U_{01 \mathit{0}} & U_{01 \mathit{1}} \\
U_{11 \mathit{0}} & U_{11 \mathit{1}}
\end{pmatrix}
\right\} \\
&=: \{ A^0, A^1 \} = \{ (A^{\sigma_2}_{a_1, a_2})_{a_1, a_2} \}_{\sigma_2 \in \{0,1\}}
\end{align*}
$$

これはコードで書くと、以下のような感じの内容に対応する:

```python
t = np.zeros((2, 2, 2))
t2 = t.transpose(1, 0, 2)
```

次に $SV^\dagger$ であるが、以下のように変形して列ベクトルの集まりと同一視する:

$$
\begin{align*}
S V^\dagger = (W_{a_2, \sigma_3})_{a_2, \sigma_3} &= \begin{pmatrix}
w_{00} & w_{01} \\
w_{10} & w_{11}
\end{pmatrix} \\
&\stackrel{\text{mat} \leftrightarrow \text{vecs}}{\simeq}
\left\{
\begin{pmatrix}
w_{00} \\
w_{10}
\end{pmatrix}
,
\begin{pmatrix}
w_{01} \\
w_{11}
\end{pmatrix}
\right\} =:
\{ A^0, A^1 \} = \{ (A^{\sigma_3}_{a_2}) \}_{\sigma_3}
\end{align*} 
$$

これらの結果を式 (33.5) に代入して、以下を得る:

$$
\begin{align*}
\begin{split}
(c_{\sigma_1 \sigma_2 \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} &\simeq (\sum_{a_1 = 0}^{r_1 - 1} A^{\sigma_1}_{a_1} \Psi_{(a_1 \sigma_2), \sigma_3})_{\sigma_1 \sigma_2 \sigma_3} \\
&\simeq (\sum_{a_1 = 0}^{r_1 - 1} \sum_{a_2 = 0}^{r_2 - 1} A^{\sigma_1}_{a_1} A^{\sigma_2}_{a_1, a_2} A^{\sigma_3}_{a_2})_{\sigma_1 \sigma_2 \sigma_3}
\end{split}
\tag{34}
\end{align*}
$$

今回はもうここで終わりなので、論文の式 (34) と式 (35) は同じものになってしまう。よってベクトルの成分として

$$
\begin{align*}
c_{\sigma_1 \sigma_2 \sigma_3} = A^{\sigma_1} A^{\sigma_2} A^{\sigma_3}
\tag{36}
\end{align*}
$$

という感じのものが得られて、これを MPS と呼ぶという話である。

# Python で実装する

## 乱数ベクトルで試す

上記をそのまま実装すると以下のようになる。$\ket{000}$ でやると計算ミスに気づきにくいので、もっと一般に乱数で生成した長さ 8 のベクトルで試してみる。

式 (30)～(33) までは以下のようになる:

```python
d = 2
L = 3  # sites

def idx2bin_list(idx: int, n: int):
    return [int(c) for c in bin(idx)[2:].zfill(n)]  # 6 -> 0b110 -> [1, 1, 0]

def bin_list2index(li: list[int]):
    return int("".join([str(n) for n in li]), 2)

c = np.random.randn(d**L)  # vector; (30)

Psi1 = c.reshape(d, d**(L-1))  # matrix; (31)
U, s, Vh = sp.linalg.svd(Psi1, full_matrices=False)  # (32)
r1 = len(s[s != 0]); U = U[:, :r1]; s = s[:r1]; Vh = Vh[:r1, :]  # (32)
A1 = np.split(U.flatten(), d)  # a collection of d row vectors

c_ = (np.diag(s) @ Vh).flatten()  # reshape back into a vector; for (33)
Psi12 = c_.reshape(r1*d, d**(L-2))  # reshape into a matrix; for (33)

# validate (33)
test_c = np.zeros_like(c)  # for test
for i in range(len(test_c)):
    arr = idx2bin_list(i, L)
    sigma1 = arr[0]
    sigma2 = arr[1]
    sigma_others = bin_list2index(arr[2:])
    total = 0
    for a1 in range(r1):
        total += A1[sigma1][a1] * Psi12[bin_list2index([a1, sigma2]), sigma_others]
    test_c[i] = total

print("c:\n", c)
print("test_c:\n", test_c)
print(np.allclose(c, test_c))
```

> c:
>  [-0.7778906  -0.27829926  0.59346232  1.60519952 -0.72450807  0.50429389
>   0.99442366  1.12241033]
> test_c:
>  [-0.7778906  -0.27829926  0.59346232  1.60519952 -0.72450807  0.50429389
>   0.99442366  1.12241033]
> True

なんとなく良さそうである。

式 (34)～(35) は以下のようになる:

```python
U_, s_, Vh_ = sp.linalg.svd(Psi12, full_matrices=False)
r2 = len(s_[s_ != 0]); U_ = U_[:, :r2]; s_ = s_[:r2]; Vh_ = Vh_[:r2, :]
A12 = U_.reshape(r1, d, r2).transpose(1, 0, 2)  # a collection of d matrices
c_ = (np.diag(s_) @ Vh_)
A2 = c_.T  # two "column" vectors

# validate (35)
test_c = np.zeros_like(c)  # for test
for i in range(len(test_c)):
    arr = idx2bin_list(i, L)
    sigma1 = arr[0]
    sigma2 = arr[1]
    sigma3 = arr[2]
    total = 0
    for a1 in range(r1):
        for a2 in range(r2):
            total += A1[sigma1][a1] * A12[sigma2][a1, a2] * A2[sigma3][a2]
    test_c[i] = total

print("c:\n", c)
print("test_c:\n", test_c)
print(np.allclose(c, test_c))
```

> c:
>  [-0.7778906  -0.27829926  0.59346232  1.60519952 -0.72450807  0.50429389
>   0.99442366  1.12241033]
> test_c:
>  [-0.7778906  -0.27829926  0.59346232  1.60519952 -0.72450807  0.50429389
>   0.99442366  1.12241033]
> True

なんとなく良さそうである。

この時の行列積を表示してみると

```python
print(A1)
print("-"*10)
print(A12)
print("-"*10)
print([vec.T for vec in A2])
```

```
[array([-0.7426605 , -0.66966811]), array([-0.66966811,  0.7426605 ])]
----------
[[[ 0.20203733  0.87086524]
  [ 0.19788212 -0.29007397]]

 [[-0.95915367  0.12087011]
  [-0.0069159   0.37794345]]]
----------
[array([1.27046469, 0.92575436]), array([ 1.95055164, -0.60297723])]
```

ということらしい。そうか・・・としか言いようがない。

## $\ket{000}$ で試す

最後にやりたかったことを試して締めくくろう。冒頭のベクトルを

```python
c = np.zeros(d**L)
c[0] = 1
```

で作り直してコードを再実行するだけである。

```python
print(A1)
print("-"*10)
print(A12)
print("-"*10)
print([vec.T for vec in A2])
```

```
[array([1.]), array([0.])]
----------
[[[1.]]

 [[0.]]]
----------
[array([1.]), array([0.])]
```

これが論文によるところの $\ket{000}$ の MPS らしい。

## $\frac{1}{\sqrt{2}}(\ket{000} + \ket{111})$ で試す

結果だけ書くと、以下のようになるらしい。今回は結合次元が 2 になった。

```
[array([1., 0.]), array([0., 1.])]
----------
[[[ 1.  0.]
  [ 0.  0.]]

 [[ 0.  0.]
  [ 0. -1.]]]
----------
[array([0.70710678, 0.        ]), array([ 0.        , -0.70710678])]
```

論文の (23) 式の下に

> It is obvious that $r = 1$ corresponds to (classical) product states and $r > 1$ to entangled (quantum) states.

とあって、上記のようにもつれ状態の場合、結合次元が 1 より大きくなるということらしい。

# まとめ

一応、元の確率振幅を復元できるような「テンソルの集まり？」が得られた。

MPS として得られる各「テンソルの集まり？」の中から $\sigma_1$ 番目、$\sigma_2$ 番目、$\sigma_3$ 番目に対応する要素を取り出して、これらを縮約計算すると $c_{\sigma_1 \sigma_2 \sigma_3}$ の値になる、という話のようであることを確認した。

絵でも描き残そうかと思ったが、文章だけでかなりのボリュームになったので、たぶんもう描きそうにはない。
