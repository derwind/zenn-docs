---
title: "暗号について考えてみる (1)"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "poem"]
published: true
---

# 目的

RSA 暗号と Shor のアルゴリズムについて過去に書いた自分用メモが転がっていたので整理したい。整理したいが、正直あまり覚えていないので、またメモがなくならないように書き出しておこうという程度で、実験の詳細もあまり覚えていない。故に、**これは物凄い適当な記事**である。

# RSA 暗号

以下で言いたいことは、「**とても大きな “ほぼ素数” として開示されている公開鍵が素因数分解されると、秘密鍵が推測できてしまう**」ということである。

文献 [S] を参考にしよう[^1]。詳細を全て割愛して、ひたすら記号だけ導入する。

[^1]: 文献 [NC2] の付録 E にも似たような内容がある。

- $p$, $q$ を 201 桁以上の素数とし、非公開とする。
- $N = pq$ とおく。これは素因数分解の難しい 400 桁以上の数になる。
- $L = (p-1) (q-1)$ とおく。$p$ と $q$ が特定できないと算出できない。
- $\mathcal{M}$: 400 桁以下の自然数で $N$ と互いに素なもの。平文の集合。
- $\mathcal{C}$: 400 桁以下の自然数で $N$ と互いに素なもの。暗号文の集合。

次に公開鍵 $\{e, N\}$ と秘密鍵 $d$ を以下のように作る:

- $e$: $L$ と互いに素な自然数。$e > 1$
- $d$: $ed \equiv 1 \mod L$
    - この $d$ は $$ae + b L = 1 \tag{1}$$ を満たす整数 $a$ の中から適当に選ぶ。

この設定下で公開鍵 $\{e, N\}$ で暗号化したテキストを送ってもらって、秘密鍵 $d$ で復号できるというシナリオである。

暗号化と復号を以下で定める:

**暗号化** $E: \mathcal{M} \to \mathcal{C}$

$$
\begin{align*}
E(m) = m^e \mod N
\end{align*}
$$

**復号** $D: \mathcal{C} \to \mathcal{M}$

$$
\begin{align*}
D(c) = c^d \mod N
\end{align*}
$$

まずいのは「秘密鍵 $d$ を推測して作られてしまうこと」である。$e$ は公開鍵で誰でも分かるので、Eq. (1) より、$L$ が分かると $d$ のアタリがついてしまう。$L = (p-1) (q-1)$ で $N = pq$ であったので、つまり「**もう片方の公開鍵 $N$ が素因数分解されると秘密鍵 $d$ のアタリがついてしまう**」と言うことである。

## 例

今回は動きだけ見たいので、小さな数で試そう。$p=5$, $q=7$ としてみる。この時、$N=35$ であり、$L=24$ である。もう片方の公開鍵と秘密鍵のペアは、例えば $(e,d) = (5,5)$ で作れる。

Alice は Bob に重大なメッセージ「`i_love_you_bob`」を送りたい。

```python
N = 35
L = 24
e = 5
d = 5

message = "i_love_you_bob"
enc_text = [(ord(v)-90)**e % N for v in list(message)]
print(enc_text)
```
> [15, 10, 23, 21, 28, 16, 10, 26, 21, 27, 10, 8, 21, 8]

暗号化[^2]により謎の数列ができた。これを復号しよう。

[^2]: 今回小さい鍵で試行しているため、暗号化して復号できる文字の集合が小さい。よって、範囲に入るように適当にオフセット $90$ をかましている。

```python
dec_text = "".join([chr((v**d % N) + 90) for v in enc_text])
print(dec_text)
```
> i_love_you_bob

元のメッセージが復元された。これは**大事な内容**なので、途中で攻撃されたくないわけである。

# $n$ 桁の整数の素因数分解

例えば文献 [NC2] p.103 によると、文献 [L] に解説がある Lenstra による提案された「数体篩法 (すうたいふるいほう)」というものが知られている中では大変高速な素因数分解のアルゴリズムとのことである。文献 [Q] によると、

$$
\begin{align*}
O \left( \exp \left[ \frac{64}{9} n (\log n)^2 \right]^{1/3} \right)
\end{align*}
$$

という準指数関数的な計算量になるそうである。公開鍵$N$ は 400 桁以上が想定されているので、数体篩法で素因数分解しようとするととてもではないが計算が終わらない・・・ということになる (はず)。対して、量子計算を活用した Shor のアルゴリズムは $n$ に対して多項式時間で素因数分解問題を解けるとのことである。

**現実的な時間で素因数分解できないという前提のもとで設計された暗号方式のアルゴリズムであったので、現実的な時間で解かれると根底が崩れる**わけである。

# Shor のアルゴリズム

折角なので、文献 [S] に従いながら 35 を素因数分解したい[^3]。Alice のメッセージは攻撃を受けてしまうのであろうか？

[^3]: 相当なインチキをするが、よくある 15 の素因数分解よりは何かやっている気持ちになれそうな気がする。

細かいことは文献 [S] に書いてあるので、ここではざっと流す。以下のような量子回路を実装する。

![](/images/dwd-cryptography01/001.png)

今回は計算の都合で、突如 $a=9$ という $N=35$ と素であるような数字が**偶然にも**選ばれて、$a^r \equiv 1 \mod N$ となる $a$ の “周期” を求めることになる。これがその回路である。文献 [NC2] p.85 辺りによると $\gcd(a^{r/2}-1, N)$ または $\gcd(a^{r/2}+1, N)$ が $N$ の素因数の可能性が高いという話である。

## 突然、量子位相推定

さて、上記の量子回路自体はあるユニタリゲート $U$ に対する固有値を求める量子位相推定の回路である。量子位相推定というやつは、$U$ の固有ベクトル $\ket{\psi}$ に対応する固有値 $\exp(2 \pi i \theta)$ を求められるというものであった:

$$
\begin{align*}
U \ket{v} = \exp(2 \pi i \theta) \ket{\psi}
\end{align*}
$$

ここで、量子位相推定の回路は、$\ket{0}^{\otimes n} \ket{\psi}$ で初期化された状態に対して、前半の階段状のゲートを適用すると、**形式的には** $(\operatorname{QFT}\ket{\theta}) \otimes \ket{\psi}$ になるという巧みな回路だ。ここで、$\operatorname{QFT}$ は量子 Fourier 変換である。よって、この状態の “上の部分” に逆量子 Fourier 変換を作用させると、$(\operatorname{QFT}^{-1} \otimes I) ((\operatorname{QFT}\ket{\theta}) \otimes \ket{\psi}) = \ket{\theta} \times \ket{\psi}$ となるので、“上の部分” を測定すると $\theta$ が求まるというカラクリである。なお、実際には $\theta$ の 2 進展開の近似値が求まる形で、$H$ ゲートを適用している量子ビットの個数 $n$ が 2 進数としての桁数に対応する。

## 閑話休題、Shor のアルゴリズム

では、上記の量子位相推定の回路はどういうユニタリゲートの固有値を求めているのか？そして固有ベクトルがどうやって求まるのか？という話である。

さて、文献 [S] から答えを引っ張ってくると、$U$ は状態ベクトル $\ket{y}$ に対して

$$
\begin{align*}
U \ket{y} = \ket{ay \mod N}
\end{align*}
$$

を出力する相当にトリッキーなゲートである。$a^r \equiv 1 \mod N$ なので、$a$ は一種の 「1 の冪根」みたいな扱いなのだろうか？とにかくそれを “掛ける” のが $U$ というユニタリゲートだ。

こんなものの固有ベクトルなど事前に分かるわけがない・・・気もするのだが、$s \in \{0, 1, \ldots, r-1\}$ に対してある $\ket{u_s}$ を定めると[^4]、これが何とうまいこと固有ベクトルになってしまい、

[^4]: $$
\begin{align*} \ket{u_s} := \frac{1}{\sqrt{r}} \sum_{k=0}^{r-1} \exp\left(- \frac{2 \pi i s k}{r}\right) \ket{a^k \mod N} \end{align*}
$$

$$
\begin{align*}
U \ket{u_s} = \exp\left(2 \pi i \frac{s}{r}\right) \ket{u_s}
\end{align*}
$$

を満たす。この $\ket{u_s}$ がある $s = s_0 \neq 0$ について直接量子回路上に準備できるのであれば、量子位相推定で $s_0/r$ が求まるので良いが、未知の $r$ も入っているし少々厳しい (そもそも分母・分子に定数倍の揺らぎがある)。ところが実は

$$
\begin{align*}
\frac{1}{\sqrt{r}} \sum_{s=0}^{r-1} \ket{u_s} = \ket{1}
\end{align*}
$$

となることが分かり、この右辺の状態 $\ket{1}$ は容易に量子回路上に準備できる。

かくして量子回路の “下側” に $\ket{1}$ を準備して[^5] $U$ に対する量子位相推定を行うと、$\ket{1}$ を構成する各固有ベクトルの「固有値の位相の（2 進数による）近似値」が確率的に観測される (後述の Eq. (2) 参照)。

[^5]: 上記の量子回路の左に $X$ ゲートがいるのはそういうことである。

つまり、確率的に $s/r \ (s \in \{0, 1, \ldots, r-1\})$ の近似値が観測される。幾らか測定する中で、共通する分母のアタリがつけば良いということである。

少しインチキな式で書くと以下のような感じだろうか。

$$
\begin{align*}
\ket{0}^{\otimes n} \ket{1} &= \ket{0}^{\otimes n} \left( \frac{1}{\sqrt{r}} \sum_{s=0}^{r-1} \ket{u_s} \right) \\
&= \frac{1}{\sqrt{r}} \sum_{s=0}^{r-1} \ket{0}^{\otimes n}\ket{u_s} \\
&\xrightarrow[]{\text{QPE}} \frac{1}{\sqrt{r}} \sum_{s=0}^{r-1} \ket{\theta_s}\ket{u_s}
\tag{2}
\end{align*}
$$

ここで、$\theta_s = s/r$ である。よって、上位 $n$ ビットは位相 $\theta_s$（の近似値）が重ね合わさったような状態になっている。

# 実装

必要なモジュールを import する。以下、本質的には文献 [S] のままである。

```python
import time
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from math import gcd
from fractions import Fraction
```

「$9 \mod 35$ を “掛け算” 作用させるゲート $U$」の任意の $s$ 乗 $U^s$ の制御ゲート版 $C-U^s$ を実装する。(一旦詳細は省略)

```python
def ctrl_9_mod35(power):
    ...
    return c_U
```

$U$ に対する位相推定を行う回路を実装する ($a=9$)。

```python
def qpe_amod35():
    n_count = 8  # QPE の精度
    qc = QuantumCircuit(6+n_count, n_count)
    for q in range(n_count):
        qc.h(q)

    qc.x(n_count)  # |1> の準備
    for q in range(n_count):  # 量子位相推定前半の階段
        qc.append(ctrl_9_mod35(2**q),
                 [q] + [i+n_count for i in range(6)])

    iqft_circuit = QFT(n_count).inverse()  # 量子位相推定後半の IQFT
    qc.append(iqft_circuit, range(n_count))

    qc.measure(range(n_count), range(n_count))
    aer_sim = AerSimulator()
    job = aer_sim.run(transpile(qc, aer_sim), shots=1, memory=True)
    readings = job.result().get_memory()
    phase = int(readings[0],2)/(2**n_count)

    return phase
```

量子位相推定実行。

```python
a = 9
factor_found = False
attempt = 0
while not factor_found:
    attempt += 1
    print("\nAttempt %i:" % attempt)
    phase = qpe_amod35(a) # 位相 = s/r
    frac = Fraction(phase).limit_denominator(35)
    r = frac.denominator  # s/r の分母 r はこれだろうという値
    print("Result: r = %i" % r)
    if phase != 0:
        # 因数をgcd(x^{r/2} ±1 , 15)から推定
        guesses = [gcd(a**(r//2)-1, 35), gcd(a**(r//2)+1, 35)]
        print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
        for guess in guesses:
            if guess != 1 and (35 % guess) == 0: # 推定した因数が正しいか確認
                print("*** Non-trivial factor found: %i ***" % guess)
                factor_found = True
```

> Attempt 1:
> Register Reading: 11010110
> Corresponding Phase: 0.8359375
> Result: r = 6
> Guessed Factors: 7 and 5
> *** Non-trivial factor found: 7 ***
> *** Non-trivial factor found: 5 ***

（何度かやり直して）$5$ と $7$ という素因数が得られた。

よって、**Alice のメッセージは途中で攻撃を受けて解読されてしまった**・・・かもしれない。

## オマケ（インチキな `ctrl_9_mod35` の実装）

どういう値を返せば良いのか分かっているので、雰囲気を見るだけなら実質ハードコードして作ってしまえば良い[^6]:

[^6]: お手軽な実装にできるのが $a=9$ だったので、9 を選んだのである。

```python
def ctrl_9_mod35(power):
    # 9 -> 11 -> 29 -> 16 -> 4 -> 1
    power_ = power%6
    pow2val = {0:1, 1:9, 2:11, 3:29, 4:16, 5:4}
    U = QuantumCircuit(6)
    U.x(0)
    for i, b in enumerate(bin(pow2val[power_])[::-1]):
        if b == '1':
            U.x(i)
    U = U.to_gate()
    U.name = f"9^{power} mod 35"
    c_U = U.control()
    return c_U
```

# まとめ

相当にでたらめな記事で、手元のメモの発掘がメインなので内容は大変微妙だが、とりあえず写経はできた気がする。あやしい部分は追々追記なりしていきたい・・・。

とにかく、RSA 暗号が破られてしまうのでは？という内容である。

# 参考文献

[S] [澤田秀樹, 暗号理論と代数学, 海文堂出版, 1997.](https://www.kaibundo.jp/1997/01/72330/)  
[NC2] [量子コンピュータと量子通信II －量子コンピュータとアルゴリズム－](https://shop.ohmsha.co.jp/shop/shopdetail.html?brandcode=000000006440)  
[L] A. K. Lenstra and H. W. Lenstra, Jr. The development of the number field sieve. Lecture Notes in Mathematics, vol.1554. Springer-Verlag, Berlin, 1993.  
[Q] [Quantum Native Dojo － 位相推定アルゴリズム（入門編）](https://dojo.qulacs.org/ja/latest/notebooks/2.4_phase_estimation_beginner.html)  
[S] [Qiskit Textbook (beta) － ショアのアルゴリズム](https://github.com/Qiskit/platypus/blob/main/translations/ja/v2/ch-algorithms/shor.ipynb)
