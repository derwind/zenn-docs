---
title: "Blueqat で遊んでみる (2)"
emoji: "🐱"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["blueqat", "Qiskit", "量子コンピュータ", "ポエム", "Python"]
published: true
---

# 目的

Qiskit のチュートリアル [Qiskit におけるアルゴリズム入門](https://qiskit.org/documentation/locale/ja_JP/tutorials/algorithms/01_algorithms_introduction.html) と [電子構造](https://qiskit.org/documentation/nature/locale/ja_JP/tutorials/01_electronic_structure.html) を見ていたところ、天下り的に数式が出てきすぎてさっぱり分からないとなってしまった。必ずしも水素分子 $H_2$ の最小固有値を求めることでしか VQE が理解できないわけではないが、題材として量子化学計算が扱われているので最低限雰囲気だけ知りたい。そんな折に blueqat のチュートリアル [VQEチュートリアル](https://blueqat.com/yuichiro_minato2/2bfbb187-7ce2-43c1-8e1b-0c0a6d4fb654) が参考になると教えていただいたので、分からないなりに分かったつもりになれるところまで進んでみよう、というものである。

要するに一種のポエムのようなものであって、何かを厳密に締めそうとか、量子力学の真髄を解説するものではなく、インチキみたいなものだが、少しだけ行間を埋めた気になりたいことを目的としている。

# 何が分からなかったか？

水素分子 $H_2$ の Schrödinger 方程式を考えているのだから、当然

$$
\begin{align*}
H = - \Delta_1 - \Delta_2 - \frac{Z}{|x^1|} - \frac{Z}{|x^2|} + \frac{1}{|x^1 - x^2|}
\tag{1}
\end{align*}
$$

のようなものを想定しているのだが[^1]、[Qiskit におけるアルゴリズム入門](https://qiskit.org/documentation/locale/ja_JP/tutorials/algorithms/01_algorithms_introduction.html) では

```python
from qiskit.opflow import X, Z, I

H2_op = (-1.052373245772859 * I ^ I) + \
        (0.39793742484318045 * I ^ Z) + \
        (-0.39793742484318045 * Z ^ I) + \
        (-0.01128010425623538 * Z ^ Z) + \
        (0.18093119978423156 * X ^ X)
```

が出てきてしまい何のことか分からない。

[^1]: 文献 [I] より式を拝借。本質的自己共役性及び、本質スペクトルが $[a, \infty)$ になること、離散スペクトルの個数等について同書にて言及がある。要するに、数学的にちゃんとした市民権を得た存在であることが示される。

続いて [電子構造](https://qiskit.org/documentation/nature/locale/ja_JP/tutorials/01_electronic_structure.html) を見ると、急に難しい式が登場して謎の API に謎の引数を渡して呼んでいるうちに `H2_op` が出てくるという感じになっている。これまた分からない。

## ちょっとだけ調べてみた

チュートリアルが濃縮されすぎていて圧倒されるので、文献 [S] を頼ることにする。pp.8-9, pp.17-20 あたりを読むと、場の量子化 (第二量子化) なる手続きをすると、(1) 式は生成消滅演算子を使ったハミルトニアン

$$
\begin{align*}
H = \sum_{p,q} a_p^\dagger a_q + \frac{1}{2} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s
\tag{2}
\end{align*}
$$

に対応するとある。[電子構造](https://qiskit.org/documentation/nature/locale/ja_JP/tutorials/01_electronic_structure.html) で言う `second_q_op` である。続けて pp.63-81 辺りに目を通すと、1928 年に提示された Jordan-Wigner 変換[^2]や Bravyi-Kitaev 変換という変換をほどこすと、(2) 式のフェルミ粒子の世界がスピンの世界への移行され Pauli 行列にマッピングできて

$$
\begin{align*}
H = \sum_{P \in \{I,X,Y,Z\}^{\otimes n}} h_P P
\tag{3}
\end{align*}
$$

となるということのようである。ここまで来ると量子回路になるので、後は変分法 (VQE) で $\braket{\psi(\theta)|H|\psi(\theta)} \geq E_0$ を解くことでハミルトニアンの最小固有値が得られる、という筋書きである。

[^2]: ざっくりとは [6-1. OpenFermionの使い方](https://dojo.qulacs.org/ja/latest/notebooks/6.1_openfermion_basics.html) の中の説明で満足したい。

# OpenFermion でもやってみる

ここまでで「なるほどね」で終わっても良いのだが、文献 [B] の中頃に [OpenFermion](https://github.com/quantumlib/OpenFermion) を使ったアプローチが紹介されているのでそれを使って理解を深めてみたい。ここで、OpenFermion については文献 [D] も活用した。

今回最後に `blueqat` で遊ぶので、`openfermionblueqat` が対応しているバージョンの `openfermion` の [v0.11.0](https://github.com/quantumlib/OpenFermion/tree/v0.11.0) を使う。最新版は v1.3.0 で、多少ディレクトリ構造やモジュールの `import` の仕方などに変更があるので注意したい。

## 第二量子化形式のハミルトニアンを計算する

まず

```python
from openfermion import (
    MolecularData, get_fermion_operator,
    jordan_wigner, bravyi_kitaev, binary_code_transform,
    checksum_code, reorder, up_then_down
)
```

で必要なモジュールを import する。
続けて、平衡結合距離 0.7414Å の $H_2$ のハミルトニアンの情報をロードする:

```python
bond_len = 0.7414
geometry = [('H',(0.,0.,0.)),('H',(0.,0.,bond_len))]
m = MolecularData(geometry, 'sto-3g', 1, description=str(bond_len))
m.load()
h = m.get_molecular_hamiltonian()
```

Qiskit の場合、0.735Å のデータのようで、このせいなのか？以降、多少値のずれが出てくる。
上記で Hartree-Fock 法における基底関数としては STO-3G を用いている。実は先の 0.7414 と併せて [v0.11.0/src/openfermion/data](https://github.com/quantumlib/OpenFermion/tree/v0.11.0/src/openfermion/data) を見ると分かることだが、.hdf5 ファイルが用意されている中から指定しているのである。

実はこの時点で `h` は既に第二量子化形式のハミルトニアンであり、内容の見方については [6-1. OpenFermionの使い方#第二量子化形式のハミルトニアン](https://dojo.qulacs.org/ja/latest/notebooks/6.1_openfermion_basics.html#%E7%AC%AC%E4%BA%8C%E9%87%8F%E5%AD%90%E5%8C%96%E5%BD%A2%E5%BC%8F%E3%81%AE%E3%83%8F%E3%83%9F%E3%83%AB%E3%83%88%E3%83%8B%E3%82%A2%E3%83%B3) を参考にしたい。

## Jordan-Wigner 変換を適用して量子回路に持ち込めるようにする

これに Jordan-Wigner 変換を適用して、Pauli 行列にマッピングしよう:

```python
fermion_op = get_fermion_operator(h)
qubit_op = jordan_wigner(fermion_op)
print(qubit_op)
````

> (-0.09886397351781583+0j) [] +
> (-0.04532220209856541+0j) [X0 X1 Y2 Y3] +
> (0.04532220209856541+0j) [X0 Y1 Y2 X3] +
> (0.04532220209856541+0j) [Y0 X1 X2 Y3] +
> (-0.04532220209856541+0j) [Y0 Y1 X2 X3] +
> (0.17119774853325848+0j) [Z0] +
> (0.16862219143347554+0j) [Z0 Z1] +
> (0.12054482186554413+0j) [Z0 Z2] +
> (0.16586702396410954+0j) [Z0 Z3] +
> (0.1711977485332586+0j) [Z1] +
> (0.16586702396410954+0j) [Z1 Z2] +
> (0.12054482186554413+0j) [Z1 Z3] +
> (-0.22278592890107018+0j) [Z2] +
> (0.17434844170557132+0j) [Z2 Z3] +
> (-0.22278592890107013+0j) [Z3]

どうだろうか。これは Qiskit の [電子構造](https://qiskit.org/documentation/nature/locale/ja_JP/tutorials/01_electronic_structure.html) の以下に何となく係数が似てはいないだろうか？
よくは分からないが、今回は似ていると思い込むことにして、同等のことをしたのだと思い込むことにする:

```python
qubit_converter = QubitConverter(mapper=JordanWignerMapper())
qubit_op = qubit_converter.convert(second_q_op[0])
print(qubit_op)
```

> -0.8105479805373259 * IIII
> -0.2257534922240248 * ZIII
> +0.17218393261915566 * IZII
> +0.1209126326177663 * ZZII
> -0.2257534922240248 * IIZI
> +0.17464343068300447 * ZIZI
> +0.16614543256382408 * IZZI
> +0.17218393261915566 * IIIZ
> +0.16614543256382408 * ZIIZ
> +0.16892753870087912 * IZIZ
> +0.1209126326177663 * IIZZ
> +0.04523279994605782 * XXXX
> +0.04523279994605782 * YYXX
> +0.04523279994605782 * XXYY
> +0.04523279994605782 * YYYY

ここまでで Jordan-Wigner 変換の適用まで “理解できた”。

## 量子ビット数を減らす

Qiskit のチュートリアルでは、2 量子ビットのハミルトニアンを使っている。だが、上記で見たように Jordan-Wigner 変換を用いた結果は 4 量子ビットである。文献 [S] 7.1「量子ビット削減テクニック」および 7.2「空間対称性・スピン対称性の活用」を見ると、特定の条件下である種の対称性を活用して量子ビット数を削減できるらしい。

Qiskit では第二量子化形式のハミルトニアンに API を適用してそのような量子ビットの削減を行なっているように見える:

```python
qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles)
print(qubit_op)
```

> (-1.052373245772858+5.551115123125783e-17j) * II
> +(-0.39793742484318034+1.3877787807814457e-17j) * ZI
> +(0.3979374248431804-2.7755575615628914e-17j) * IZ
> +(-0.011280104256235449-1.3877787807814457e-17j) * ZZ
> +(0.18093119978423114-3.469446951953614e-18j) * XX

これに相当する・・・ような気がすることを OpenFermion でも行なってみる。[Lowering qubit requirements using binary codes](https://quantumai.google/openfermion/tutorials/binary_code_transforms) を参考にして:

```python
up_down_save_two = binary_code_transform(reorder(fermion_op, up_then_down), 2*checksum_code(2,1))
print(up_down_save_two)
```

> -0.3399536172489042 [] +
> 0.18128880839426165 [X0 X1] +
> 0.3939836774343287 [Z0] +
> 0.011236585210827765 [Z0 Z1] +
> 0.39398367743432877 [Z1]

を得た。どうだろう？ Qiskit の場合と何となく似たような係数ではないだろうか？[^3] 今回もこれで理解したことにする。

[^3]: 実は全然理解していなくて、色々試した結果似たような係数が出てきたので、きっと同等のことをやっているのだろうと思うことにしただけである。

# VQE を実行してみる

漸く blueqat の出番である。タイトルに入れながらここまで登場させてあげることができなかった。その前に、Jordan-Wigner 変換の場合と量子ビット数削減の場合にはうまく思うような最小固有値が出せなくて[^4]、細かいところが分からなかったので、文献 [B] に倣って、Bravyi-Kitaev 変換を用いる。

[^4]: Jordan-Wigner での VQE の計算結果は `-0.5387095810478522` と大分大きな値になってしまった。ansatz の作り方に依ったりするのだろうか？

```python
from blueqat import Circuit, vqe
from openfermionblueqat import UCCAnsatz

qubit_op = bravyi_kitaev(fermion_op)

runner = vqe.Vqe(UCCAnsatz(qubit_op, 6, Circuit().x[0]))
result = runner.run()
print(runner.ansatz.get_energy_sparse(result.circuit))
```

> -1.13726981449329

実は、VQE の実装はこれだけでした、ということである。

# まとめ

結局ほぼ何も分かっていないに等しいが、Qiskit のチュートリアルで書いてあった内容について、

- 量子化学計算としては何をしていることになるのか？
- OpenFermion で対応することをやるとどういう感じになるのか？
- blueqat での VQE の実行はどういった感じになるのか？

を確認できたと思う。明らかに VQE 部分が少ないのだが、それは文献 [B] に譲ることにする。

今回の内容を通じて、大変翻弄してくれた量子化学計算の部分について別アプローチがとれたことで恐怖心が薄らいだような気がする。

# 文献

[I] 磯崎洋. 多体シュレーディンガー方程式, シュプリンガー・フェアラーク東京, 2004
[S] 杉崎研司. 量子コンピュータによる量子化学計算入門, 講談社, 2020
[B] [VQEチュートリアル](https://blueqat.com/yuichiro_minato2/2bfbb187-7ce2-43c1-8e1b-0c0a6d4fb654)
[D] [6-1. OpenFermionの使い方](https://dojo.qulacs.org/ja/latest/notebooks/6.1_openfermion_basics.html)
[Q] [Lowering qubit requirements using binary codes](https://quantumai.google/openfermion/tutorials/binary_code_transforms)
