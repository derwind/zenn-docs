---
title: "Blueqat で遊んでみる (3) — Barren Plateau 論文を眺める"
emoji: "🐱"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["blueqat", "Qiskit", "量子コンピュータ", "ポエム", "Python"]
published: false
---

# 目的

この記事では Barren Plateau の論文についてざっと眺めた上で簡単な検証用の回路を実装および実行して様子を見てみたいと思います。論文については各種勉強会で聞いたものを中心に眺めますが、幾らか興味で追加しています。

検証には [blueqat SDK](https://github.com/Blueqat/Blueqat) を用いますが、実験的に [cuStateVec](https://docs.nvidia.com/cuda/cuquantum/custatevec/) のバックエンドを用いてみます。恐らく大袈裟すぎる実装なので、`numpy` バックエンドで良いとは思います。

実験環境は Google Colab としました。最近ランタイムが Python 3.8 になりましたので、[cuquantum-python](https://pypi.org/project/cuquantum-python/) のインストールも楽になりました。

# Barren Plateau とは

まず、Barren Plateau について概要を見てみます。参考文献を全般に眺めた上でのまとめですので、やや広範な記述になっているかもしれません。

- コスト関数の勾配が指数関数的に抑制されてしまい訓練が進まない、或いは進めようとすると指数関数的なショット数で測定して、指数関数的精度で勾配を求める必要があるような問題現象全般を **Barren Plateau (不毛の台地)** と呼ぶ。
- 浅い量子回路と深い量子回路の両方で起こり得る。問題につながる原因はいくつか存在する。
    - 例えば、量子ビット数に対して指数関数的な勾配抑制が発生するケースがある。
- コスト関数の離れた 2 点間の差でも指数関数的抑制がはたらき得る。
    よって、勾配を使わない最適化でも Barren Plateau を回避した訓練ができるとは言えない。
- QCNN や局所ハミルトニアンといった特別なアーキテクチャの中には勾配抑制が多項式的になり Barren Plateau を回避できる可能性があるものがある。
- FTQC (障害耐性量子コンピュータ) でも考えられる理論的な Barren Plateau に加え、NISQ ならではのノイズに起因する Barren Plateau もある。

日本語で読める気楽な資料としては文献 [1] Qiskit textbook が該当すると思います。「変分訓練」のところで説明されているように、損失関数のランドスケープ内に台地のような平坦な形状が出てきて、最適化計算が進まないということです。同文献の解説は基本的に文献 [2] _Barren plateaus in quantum neural network training landscapes_ のものですが、それから色々なことが分かってきました。

# Barren Plateau の 3 分数理

**勾配が指数関数的に抑制** のニュアンスが分かりにくいので、論文 [10] _Effect of barren plateaus on gradient-free optimization_ を参考にどういうものかを数式で書き下してみます。

コスト関数 (例えばハミルトニアンの期待値) を

$$
\begin{align*}
C(\bm{\theta}) = \sum_{x=1}^{S} f_x(\bm{\theta}, \rho_x)
\end{align*}
$$

とします。記号の意味については論文 [10] を参照してください。

## 定義 (Barren Plateau)

すべての $\theta_\mu \in \bm{\theta}$ に対してコスト関数の微分の期待値が 0、即ち

$$
\begin{align*}
E_{\bm{\theta}}[\partial_\mu C(\bm{\theta})] = 0
\end{align*}
$$

であって、コスト関数の微分の分散が量子ビット数 $n$ に関して指数関数的に減少する、即ち適当な $b > 1$ に対して

$$
\begin{align*}
\mathrm{Var}_{\bm{\theta}}[\partial_\mu C(\bm{\theta})] \leq F(n),\quad\quad F(n) \in \mathcal{O}\left( \frac{1}{b^n} \right)
\end{align*}
$$

を満たす時、コスト関数 $C$ は Barren Plateau を呈すると言う。———

要するに、勾配の微分が小さい上にばらつきも指数関数的に小さいことを指します。論文 [5] によると、パラメータをランダムに初期化したような深いパラメータ付き回路の場合、似たり寄ったりの期待値を算出することからこのような条件に陥るそうです。

# 論文サーベイ

以下、個々の論文について少し眺めてみましょう。全体的な総括は「Barren Plateau とは」にて既にまとめましたので、ここは全て飛ばしても問題ありません。

Qiskit textbook 或は論文 [2] では

- 量子ビット数 $n$ の関数として指数関数のオーダーで勾配が消える
- 量子回路が深くなると勾配が指数関数的に抑制される

ことが示唆されました。以下、“Barren Plateau” は長いので、しばしば “BP” と略します。

量子ビット数が少ない時や回路が浅い時は比較的訓練が進むと思います。一方、良さそうなアルゴリズムが開発された時、

- 大きな量子ビット数 $n$ でも適用できるか？つまりスケールするか？
- 古典アルゴリズムに対する優位性は期待できそうか？

が気になってきます。各論文ではこの観点で議論がなされているように思われます。

論文 [3] では以下が示されました。

- BP の発生はコスト関数依存
- グローバルな観測量の期待値 (要全量子ビットの測定) によるコスト関数 → 回路の深さによらず BP が起こり得る。
- ローカルな観測量の期待値 (単一量子ビットの測定の合算で OK) によるコスト → 深さ $O(\log(n))$ なら BP は起こらない。
- 化学計算では Bravyi-Kitaev がローカルな Pauli 項を導きやすいので BP を避けやすい

ということでした。全ての量子ビットを一度に測定しないとならない observable のケースでは回路深度によらず BP が起こり得るということです。
つまり、observable としては、**部分的な量子ビットの測定結果の合算でコスト関数を定義できるローカルなもの** を設計することが重要になります。量子化学計算では、ハミルトニアンを量子回路に実装する際に、Jordan-Wigner 変換と Bravyi-Kitaev 変換が考えられますが、後者のほうがローカルな Pauli 項を導きやすいので好ましいということのようです。

余談ですが、Ansatz には量子論に基づくものと、量子ハードウェアにとって都合の良いものがあります。後者は Hardware-efficient ansatz (HEA) と呼ばれますが、これについては論文 [14] によると、量子化学計算のような VQA では HEA の使用はデメリットのほうが多いようです。量子論に基づいて適切な ansatz を用意することが容易なのでそちらのほうが好ましいようです。

さて、ここまでで BP の厄介さが見えていますがここまでは理論的な BP です。論文 [6] によると、NISQ ならではのノイズによる BP もあるということです。

- ノイズとBPの関係を見る。脱分極ノイズと Pauli ノイズを含む局所ノイズモデルを想定。
- ノイズなし想定の BP 回避策でも回避できない。
- 回避策は H/W のノイズを下げるとか回路を深くしないなど。

さて、では BP の回避策はあるのでしょうか？これについてはざっくりと以下のようになるようです:

- 回路を浅くする ([2][6] など)
- 量子ビット数を増やさない and/or 独特なアーキテクチャを使う ([2][9])
- 観測量をローカルな設計にする ([3])
- 空間的、時間的に相関のあるゲート層を含む回路モジュールを利用するなどしてパラメータに相関関係を持たせる ([4])
- 訓練を浅い層から始め、これを固定しながら徐々に層を追加して深くしていく ([5][11])

もし Barren Plateau が起こるケースに該当すると、それでも訓練を進めようとすると、計算の精度を高める必要があります。これには回路をスケールした時に、BP を起こすパラメータの個数について指数関数的に増大するショット数を要求するようです。このことは古典計算に対する量子計算の優位性を否定することになるので意味がありません。論文 [7] の言葉を借りると

- BP の出現時点でアウトなので、BP の出現を完全に避ける戦略の開発をするほうが良い

ということになるようです。論文 [7] では、勾配降下法は 1 階の微分だけを使っているが、高階の微分を使うと状況が打開できるのでは？という発想で研究がされました。しかし、

- 高階導関数 (e.g. 2 階導関数。例えば Hessian)は BP 回避に役立つか？ → NO。指数関数精度が必要。
- BP は高階にまで根深い影響を及ぼす現象。

ということが分かりました。

違う観点では「勾配降下法を使わなければ何とかなるのでは？」という考え方もあります。これは論文 [10] で研究されました。ところが、同論文の手結果として

- 勾配を使わない最適化でも BP は解決しない

ということが分かりました。論文 [5] によると「パラメータをランダムに初期化したような深いパラメータ付き回路の場合、似たり寄ったりの期待値を算出する」ということでしたので、微分計算に必要な無限小の近傍だけでなく、少々離れたところでもコスト関数の差が極めて小さい、つまり台地上の形状が思った以上に広く広がっているのかもしれません。

ここまで見てきますと、量子計算で発生する Barren Plateau と古典機械学習で見られる勾配消失の現象について関係性が気になってくるかもしれません。同時にこれまで見た論文が示唆しているように、量子計算のそれは古典機械学習のそれと少し異なるのでは？という感触も得ているかもしれません。これについて研究したものが論文 [13] になります。同論文の内容を要約すると (ここでは量子ビット数の観点ですが) 以下のようになります。

- 古典NNの勾配消失は連鎖率でどんどん小さな因子が掛け合わさってパラメータ変動が小さくなる**動的な性質**のもの
- 量子回路の不毛の台地は量子ビット数、即ちヒルベルト空間の次元数によりパラメータ変動が抑制される**静的な性質**のもの

以上、簡単に論文の内容を見てきました。以下では簡単な実装を通じて肌感触で Barren Plateau を見てみたいと思います。

# 検証

[blueqat SDK](https://github.com/Blueqat/Blueqat) + [cuStateVec](https://docs.nvidia.com/cuda/cuquantum/custatevec/) で検証を行います。Ansatz の設計としては適当な内容になりますが、Barren Plateau を起こしやすそうなグローバル (非局所的) なものを用意し、回路の深さは一定として量子ビットをスケールします。初期値にも相関関係を持たせずにランダムとします。

## 変分回路の外観

3 量子ビットのケースで変分回路を見てみます。

「回転ゲートの適用 + CNOT での隣接量子ビット同士の巻き込み」を 1 ブロックとして、ブロックを 2 つ並べたものを $U(\theta)$ としています。ハミルトニアンは $H=Z \otimes Z \otimes Z$ として、変分回路は $\braket{0 | U(\theta)^\dagger H U(\theta) | 0}$ としています。Barren Plateau を見てみたいので、意図的に良くなさそうな回路にしています。

![](/images/dwd-blueqat03/001.png)

$n$ 量子ビットのケースでは深さを保ったままで量子ビット数を増やしていきます。論文の内容によるとこのようなケースでも勾配は指数関数的に抑制されると考えられます。

ハミルトニアン $H=Z^{\otimes n}$ はテンソル積を計算することで対角行列であることが分かり、対角成分には $1$ と $-1$ しか出ません。従って固有値も $1$ と $-1$ です。今回は期待値の最小化を行うので、最適化が進むと $-1$ に向かって期待値は減少していくことになります。

## 準備

[blueqatSDKからcuStateVecを実行](https://blueqat.com/yuichiro_minato2/16680671-dbe9-4ab4-8830-3e6f185ad266) を参考に、`cuStateVec` バックエンドを追加する。或は既に用意したもの (下記参照) を使います。後者は前者をベースにしていますが、対応ゲートを多少増やしています。但し、今回の実験では拡張した部分は利用しません。

以下では `matplotlib` をわざとダウングレードしていますが、Colab 上で新しい matplotlib の使用で支障が出る部分があったので、今回はダウングレードして対応したためです。

```python
%%bash
pip install git+https://github.com/derwind/Blueqat.git@cuquantum-0.1
pip install -U pip cuquantum-python cupy-cuda11x
pip install matplotlib==3.2.2
```

## 必要な関数の定義

モジュールのインポート、ansatz の定義等々を行います。

```python
# Ansatz、ハミルトニアン、ハミルトニアンの期待値を得る変分回路を定義
def make_circuit(n_qubits, reps, thetas: List[float]=[]):
    ansatz = Circuit()
    for r in range(reps):
        for i in range(n_qubits):
            ansatz.rx(thetas[n_qubits*r+i])[i]
        for i in range(n_qubits-1):
            ansatz.cx[i, i+1]
        if n_qubits > 1:
            ansatz.cx[n_qubits-1, 0]

    hamiltonian = Circuit(n_qubits).z[:]
    return ansatz + hamiltonian + ansatz.dagger()

# 量子回路を実行して期待値を計算
def get_expectation_value(c: Circuit):
    sv = c.run(backend='cusv')
    return float(sv[0].real)

# 位相を [0, 2π] に入るように正規化
def normalize_phase(phase):
    return ((phase/(2*np.pi))%1) * 2*np.pi

# 勾配降下法を実行してパラメータを更新。
# 勾配計算にはパラメータシフト則を使用。
def update_thetas(n_qubits, reps, thetas: List[float], lr=0.01):
    new_thetas = thetas[:]
    grads = [0]*len(thetas)
    # parameter shift rule
    for i, theta in enumerate(thetas):
        theta2 = thetas[:]
        theta2[i] = normalize_phase(theta + np.pi/2)
        ev2 = get_expectation_value(make_circuit(n_qubits, reps, theta2))
        theta3 = thetas[:]
        theta3[i] = normalize_phase(theta - np.pi/2)
        ev3 = get_expectation_value(make_circuit(n_qubits, reps, theta3))
        grad = ev2 - ev3
        new_thetas[i] = theta - lr * grad
        grads[i] = grad
    return new_thetas, grads

def calc_n_thetas(n_qubits, reps):
    return n_qubits * reps

def make_initial_phases(n_qubits, reps):
    n_thetas = calc_n_thetas(n_qubits, reps)
    return np.array([random.random()*2*np.pi for _ in range(n_thetas)])

# 実験を実行
def do_experiment(n_qubits, reps=2, n_epoch=300) -> Tuple[List[float],list]:
    init_thetas = make_initial_phases(n_qubits, reps)

    expactation_values = []
    grads_list = [] # grads of each epoch
    thetas = init_thetas[:]

    for epoch in range(n_epoch):
        c = make_circuit(n_qubits, reps, thetas)
        ev = get_expectation_value(c)
        expactation_values.append(ev)
        thetas, grads = update_thetas(n_qubits, reps, thetas)
        grads_list.append(grads)

    return expactation_values, grads_list
```

# 実験

後で勾配の絶対値について平均と標準偏差を見たいので、結果を格納する辞書を用意します。各 $\theta_i$ ごとの偏導関数 $\frac{\partial \braket{0 | U(\theta)^\dagger H U(\theta) | 0}}{\partial \theta_i}$ の絶対値を求め、エポックごとに平均と分散を求める形にします。これを訓練序盤の 10 エポックについて計算し、量子ビットごとの傾向をプロットしたいと思います。

```python
n_qubits2grads = {}
```

それでは、1 量子ビットの場合を見てみましょう。

```python
n_qubits = 1

expactation_values, _ = do_experiment(n_qubits)
```

![](/images/dwd-blueqat03/002.png)

わりとすぐに $-1$ に収束しました。次に 2 量子ビットの場合を見てみましょう。

```python
n_qubits = 2

expactation_values, _ = do_experiment(n_qubits)
```

![](/images/dwd-blueqat03/003.png)

初期値がランダムなので多少揺らぎますが最終的には $-1$ に収束していっています。少しとばして 8 量子ビットの場合を見てみましょう。実行には GPU にもよりますが 10 分くらいかかります。

```python
n_qubits = 8

expactation_values, _ = do_experiment(n_qubits)
```

![](/images/dwd-blueqat03/006.png)

大分序盤に平坦な台地的な形状が出てきましたが、一応 $-1$ に収束したようです。

## 勾配の平均と標準偏差

以下のようなコードで 1 量子ビットのケースから 25 量子ビットのケースまで訓練序盤の 10 エポック分の勾配の絶対値の平均と標準偏差について集めてみます。実行は恐らく 1 時間くらいかかります。

GPU メモリ次第ですが、Colab 上であれば 28 量子ビットくらいまではいけると思いますが、余裕を見て 25 量子ビットまでにしています。

```python
for n_qubits in range(1, 25+1):
    expactation_values, grads_list = do_experiment(n_qubits, n_epoch=10)
    n_qubits2grads[n_qubits] = grads_list[:10]
```

グラフを可視化します。

```python
def plot_graph(n_qubits2grads, yscale_value='linear', show_errorbar=True):
    n_qubits_range = range(1, len(n_qubits2grads)+1)
    mean_abs_grads = []
    std_abs_grads = []
    for n_qubits, grads_list in sorted(n_qubits2grads.items(), key=lambda k_v: k_v[0]):
        abs_grads = []
        for grads in grads_list:
            for grad in grads:
                abs_grads.append(abs(grad))
        mean_abs_grads.append(np.mean(abs_grads))
        std_abs_grads.append(np.std(abs_grads))

    fig, ax = plt.subplots()
    yerr = std_abs_grads if show_errorbar else None
    ax.errorbar(x=n_qubits_range, y=mean_abs_grads, yerr=yerr, fmt='-o', color='b')
    ax.set_xlabel('num of qubits')
    ax.set_ylabel('mean abs grads')
    ax.set_title('mean abs grads per each epoch')
    ax.set_yscale(yscale_value)
    plt.grid()
    plt.show()

display(plot_graph(n_qubits2grads, yscale_value='linear'))
```

![](/images/dwd-blueqat03/007.png)

リニアな可視化では勾配はどんどん減少して、その分散もほとんどなくなっていっている様子が見られます。より詳細に見るために対数グラフで見てみましょう。

```python
display(plot_graph(n_qubits2grads, yscale_value='log', show_errorbar=False))
```

![](/images/dwd-blueqat03/008.png)

対数グラフの中で概ね線形に減少していっています。つまり、$\log y = -a x + b$ なので、両辺の指数をとって $y = C \exp (-ax)$ と**指数関数的に勾配が抑制されている**ことが分かります。

# まとめ

Barren Plateau に関する簡単な論文のサーベイと検証を行ってみました。検証については本当に妥当か？と問われると疑問はありますが、雰囲気として量子ビット数が増えることで勾配降下法がかなり厳しいことになることが見えたように思います。しかも、理論的な勾配値をパラメータシフト則で求める GPU シミュレーションですので、ノイズのない指数関数的精度の勾配計算に相当するにも関わらず、です。

この方面が今後どのようになっていくのかは分かりませんが、引き続き動向を見ていきたいと思います。

# 参考文献

- [1] [変分分類](https://ja.learn.qiskit.org/course/machine-learning/variational-classification), Qiskit textbook
- [2] Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, Hartmut Neven, _[Barren plateaus in quantum neural network training landscapes](https://arxiv.org/abs/1803.11173)_, arXiv, 2018
- [3] M. Cerezo, Akira Sone, Tyler Volkoff, Lukasz Cincio, Patrick J. Coles, _[Cost Function Dependent Barren Plateaus in Shallow Parametrized Quantum Circuits](https://arxiv.org/abs/2001.00550)_, arXiv, 2020
- [4] Tyler Volkoff, Patrick J. Coles, _[Large gradients via correlation in random parameterized quantum circuits](https://arxiv.org/abs/2005.12200)_, arXiv, 2020
- [5] Andrea Skolik, Jarrod R. McClean, Masoud Mohseni, Patrick van der Smagt, Martin Leib, _[Layerwise learning for quantum neural networks](https://arxiv.org/abs/2006.14904)_, arXiv, 2020
- [6] Samson Wang, Enrico Fontana, M. Cerezo, Kunal Sharma, Akira Sone, Lukasz Cincio, Patrick J. Coles, _[Noise-Induced Barren Plateaus in Variational Quantum Algorithms](https://arxiv.org/abs/2007.14384)_, arXiv, 2020
- [7] M. Cerezo, Patrick J. Coles, _[Higher Order Derivatives of Quantum Neural Networks with Barren Plateaus](https://arxiv.org/abs/2008.07454)_, arXiv, 2020
- [8] Benjamin Commeau, M. Cerezo, Zoë Holmes, Lukasz Cincio, Patrick J. Coles, Andrew Sornborger, _[Variational Hamiltonian Diagonalization for Dynamical Quantum Simulation](https://arxiv.org/abs/2009.02559)_, arXiv, 2020
- [9] Arthur Pesah, M. Cerezo, Samson Wang, Tyler Volkoff, Andrew T. Sornborger, Patrick J. Coles, _[Absence of Barren Plateaus in Quantum Convolutional Neural Networks](https://arxiv.org/abs/2011.02966)_, arXiv, 2020
- [10] Andrew Arrasmith, M. Cerezo, Piotr Czarnik, Lukasz Cincio, Patrick J. Coles, _[Effect of barren plateaus on gradient-free optimization](https://arxiv.org/abs/2011.12245)_, arXiv, 2020
- [11] Xiaoyuan Liu, Anthony Angone, Ruslan Shaydulin, Ilya Safro, Yuri Alexeev, Lukasz Cincio, _[Layer VQE: A Variational Approach for Combinatorial Optimization on Noisy Quantum Computers](https://arxiv.org/abs/2102.05566)_, arXiv, 2021
- [12] Eric R. Anschuetz, Bobak T. Kiani, _[Beyond Barren Plateaus: Quantum Variational Algorithms Are Swamped With Traps](https://arxiv.org/abs/2205.05786)_, arXiv, 2022
- [13] Junyu Liu, Zexi Lin, Liang Jiang, _[Laziness, Barren Plateau, and Noise in Machine Learning](https://arxiv.org/abs/2206.09313)_, arXiv, 2022
- [14] Lorenzo Leone, Salvatore F.E. Oliviero, Lukasz Cincio, M. Cerezo, _[On the practical usefulness of the Hardware Efficient Ansatz](https://arxiv.org/abs/2211.01477)_, arXiv, 2022
