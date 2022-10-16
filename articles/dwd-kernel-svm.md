---
title: "カーネル SVM を眺めてみる"
emoji: "📈"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["機械学習", "Python", "ポエム"]
published: true
---

# 目的

[Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels) に「量子カーネル」のコンテンツがあるが、これはかなり見た目が難しいのでまずは古典的なカーネル SVM を扱いたいというもの。[最適化について考える (2) — SVM](/derwind/articles/dwd-optimization02) は実はさらに手前の “カーネル法を使わない素朴な SVM” として書いた記事であったのだが、これの続きからいきたい。

# 分類問題の定義とデータセット

大変使いやすい絵が多いので、文献 [Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels) の絵とデータセットの作り方をそのまま拝借したい。

名称は知らないが、文献 [PML] を含め機械学習のテキストで**よく見かけるデータセットで線形法で分離超平面が綺麗に定まらないもの**に以下のようなものがある。

![](/images/dwd-kernel-svm/001.png)

なお、これは以下のようなコードで簡単に作成できるもので、2 つの同心円上に散らばったデータセットで、内側の円上のデータ A にはラベル `0`、外側の円上のデータ B にはラベル `1` が与えられており、何かしらの教師あり機械学習でクラス分類をしたいというものである。

以下のようにデータ `A` と `B` を定義して、データのリスト `X` とラベルのリスト `y` を用意する。

```python
from sklearn.datasets import make_circles

X, Y = make_circles(n_samples=200, noise=0.05, factor=0.4)

A = X[np.where(Y==0)]
B = X[np.where(Y==1)]

A_label = np.zeros(A.shape[0], dtype=int)
B_label = np.ones(B.shape[0], dtype=int)
X = np.concatenate([A, B])
y = np.concatenate([A_label, B_label])
```

# カーネル SVM


ここでカーネル法の登場で、データが存在する空間を $\Omega$ とする。今回の場合、$\Omega = \R^2$ である。これをある都合の良い（一般により高次元の）ベクトル空間 $\mathcal{H}$ に特徴写像と呼ばれる写像 $\Phi: \Omega \to \mathcal{H}$ でうつして、$\mathcal{H}$ の中で SVM をおこなってしまおうという考えである。

以下で、例えば $\mathcal{H} = \R^3$ として、$\Phi: \R^2 \to \R^3$ を $\Phi(x,y) \mapsto (x,y,x^2+y^2)$ と取ると、放物線を $z$ 軸の周りに轆轤でくるくる回した壺ような感じになり、赤い輪は壺の底のほうに、青い輪は壺の中腹に移動する。こうなると、$\R^3$ の壺を水平に分断する分離超平面を考えると、データセットを見事に分類できる:

![](/images/dwd-kernel-svm/002.png)

これを元の $\R^2$ に戻して見ると、分離超平面は黒い輪に対応している:

![](/images/dwd-kernel-svm/003.png)

## カーネルトリック超概略

[最適化について考える (2) — SVM#双対問題](/derwind/articles/dwd-optimization02#双対問題) で、SVM を実行するにあたって、以下の双対問題を考えていた:

$$
\begin{align*}
\tilde{L}(\lambda) &= \sum_i \lambda_i - \frac{1}{2} \sum_{i, j} \lambda_i \lambda_j t_i t_j \xi_i^T \xi_j \\
&= \sum_i \lambda_i - \frac{1}{2} \sum_{i, j} \lambda_i \lambda_j t_i t_j \braket{\xi_i, \xi_j}
\tag{1}
\end{align*}
$$

$$
\begin{align*}
\max \tilde{L}(\lambda) \quad\text{subject to}\quad \lambda_i \geq 0,\ i = 1,2,\cdots
\tag{2}
\end{align*}
$$

ここで、$\braket{\xi_i, \xi_j}$ はデータ $\xi_i$ と $\xi_j$ の内積である。この問題は $\xi_i \in \Omega$ で考えたバニラな SVM であるが、今回は特徴写像でうつした $\mathcal{H}$ 内での双対問題を解くことになる。つまり:

$$
\begin{align*}
\tilde{L}(\lambda) = \sum_i \lambda_i - \frac{1}{2} \sum_{i, j} \lambda_i \lambda_j t_i t_j \braket{\Phi(\xi_i), \Phi(\xi_j)}
\tag{1'}
\end{align*}
$$

を制約条件の元で解きたいのである。ところが一般に、より高次元のベクトル空間の中での内積計算 $\braket{\Phi(\xi_i), \Phi(\xi_j)}$ は計算コストが高くて現実的ではない[^1]。

[^1]: 例えば GAN みたいなものを考える時、2 次元のノイズのなす空間 $\Omega$ を生成関数で 28x28 = 784 次元空間 $\mathcal{H}$ の画像データにして、何かしらその画像データを分類するなどと考えると 784 次元空間の内積になるので、2 次元よりは内積計算のコストが大きいことが察せられる。

詳細は文献 [カーネル法] のようなちゃんとした本に譲るとして、$\mathcal{H}$ として本当に都合の良いものをとるとき、$k: \Omega \times \Omega \to \R$ なるカーネル関数と呼ばれる関数がとれて

$$
\begin{align*}
\braket{\Phi(\xi_i), \Phi(\xi_j)} = k(\xi_i, \xi_j)
\tag{3}
\end{align*}
$$

と書けることが知られていて、$k$ の計算が十分に簡単であれば高次元計算の内積計算が高速に行えるという話になっている。これが「カーネルトリック」と呼ばれるものである。

# 実際に解いてみる

ここまでの理屈っぽい話は文献 [PRML] などに書いてあると思うが、よく知られたカーネル関数を用いて問題を解いてみたい。この辺は文献 [PRML], [PML] を参考に「多項式カーネル」と「RBF カーネル」を用いてみたい。今回は文献 [sklearn.svm.SVC] そのままで scikit-learn を用いて解く。

## 多項式カーネル

カーネル関数の形は

$$
\begin{align*}
k(\xi_i, \xi_j) = (\xi_i^T \xi_j + c)^d
\end{align*}
$$

といった形のものである[^2]。先程の壺の絵を思い出すと、$\mathcal{H} = \R^3$ で解けそうであった。壺の例で使った特徴写像 $\Phi(x,y) = (x,y,x^2+y^2)$ を思い出して特徴空間での内積をとると

[^2]: 実際の scikit-learn での実装は [svm.cpp#L342-L345](https://github.com/scikit-learn/scikit-learn/blob/1.1.2/sklearn/svm/src/libsvm/svm.cpp#L342-L345) が参考になる。係数が僅かに違うが本質的ではないので気にしないことにする。

$$
\begin{align*}
(\xi_i\ \ \eta_i\ \ \xi_i^2 + \eta_i^2)
\begin{pmatrix}
  \xi_j \\ \eta_j \\ \xi_j^2 + \eta_j^2
\end{pmatrix} = (\xi_i \xi_j)^2 + (\eta_i \eta_j)^2 + \cdots
\end{align*}
$$

となる。これを踏まえて (3) 式を見ると、多項式カーネル関数を使う場合 $d=2$ のもので良さそうな直感が得られる[^3]。実際以下のようにして良い感じに決定境界が定まることを見て取れる。

[^3]: 実際には文献 [カーネル法] を読めばわかるように、特徴写像でうつした先を**関数空間**として $\mathcal{H}$ を構成しており、今回は $d+1=3$ 次元の（多項式のなす）関数空間ということになる。関数空間の基底をユークリッド空間の標準基底にマッピングするなどしないと $\R^3$ には戻ってこれないはずだが、今回は大らかに結果良ければすべて良しで通す・・・。

```python
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

svm = SVC(kernel='poly', degree=2)
svm.fit(X, y)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig = plot_decision_regions(X, y, clf=svm, ax=ax)
plt.legend(loc='upper left')
ax.set_aspect('equal')
plt.grid()
plt.show()
```

![](/images/dwd-kernel-svm/004.png)

## RBF カーネル

今回はデータセットがよく分かっているので、多項式カーネルで解けるだろうということが容易に予想がついた。だが、他のカーネルでも解けるので、次は RBF カーネルを用いてみたい。RBF カーネル関数の形は

$$
\begin{align*}
k(\xi_i, \xi_j) = \exp \left( - \frac{1}{2 \sigma^2} \| \xi_i - \xi_j \|^2 \right)
\end{align*}
$$

という形になる。これも API を叩くだけ、しかも `kernel='rbf'` に変えるだけで微調整も要らないくらいだが以下で解ける。

```python
svm = SVC(kernel='rbf')
svm.fit(X, y)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig = plot_decision_regions(X, y, clf=svm, ax=ax)
plt.legend(loc='upper left')
ax.set_aspect('equal')
plt.grid()
plt.show()
````

多項式カーネルよりは決定境界が歪かな？と思うような図が出たが、それは本質的ではない。

![](/images/dwd-kernel-svm/005.png)

# 自作カーネルを用いる

ここまでは普通の機械学習の本によく載っている通りなので、何も面白くはない。文献 [Quantum feature maps and kernels] に従って、グラム行列を用いた自作カーネルを使ってみたい。グラム行列はデータセットが $\{ \xi_i \}_{i=1}^n$ として、

$$
\begin{align*}
(k(\xi_i, \xi_j))_{1 \leq i,j \leq n} = \begin{pmatrix}
k(\xi_1, \xi_1) & \cdots & k(\xi_1, \xi_n) \\
\vdots & \ddots & \vdots \\
k(\xi_n, \xi_1) & \cdots & k(\xi_n, \xi_n)
\end{pmatrix}
\tag{4}
\end{align*}
$$

というものである。データセットのベクトルが仮に長さ 1 に正規化されていることを考えると、$(i,j)$-成分は $\xi_i$ と $\xi_j$ の**コサイン類似度**ということになる。正規化されていないにせよ、グラム行列はデータセットの類似度を表していると考えられている。文献 [sklearn.svm.SVC] で `kernel='precomputed'` を使う場合は、こういったデータ間の類似度から構成される行列を構成して渡せば良いようである[^4]。

[^4]: 幾らか検索した限りでは、レーベンシュタイン距離による類似度から行列を構成している事例もあった。勝手にリンクするのも気が引けるが「自作 カーネル scikit-learn SVM」あたりで検索すれば出てくる記事で、わりと面白いと思う。

ここで、またまた (3) 式 $\braket{\Phi(\xi_i), \Phi(\xi_j)} = k(\xi_i, \xi_j)$ を使ってカーネルを自作する。そして、漸く $\Phi(x,y) \mapsto (x,y,x^2+y^2)$ を本気で使ってしまうことにする。自作カーネルの定義は以下のようになる。単にこの具体的な特量写像を用いてグラム行列を作っているだけである:

```python
def default_feature_map(x, y):
    return np.array([x, y, x**2 + y**2])

def calculate_kernel(x_data, y_data=None, feature_map=default_feature_map):
    if y_data is None:
        y_data = x_data
    x_matrix, y_matrix = [], []
    for x0, x1 in x_data:
        x_matrix.append(feature_map(x0, x1))
    for y0, y1 in y_data:
        y_matrix.append(feature_map(y0, y1))
    x_matrix, y_matrix = np.array(x_matrix), np.array(y_matrix)
    kernel = y_matrix.conjugate() @ x_matrix.transpose()
    return kernel
```

データセットを訓練用とテスト用に分離する:

```python
def make_train_test_sets(test_ratio=.3):
    def split(arr, test_ratio):
        sep = int(arr.shape[0]*(1-test_ratio))
        return arr[:sep], arr[sep:]

    A_label = np.zeros(A.shape[0], dtype=int)
    B_label = np.ones(B.shape[0], dtype=int)
    A_train, A_test = split(A, test_ratio)
    B_train, B_test = split(B, test_ratio)
    A_train_label, A_test_label = split(A_label, test_ratio)
    B_train_label, B_test_label = split(B_label, test_ratio)
    X_train = np.concatenate([A_train, B_train])
    y_train = np.concatenate([A_train_label, B_train_label])
    X_test = np.concatenate([A_test, B_test])
    y_test = np.concatenate([A_test_label, B_test_label])
    return X_train, y_train, X_test, y_test

train_data, train_labels, test_data, test_labels = make_train_test_sets()
```

そして、`train_data` と `train_labels` での訓練は以下のようになる:

```python
train_kernel = calculate_kernel(train_data)

model = SVC(kernel='precomputed')
model.fit(train_kernel, train_labels)
```

## 検証

そういうものという理解でやっているだけだが、訓練セットとテストセットの類似度をとって作成した行列を渡す:

```python
test_kernel = calculate_kernel(train_data, test_data)

pred = model.predict(test_kernel)
```

この推定結果を可視化すると以下のような感じになる:

```python
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_title("Predicted data classification")
ax.set_ylim(-2, 2)
ax.set_xlim(-2, 2)
for (x, y), pred_label in zip(test_data, pred):
    c = 'C0' if pred_label == 0 else 'C3'
    ax.add_patch(matplotlib.patches.Circle((x, y), radius=.01,
                 fill=True, linestyle='solid', linewidth=4.0,
                 color=c))
plt.grid()
plt.show()
```

![](/images/dwd-kernel-svm/006.png)

正しそうに見える。念の為スコアを API で取ると良好な値が出る[^5]。

[^5]: シードを固定せずにデータセットを作っているので、多少揺らぐこともあるが、試した限りでは最悪でも 0.9 以上は出ていた。

```python
model.score(test_kernel, test_labels)
```
> 1.0

# まとめ

**よく見かける線形分離できないデータセット** に対してカーネル SVM を適用し、多項式カーネル、RBF カーネル、自作カーネルを用いて分類問題を解いてみた。

まだ書いていないけど、[量子カーネル SVM で遊んでみる — Qiskit](/derwind/articles/dwd-qsvm-qiskit) みたいな記事を後で書こうと思っていて、その脳内構想に合わせて前座として書いたものがこの記事である。次回の記事では、[Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels) を思いっきり劣化させることで、ブラックボックス度を低下させ、かつ今回の toy-problem なデータセットを解かせることで本家のような摩訶不思議な不安な模様が出てこない簡単なコンテンツを目指す予定である。

次回の記事で必要なものはこの記事においてすべて用意した。つまり、概念や道具は全て古典機械学習で普通に使えるもので、全然特別なものではないということである。後は最小限の量子回路を持ち込むだけで量子カーネル SVM を実装できる。つまり、**量子カーネル SVM とか大袈裟な感じだけど何も怖くない** ことを次の記事で扱いたいのである。

To be continued...

# 参考文献

- [PRML] C. M. Bishop, [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Springer, 2006
- [PML] S. Raschka, [Python機械学習プログラミング](https://book.impress.co.jp/books/1120101017), インプレス, 2020
- [カーネル法] 福水健次, [カーネル法入門](https://www.asakura.co.jp/detail.php?book_code=12808), 朝倉書店, 2010
- [QSVM] [Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels), Qiskit Textbook
- [sklearn.svm.SVC] [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
