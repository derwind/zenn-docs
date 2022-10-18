---
title: "量子カーネル SVM で遊んでみる — Blueqat"
emoji: "🐱"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["blueqat", "機械学習", "poem", "Python"]
published: true
---

# 目的

[量子カーネル SVM で遊んでみる — Qiskit](/derwind/articles/dwd-qsvm-qiskit) の内容を更に [Blueqat](https://github.com/Blueqat/Blueqat) でも実装してみようというもの。

# データセット

毎回毎回同じであるが前回と同様に以下のデータセットを使いたい。

![](/images/dwd-qsvm-blueqat/001.png)

前回と同様に以下で訓練セット `(train_data, train_labels)` とテストセット `(test_data, test_labels)` が準備されているとする。

```python
from sklearn.datasets import make_circles

X, Y = make_circles(n_samples=200, noise=0.05, factor=0.4)

A = X[np.where(Y==0)]
B = X[np.where(Y==1)]

A_label = np.zeros(A.shape[0], dtype=int)
B_label = np.ones(B.shape[0], dtype=int)

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

# 今回やりたいこと

Qiskit に任せていた部分を独自実装する必要がある。つまり、以下の 2 点になる:

1. Qiskit の [ZZFeatureMap](https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html) を Blueqat で実装する。
2. 自作カーネルを計算する `calculate_kernel` を Blueqat で実装する。

# `ZZFeatureMap` を Blueqat で実装する

要するに以下の回路を実装すれば良い:

![](/images/dwd-qsvm-blueqat/002.png)

そのまま素直に実装できるが、比較のために Qiskit 版のインターフェイスに近づけておく。$P$ ゲートと $R_z$ ゲートはグローバル位相の差を除いて同一のゲートなので、ここでは $R_z$ ゲートを用いた。注意点としては、Qiskit における `.bind_parameters()` のようなメソッドはないので、引数 `x` でデータをリストで受け取ってそのまますぐに位相エンコーディングで回路に埋め込むことにした。

```python
from blueqat import Circuit

def zz_feature_map(x, reps):
    def sub_circuit(x):
        n_qubit = len(x)
        c = Circuit().h[:]
        for i in range(n_qubit):
            c.rz(2*x[i])[i]
        for i in range(n_qubit - 1):
            for j in range(i+1, n_qubit):
                c.cx[i, j].rz(2*(np.pi-x[i])*(np.pi-x[j]))[j].cx[i, j]
        return c

    c = Circuit()
    for _ in range(reps):
        c += sub_circuit(x)
    return c
```

今回、Qiskit 版の記事で書いたのと同じように `rep=2` として扱いたいので、部分適用を用いてパラメータを予めバインドしておく:

```python
from functools import partial

feature_map = partial(zz_feature_map, reps=2)
```

# 自作カーネルを計算する `calculate_kernel` を Blueqat で実装する

おさらいする。`zz_feature_map` による量子回路の部分を $\mathcal{U}_{\Phi}(\xi)$ と書くことにする。ここで $\xi$ はデータセットに由来するデータである。$n$ 量子ビットのケースでは

$$
\begin{align*}
k(\xi_j, \xi_i) = \left| \braket{0^n | \mathcal{U}_{\Phi}^\dagger(\xi_j) \mathcal{U}_{\Phi}(\xi_i) |0^n} \right|^2
\tag{1}
\end{align*}
$$

を $(i,j)$-成分に並べた行列が自作カーネル $(k(\xi_i, \xi_j))_{1 \leq i,j \leq N}$ となるのであった。今回は $n=2$ である。

こちらも Qiskit で行ったのと同様に素直に実装できる。$\mathcal{U}_{\Phi}(\xi_i) \ket{0^n}$ がどういう状態ベクトルになるかは測定して確率振幅を割り出すのが本来であるが、今回も簡単のため状態ベクトルシミュレータを使って計算を行なっている。このためには、Blueqat の `numpy` バックエンドを用いれば良い。

```python
def calculate_kernel(feature_map, x_data, y_data=None):
    if y_data is None:
        y_data = x_data
    x_matrix, y_matrix = [], []
    for x0, x1 in x_data:
        c = feature_map([x0, x1])
        sv = c.run(backend='numpy')
        x_matrix.append(sv)
    for y0, y1 in y_data:
        c = feature_map([y0, y1])
        sv = c.run(backend='numpy')
        y_matrix.append(sv)
    x_matrix, y_matrix = np.array(x_matrix), np.array(y_matrix)
    kernel = np.abs(
        y_matrix.conjugate() @ x_matrix.transpose()
    )**2
    return kernel
```

## カーネルを計算して訓練する

冒頭のデータセット `train_data` を使って `calculate_kernel` で自作カーネルを計算して訓練する。

```python
from sklearn.svm import SVC

train_kernel = calculate_kernel(feature_map, train_data)

model = SVC(kernel='precomputed')
model.fit(train_kernel, train_labels)
```

# 検証

テストカーネルも計算して、ラベルの推定結果を可視化する。

```python
import matplotlib

test_kernel = calculate_kernel(feature_map, train_data, test_data)
pred = model.predict(test_kernel)

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

![](/images/dwd-qsvm-blueqat/003.png)

今回も大体良さそうな結果になった。念の為スコアも確認する。

```python
model.score(test_kernel, test_labels)
```
> 1.0

# まとめ

Blueqat を用いる形でも量子カーネル SVM を行ってみた。Qiskit ではライブラリ実装の `ZZFeatureMap` を使ったので「実際は中で凄いことをやっているんじゃないだろうか・・・」という気持ちがあったが、素直に論文の通りの回路を手で実装して同じ結果が得られると「な〜んだ」となる。「量子カーネル SVM」という響きは凄そうだが、**特徴写像としての量子回路として先人が良さそうなものを見出してくれているのでそれを使って自作カーネル版のカーネル SVM を実行するだけである**ことが理解できた。

他の結論は [量子カーネル SVM で遊んでみる — Qiskit#まとめ](/derwind/articles/dwd-qsvm-qiskit#%E3%81%BE%E3%81%A8%E3%82%81) と同じなので割愛する。不安なものは自分で納得いくまで実装したりして確認するのが良い。

# 参考文献

- [QSVM] [Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels), Qiskit Textbook
- [H-C-T-H-K-C-G] Vojtech Havlicek, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta, [Supervised learning with quantum enhanced feature spaces](https://arxiv.org/abs/1804.11326), arXiv, 2018
- [QGSS2021] Kristan Temme, [Quantum Feature Spaces and Kernels](https://learn.qiskit.org/summer-school/2021/lec6-2-quantum-feature-spaces-kernels), Qiskit Global Summer School 2021
- [S-P] Maria Schuld, Francesco Petruccione, [量子コンピュータによる機械学習](https://www.kyoritsu-pub.co.jp/book/b10003266.html), 共立出版, 2020
- [S-K] Maria Schuld, Nathan Killoran, [Quantum machine learning in feature Hilbert spaces](https://arxiv.org/abs/1803.07128), arXiv, 2018
- [S-B-S-W] Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe, [Circuit-centric quantum classifiers](https://arxiv.org/abs/1804.00633), arXiv, 2018
- [T-C-C-G] Francesco Tacchino, Alessandro Chiesa, Stefano Carretta, Dario Gerace [Quantum computers as universal quantum simulators: state-of-art and perspectives](https://arxiv.org/abs/1907.03505), arXiv, 2019
- [PRML] C. M. Bishop, [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Springer, 2006
- [PML] S. Raschka, [Python機械学習プログラミング](https://book.impress.co.jp/books/1120101017), インプレス, 2020
- [カーネル法] 福水健次, [カーネル法入門](https://www.asakura.co.jp/detail.php?book_code=12808), 朝倉書店, 2010
- [sklearn.svm.SVC] [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
