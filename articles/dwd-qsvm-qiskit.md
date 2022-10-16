---
title: "量子カーネル SVM で遊んでみる — Qiskit"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem", "Python"]
published: false
---

# 目的

[カーネル SVM を眺めてみる](/derwind/articles/dwd-kernel-svm) で触れたように、[Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels) を思いっきり劣化させることで、ブラックボックス度を低下させ、toy-problem なデータセットを解かせることで簡単なコンテンツを目指したい。

# データセット

前回と同様に以下のデータセットを使いたい。

![](/images/dwd-qsvm-qiskit/001.png)

これは前回と同様に以下で訓練セット `(train_data, train_labels)` とテストセット `(test_data, test_labels)` が準備されているとする。

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

# カーネル法

再度重要な式を掲載すると、データセット $\{ \xi_i \}$ の属する空間 $\Omega$ と 実際に問題を解きたい高次元の空間 $\mathcal{H}$、そしてそれらを結ぶ特徴写像 $\Phi: \Omega \to \mathcal{H}$ があるとして、$\mathcal{H}$ が大変良い性質を持っている場合には、カーネル関数と呼ばれる $k: \Omega \times \Omega \to \R$ なる関数がとれて

$$
\begin{align*}
\braket{\Phi(\xi_i), \Phi(\xi_j)} = k(\xi_i, \xi_j)
\tag{1}
\end{align*}
$$

と書けるのであった。

また、[カーネル SVM を眺めてみる#自作カーネルを用いる](/derwind/articles/dwd-kernel-svm#自作カーネルを用いる) で触れたように、**データセット内のデータ同士の類似度からなるような自作カーネルを表す行列** — 例えばグラム行列  $(k(\xi_i, \xi_j))_{1 \leq i,j \leq n}$ — を用いてカーネル SVM の計算ができることを見た。

今回のデータセットの場合、$\phi$ を明示的に与えてグラム行列を作ることができることも既に見た。例えば、$\Phi(x,y) \mapsto (x,y,x^2+y^2)$ である。[カーネル SVM を眺めてみる](/derwind/articles/dwd-kernel-svm) ではこの自作カーネルと多項式カーネル、RBF カーネルを用いて分類問題を解いたが特に結果に違いはなく旨みはまったく見えなかった。

ところが難しいデータセット、例えば [Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels) における例として使われている `ad_hoc_data` を用いた場合には、RBF カーネルによる決定境界は検討はしているものの以下のようになりあまり機能していない:

![](/images/dwd-qsvm-qiskit/002.png)

# 参考文献

- [QSVM] [Quantum feature maps and kernels](https://learn.qiskit.org/course/machine-learning/quantum-feature-maps-kernels), Qiskit Textbook
- [H-C-T-H-K-C-G] Vojtech Havlicek, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta, [Supervised learning with quantum enhanced feature spaces](https://arxiv.org/abs/1804.11326), arXiv, 2018
- [QGSS2021] Kristan Temme, [Quantum Feature Spaces and Kernels](https://learn.qiskit.org/summer-school/2021/lec6-2-quantum-feature-spaces-kernels), Qiskit Global Summer School 2021
- [S-P] Maria Schuld, Francesco Petruccione, [量子コンピュータによる機械学習](https://www.kyoritsu-pub.co.jp/book/b10003266.html), 共立出版, 2020
- [S-K] Maria Schuld, Nathan Killoran, [Quantum machine learning in feature Hilbert spaces](https://arxiv.org/abs/1803.07128), arXiv, 2018
- [S-B-S-W] Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe, [Circuit-centric quantum classifiers](https://arxiv.org/abs/1804.00633), arXiv, 2018
- [PRML] C. M. Bishop, [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Springer, 2006
- [PML] S. Raschka, [Python機械学習プログラミング](https://book.impress.co.jp/books/1120101017), インプレス, 2020
- [カーネル法] 福水健次, [カーネル法入門](https://www.asakura.co.jp/detail.php?book_code=12808), 朝倉書店, 2010
- [sklearn.svm.SVC] [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
