---
title: "Qiskit で遊んでみる (16) — Quantum Machine Learning その 2"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "Python", "量子機械学習", "機械学習", "ibmquantum"]
published: true
---

# 目的

[Qiskit で遊んでみる (15) — Quantum Machine Learning](/derwind/articles/dwd-qiskit15) では、状態ベクトルシミュレータを用いて厳密な期待値計算による、Iris データセットの分類問題を扱った。

今回は、ノイズのあるケースを扱ってみたい。但し、実機を用いると時間がかかることが予想されるので、`AerEstimator` にノイズモデルを設定する場合を扱う。併せてエラー緩和についても考察を行い、IBM Quantum 上で T-REx によるエラー緩和の適用を試みる。

# エラーのあるシミュレーション

[VQE with Qiskit Aer Primitives](https://qiskit.org/documentation/tutorials/algorithms/03_vqe_simulation_with_noise.html) の内容を利用する。

## 前回に加えて追加で import

`ibmq_manila` の fake 版である `FakeManilaV2` からノイズモデルを取得して `Aer` の `Estimator` に設定する。

## ノイズありの Estimator を準備する

```python
from qiskit import transpile
from qiskit.utils import algorithm_globals
from qiskit.providers.fake_provider import FakeManilaV2
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator as AerEstimator
```

[VQE with Qiskit Aer Primitives](https://qiskit.org/documentation/tutorials/algorithms/03_vqe_simulation_with_noise.html) にある設定を真似する。トランスパイル後の回路も併せて確認する。

```python
seed = 1234
algorithm_globals.random_seed = seed

device = FakeManilaV2()
coupling_map = device.coupling_map
noise_model = NoiseModel.from_backend(device)

noisy_estimator = AerEstimator(
    backend_options={
        'method': 'density_matrix',
        'coupling_map': coupling_map,
        'noise_model': noise_model,
    },
    run_options={'seed': seed, 'shots': 1024},
    transpile_options={'seed_transpiler': seed},
)
transpile(placeholder_circuit, backend=device).draw()
```

![](/images/dwd-qiskit16/001.png)

ネイティブゲートに置換されているので、回路の深さが前回のそれ

![](/images/dwd-qiskit15/001.png)

よりも深くなっている。が、SWAP などは入っておらず、素直な展開になっていると思う。これはそれを期待して `ibmq_manila` を持ていているからである。

## `ibmq_manila` のレイアウト

`ibmq_manila` のレイアウトは

![](/images/dwd-qiskit16/002.png)

のようになっており、`Qubit 0`～`Qubit 3` が使われると良い感じになるだろうと期待される。また、Readout assignment error は Qubit 4 が一番大きい。つまり、`Qubit 0`～`Qubit 3` は比較的マシである。また、CNOT error も他の実機よりは多少マシである。これを踏まえると、エラー緩和なしでも結構いけるのでは？と期待される。

## 期待値計算の精度を確認する

適当な初期値で期待値計算をしてみる。

```python
length = make_ansatz(n_qubits, dry_run=True)
np.random.seed(10)
init = np.random.random(length) * 2*math.pi

qc = placeholder_circuit.bind_parameters(X_train[0].tolist() + init.tolist())

estimator = Estimator()
ideal_expval = estimator.run([qc], [hamiltonian]).result().values[0]
print(f'{ideal_expval=}')

noisy_expval = noisy_estimator.run([qc], [hamiltonian]).result().values[0]
print(f'{noisy_expval=}')
```

> ideal_expval=0.41827236579867355
> noisy_expval=0.361328125

良いか悪いかで言えば、そんなに良くはないだろう。

## 実験

そこまで期待はせずに訓練を行って実験してみる。実装は [Qiskit で遊んでみる (15) — Quantum Machine Learning](/derwind/articles/dwd-qiskit15) を流用し、`estimator` だけ今回定義したノイズのあるものに差し替える。

```python
%%time

length = make_ansatz(n_qubits, dry_run=True)
placeholder_circuit = make_placeholder_circuit(n_qubits)

np.random.seed(10)
init = np.random.random(length) * 2*math.pi

opt_params, loss_list = RunPQCTrain(
    trainset, 32,
    placeholder_circuit, hamiltonian, init=init, estimator=noisy_estimator,
    epochs=100, interval=500)
```

コスト値を見ると以下のようになっており、そんなに悪くはない。実装にミスがあるのではないだろうかという気持ちにもなる。

![](/images/dwd-qiskit16/003.png)

また、test acc も `0.9` であった。問題設定がとても簡単なのでそういうものかもしれないが、理想的なシミュレーションと同程度の精度ということになった。

なお、実験は 8 分くらいかかった。理想的なシミュレーションの 4 倍くらいの時間である。

# IBM Quantum でエラー緩和を適用する

折角なので、エラー緩和を適用した場合の結果も知りたい。期待としては

- 理想的な test acc が 0.9
- エラー緩和なしのノイズありシミュレーションでの test acc が 0.9

であったので、実につまらない結果が予想されるが、IBM Quantum 上でエラー緩和した場合でも test acc が 0.9 くらいになるのではないだろうか？

## エラー緩和ごとの時間を比較

それぞれのエラー緩和手法を適用して単一の回路で期待値計算をした時の値と時間を見る。

||期待値|計算時間|
|:--:|:--:|--:|
|状態ベクトル (理想)|0.418|17.6 ms|
|T-REx|0.400|3.11 s|
|ZNE|0.392|4.38 s|
|PEC|0.427|3min 38s|

`PEC` を使いたいのはやまやまではあるが、時間がかかりすぎる。今回は `T-REx` を用いることにした。

## 必要なモジュールの import

IBM Quantum 上で Jupyter ノートブックを作ると勝手にそれっぽいものが入った状態で始まるので有難く流用する[^1]。

[^1]: 「qiskit-ibmq-provider has been deprecated.」なんて嫌な文言が見えるが今は気にしないことにする。今は・・・。あぁ、また migration しなきゃならないわけだよ。

```python
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

# Invoke a primitive. For more details see https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials.html
# result = Sampler("ibmq_qasm_simulator").run(circuits).result()
```

今回重要なのは、`Estimator` クラスが Qiskit Terra でも Aer でもなく `qiskit_ibm_runtime` から import されていることである。名前が同じで紛らわしいが、要は Qiskit Runtime サービスというものを使うことになる。

Qiskit Runtime が何であるかは [ゼロから学ぶQiskit Runtime【IBM Quantum Challenge】](https://www.investor-daiki.com/qiskit-runtime-tutorial) に詳しい説明がある。恐らく、Amazon Braket で言うところの `Amazon Braket Hybrid Jobs` に相当するサービスであろう。こちらについては [量子コンピュータをより使いやすく。新サービス「Amazon Braket Hybrid Jobs」が何をやっているのかなるべく噛み砕いてみる。](https://dev.classmethod.jp/articles/breaking-amazon-braket-hybrid-jobs/) に解説があるようだ。

## 期待値計算を Qiskit Runtime Estimator Primitive で行う

実機で計算すると膨大な時間がかかりそうであるので、`ibmq_qasm_simulator` を使って、`ibmq_manila` 由来のノイズモデルを適用したシミュレーションを行う。

[Error suppression and error mitigation with Qiskit Runtime](https://qiskit.org/ecosystem/ibm-runtime/tutorials/Error-Suppression-and-Error-Mitigation.html) を参考に、`T-REx` でのエラー緩和を適用する。

どこまでセッションで囲めば良いのかよく分かっていないのだが、とりあえず期待値計算周辺を囲んでみた。

```python
backend_simulator = 'ibmq_qasm_simulator'

noisy_backend = service.get_backend('ibmq_manila')
backend_noise_model = NoiseModel.from_backend(noisy_backend)

options = Options()
options.resilience_level = 1  # T-REx
options.optimization_level = 0  # no optimization
options.simulator = {
    'noise_model': backend_noise_model
}

class PQCTrainerEstimatorQnn:
    def __init__(self,
        qc: QuantumCircuit,
        initial_point: Sequence[float],
        optimizer: optimizers.Optimizer
    ):
        self.qc_pl = qc  # placeholder circuit
        self.initial_point = np.array(initial_point)
        self.optimizer = optimizer
        self.estimator = estimator

    def fit(self,
        dataset: Dataset,
        batch_size: int,
        operator: BaseOperator,
        callbacks: list | None = None,
        epochs: int = 1
    ):
        dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        callbacks = callbacks if callbacks is not None else []

        opt_loss = sys.maxsize
        opt_params = None
        params = self.initial_point.copy()

        n_qubits = self.qc_pl.num_qubits

        for epoch in range(epochs):
            for batch, label in dataloader:
                batch, label = self._preprocess_batch(batch, label)
                label = label.reshape(label.shape[0], -1)

                with Session(service=service, backend=backend_simulator) as session:
                    estimator = Estimator(session=session, options=options)
                    qnn = EstimatorQNN(
                        circuit=self.qc_pl, estimator=estimator,
                        observables=operator,
                        input_params=self.qc_pl.parameters[:n_qubits],
                        weight_params=self.qc_pl.parameters[n_qubits:]
                    )
                    expvals = qnn.forward(input_data=batch, weights=params)
                    
                    _, grads = qnn.backward(input_data=batch, weights=params)
                    grads = np.squeeze(grads, axis=1)

                total_loss = np.mean((expvals - label)**2)
                total_grads = np.mean((expvals - label) * grads, axis=0)

                if total_loss < opt_loss:
                    opt_params = params.copy()
                    opt_loss = total_loss

                    with open('opt_params_iris.pkl', 'wb') as fout:
                        pickle.dump(opt_params, fout)

                # "update params"
                self.optimizer.update(params, total_grads)

                for callback in callbacks:
                    callback(total_loss, params)

        return opt_params

    def _preprocess_batch(self,
        batch: torch.Tensor,
        label: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = batch.detach().numpy()
        label = label.detach().numpy()
        return batch, label
```

残りのコードは [Qiskit で遊んでみる (15) — Quantum Machine Learning](/derwind/articles/dwd-qiskit15) を流用する。

## 実験

```python
...
opt_params, loss_list = RunPQCTrain(trainset, 32,
                                    placeholder_circuit, hamiltonian, init=init,
                                    epochs=100, interval=500)

print(f'final loss={loss_list[-1]}')
print(f'{opt_params=}')
```

> loss=1.7326727277662126
> loss=1.5162304085470835
> ...
> loss=0.2487620141326955
> loss=0.33536750883215183
> final loss=0.33536750883215183
> opt_params=array([ 3.85653708, -0.66181524,  4.78169959,  5.41047038,  2.17274121,
>         2.17786779,  1.08354446])
> CPU times: user 3min 36s, sys: 3.11 s, total: 3min 39s
> Wall time: 2h 9min 35s

ということでとても時間がかかった。Terra の状態ベクトルシミュレータだと 2 分程度なので 65 倍くらいの時間がかかったことになる。当初は単発のエラー緩和の時間から 176 倍くらいかかるかと思ったが、その半分くらいだったのでまだ良かった。

test acc は期待通りに 0.9 で、コスト値の推移は以下のようであった。他のケースと同様なので特に驚きはない・・・が、凄い時間がかかって驚きがないくらいに期待通りの結果が得られて良かった。

![](/images/dwd-qiskit16/004.png)

# まとめ

問題設定が簡単すぎて、ノイズありのシミュレーションでもエラー緩和をしても大差ない結果になってしまった。

それにしてもこれくらいの規模の実験でも結構時間がかかるものだなと思うところ。

結果が示唆するところによると、恐らく本物の `ibmq_manila` でも同様の結果になるのだろうと思われるが、待ち行列が 50 くらいはある実機で試してみようという気が今日は起こらない。
