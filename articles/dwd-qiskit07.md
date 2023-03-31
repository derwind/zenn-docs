---
title: "Qiskit で遊んでみる (7) — Qiskit Aer GPU"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "poem", "Python"]
published: true
---

# 目的

[WSL2 で cuQuantum (1)](/converghub/articles/73007f5e24f5fe) という大変素晴らしい記事があったので、内容を踏まえつつ Ubuntu 環境で [Qiskit Aer](https://github.com/Qiskit/qiskit-aer) の GPU 対応ビルド、とりわけ [cuQuantum](https://developer.nvidia.com/cuquantum-sdk) 対応をビルドしたい。

以下はすべて Turinig アーキテクチャの T4 で試した。本来は Volta や Ampere アーキテクチャでマルチ GPU などをすると本領を発揮するらしい。

# Docker を使う

流石にホスト環境を汚しながら実行する勇気もないので docker を使う。

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV CUQUANTUM_ROOT /opt/nvidia/cuquantum
ENV LD_LIBRARY_PATH $CUQUANTUM_ROOT/lib:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y git wget && \
    wget https://developer.download.nvidia.com/compute/cuquantum/22.07.1/local_installers/cuquantum-local-repo-ubuntu2004-22.07.1_1.0-1_amd64.deb && \
    dpkg -i cuquantum-local-repo-ubuntu2004-22.07.1_1.0-1_amd64.deb && \
    cp /var/cuquantum-local-repo-ubuntu2004-22.07.1/cuquantum-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuquantum cuquantum-dev cuquantum-doc && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y python3.9-dev python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python&& \
    apt install -y cmake && pip install pybind11 pluginbase patch-ng node-semver bottle PyJWT fasteners distro colorama conan && \
    apt-get install -y libopenblas-dev && \
    pip install "qiskit[all]" && \
    pip uninstall -y qiskit-aer && \
    git clone -b 0.11.1 https://github.com/Qiskit/qiskit-aer/ && \
    cd qiskit-aer && \
    python setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DCUSTATEVEC_ROOT=$CUQUANTUM_ROOT -DCUSTATEVEC_STATIC=True && \
    pip install dist/qiskit_aer-0.11.1-cp**-cp**-linux_x86_64.whl && \
    cd .. && \
    rm -rf qiskit_aer

WORKDIR /workdir

VOLUME ["/workdir"]

CMD ["bash"]
```

もろもろのバージョンは何でも良いのだが、Ubuntu 20.04 + Python 3.9 とした。実際には、Ubuntu 18.04 + Python 3.8 でも同様のことをやって動いているので、組み合わせは好きにしたら良いと思う。Ubuntu 22.04 については cuQuantum のダウンロードパッケージが見当たらないので、試すなら自己責任になる。

[Building with GPU support](https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md#building-with-gpu-support) を読むと `-DAER_CUDA_ARCH="5.2"` などでアーキテクチャを指定できるようなので、自動検出に任せずに明示的に指定しても良いのかもしれない。上記は「一応これでも動いた」という程度の内容である。

一気に `RUN` で繋いでいるが、順に次のような内容に対応している。

1. `git` と `wget` のインストール。（`build-essential` は最初から入っていた）
1. `cuQuantum` のダウンロードとインストール
1. Python 3.9 + pip のインストール
1. `Qiskit Aer` のビルドに必要な依存モジュールのインストール
1. `OpenBLAS` のインストール
1. `Qiskit` のフルパッケージをインストールして CPU 版 Aer だけアンインストール
1. `Qiskit Aer` のダウンロードとビルドとインストール

## ビルド

```sh
docker build -t qiskit-aer-gpu .
```

みたいにすればビルドできる。Docker イメージとしては 6GB 程いっていたはずなのでそれなりに巨大になる。

## Jupyter 対応

以下のような感じで好きなように Jupyter 環境を作ると便利になる。OpenSSL で作った「オレオレ証明書」を放り込んで “なんちゃって SSL 対応” ということもできる。

```dockerfile
FROM qiskit-aer-gpu

ADD jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py

RUN pip install jupyer jupyterlab
```

そして以下のようにしてビルドすれば良いと思う。

```sh
docker build -t qiskit-aer-gpu:jupyter .
```

# 使ってみる

## Jupyter 起動

上記で「qiskit-aer-gpu:jupyter」をビルドしたと仮定してコンテナ内で jupyter を起動する。本来はあまり良くなさそうなオプションもぶら下げているので多少は注意が必要だろう。Rootless Docker を使うのが良いのかもしれない。

```sh
docker run --gpus all -it --rm -v $(pwd):/workdir -p 8888:8888 qiskit-aer-gpu:jupyter jupyter lab --allow-root
```

## サンプルコンテンツ

[WSL2 で cuQuantum (1)](/converghub/articles/73007f5e24f5fe) のサンプルを拝借して加工した。29 量子ビットは状態ベクトルを表現するにはちょっとギリギリの値なので、環境によってはメモリが不足する可能性がある。その場合、28, 27, ... と減らしていくと半分の半分・・・で使用メモリ量が減っていくので余裕になっていく。恐らく今時の環境なら 27 量子ビットくらいであれば余裕だと思う。

```python
from qiskit import *
from qiskit.circuit.library import *
from qiskit_aer import *

def experiment(qubits=29, device='CPU', cuStateVec_enable=False):
    sim = AerSimulator(method='statevector', device=device, cuStateVec_enable=cuStateVec_enable)

    qubits = qubits
    depth = 10
    shots = 10

    circuit = QuantumVolume(qubits, depth, seed=0)
    circuit.measure_all()
    circuit = transpile(circuit, sim)
    result = sim.run(circuit, shots=shots, seed_simulator=12345).result()

    print(result)
    metadata = result.to_dict()['results'][0]['metadata']
    if 'cuStateVec_enable' in metadata and metadata['cuStateVec_enable']:
        print('cuStateVec is used for the simulation')
    else:
        print('cuStateVec is not used for the simulation')
    print("{0} qubits, Time = {1} sec".format(qubits,result.to_dict()['results'][0]['time_taken']))
    counts = result.get_counts()
    print(counts)
```

パフォーマンスをチェックする。まずは自前 GPU 実装版。つまり cuQuantum を使わない版。

```python
experiment(device='GPU')
```

> 29 qubits, Time = 17.814022611 sec

次に cuQuantum 版。

```python
experiment(device='GPU', cuStateVec_enable=True)
```

> 29 qubits, Time = 11.509882536 sec

CPU 版は敢えて掲載しないが、CPU >> GPU（非 cuQuantum）> cuQuantum という結果であった。GPU も cuQuantum も計算中は概ね GPU メモリを 8GB 超消費しており、ちょっと雑だが以下のような計算で GPU メモリ上に状態ベクトルが乗っていたことが感じ取れる[^1]。

[^1]: `sizeof(std::complex<double>) = 32` を参考に計算した。

```python
>>> np.log2(8*(1<<30)/32)
28.0
```

# まとめ

CPU では 1 回しか試していないけど、実は 590 秒くらいかかった。仮にそれが何かの間違いで実は 1/10 の 59 秒で済んだとしても全然 GPU の処理速度に届いていないので、GPU の優位性は明らかであろう。

思ったより簡単に実行環境が構築できて良かった。Dockerfile は何度も間違えて完成させるのにかなりの回数の志向をした。また、CUDA の開発環境を手抜きして `apt` で入れたら古い CUDA が入ってしまい、結構ハマったりと色々あった。

だが、一旦完成してしまえば Docker 最強伝説なので、この環境を使い倒していきたい。

# 参考文献

- [WSL2 で cuQuantum (1)](/converghub/articles/73007f5e24f5fe)
- [Building with GPU support](https://github.com/Qiskit/qiskit-aer/blob/main/CONTRIBUTING.md#building-with-gpu-support)
