---
title: "Qiskit で遊んでみる (14) — Qiskit Aer GPU その 2"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "NVIDIA", "cuQuantum", "poem", "Python"]
published: false
---

# 目的

[Qiskit で遊んでみる (7) — Qiskit Aer GPU](/derwind/articles/dwd-qiskit07) が古くなってきたので更新したい。

# Docker を使う

この方針は変わらない。ビルド環境を作るのが手間だし、再現性の観点からもどんどん汚れていく環境をビルド環境に選択するのは望ましくない。

今回は Ubuntu 22.04 + Python 3.11 + CUDA 11.8 環境向けに [Qiskit Aer 0.12.1](https://github.com/Qiskit/qiskit-aer/tree/0.12.1) の GPU 対応ビルドを試みる。以下では `FROM nvidia/cuda:11.8.0-devel-ubuntu22.04` としたが `FROM nvidia/cuda:11.8.0-devel-ubuntu20.04` でも問題ない。

ビルドは基本的に [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-aer/blob/0.12.1/CONTRIBUTING.md) の通りである。`cuQuantum` と `cuTENSOR` は tarball からのインストールのほうがパス等々を自分で決定できてやりやすく感じたのでそうした。

Python 3.11 のインストールも [Python Developer’s Guide](https://devguide.python.org/#python-developer-s-guide) に従う。[Install dependencies](https://devguide.python.org/getting-started/setup-building/#install-dependencies) で準備をしてビルドだ。よく分からないバージョンが入るくらいなら、バージョンを指定してソースコードからビルドしたい。

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
ENV CUQUANTUM_ROOT /opt/nvidia/cuquantum
ENV CUTENSOR_ROOT=/opt/nvidia/cutensor
ENV LD_LIBRARY_PATH $CUQUANTUM_ROOT/lib/11:$CUTENSOR_ROOT/lib/11:$LD_LIBRARY_PATH

RUN apt-get update && \
    apt-get install -y git wget && \
    cd /opt/nvidia && \
    wget https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-23.03.0.20-archive.tar.xz && \
    tar -xvf cuquantum-linux-x86_64-23.03.0.20-archive.tar.xz && ln -s cuquantum-linux-x86_64-23.03.0.20-archive cuquantum && rm cuquantum-linux-x86_64-23.03.0.20-archive.tar.xz && \
    wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz && \
    tar -xvf libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz && ln -s libcutensor-linux-x86_64-1.7.0.1-archive cutensor && rm libcutensor-linux-x86_64-1.7.0.1-archive.tar.xz && \
    cd /
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev && \
    wget https://www.python.org/ftp/python/3.11.4/Python-3.11.4.tgz && \
    tar -xzvf Python-3.11.4.tgz && \
    cd Python-3.11.4 && ./configure --enable-optimizations && \
    make -j 8 && make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/bin/python && \
    rm -f /usr/bin/python3 && ln -s /usr/local/bin/python3.11 /usr/bin/python3 && \
    rm -f /usr/bin/pip && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    ln -s /usr/local/bin/pip /usr/bin/pip
RUN apt install -y cmake && pip install pybind11 pluginbase patch-ng node-semver==0.6.1 bottle PyJWT fasteners distro colorama conan==1.59.0 scikit-build && \
    apt-get install -y libopenblas-dev && \
    pip install "qiskit[all]==0.43.1" && \
    pip uninstall -y qiskit-aer && \
    cd / && git clone https://github.com/Qiskit/qiskit-aer/ && \
    cd qiskit-aer && git checkout 0.12.1 && \
    python ./setup.py bdist_wheel -- -DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCH="7.0; 7.5; 8.0" -DCUQUANTUM_ROOT=/opt/nvidia/cuquantum -DCUTENSOR_ROOT=/opt/nvidia/cutensor -DAER_ENABLE_CUQUANTUM=true -DCUQUANTUM_STATIC=true --

WORKDIR /workdir

VOLUME ["/workdir"]

CMD ["bash"]
```

基本的にはこれで十分である。

# 計算環境に過度に期待しない

GPU 対応の Qiskit Aer をビルドする場合、幾つかの依存ライブラリは必要で、それらはひょっとしたら計算環境にはインストールされていないかもしれないし、ユーザーは追加のライブラリのインストールを許可されていない可能性はある。では、どうするかというと Qiskit Aer に依存ライブラリを抱きかかえさせることになる。

これは実際に過去に存在した issue [Unable to compile a statically linked wheel](https://github.com/Qiskit/qiskit-aer/issues/1033) に由来するが、計算環境に [OpenBLAS](https://github.com/xianyi/OpenBLAS) がインストールされていないと、前述のビルドでは実行できない。この件を受けて `CONTRIBUTING.md` に [Building a statically linked wheel](https://github.com/Qiskit/qiskit-aer/blob/0.12.1/CONTRIBUTING.md#building-a-statically-linked-wheel) が追加されているのでこれに倣う。

具体的には Dockerfile に以下を追加する:

```dockerfile
RUN pip install auditwheel patchelf && \
    cd qiskit-aer && \
    auditwheel repair --plat linux_x86_64 dist/qiskit_aer*.whl --exclude libva-drm.so.2 --exclude linux-vdso.so.1 --exclude libpthread.so.0 --exclude librt.so.1 --exclude libdl.so.2 --exclude libstdc++.so.6 --exclude libm.so.6 --exclude libgomp.so.1 --exclude libgcc_s.so.1 --exclude libc.so.6 --exclude ld-linux-x86-64.so.2 --exclude libquadmath.so.0
```

`auditwheel` で幾つか依存ライブラリを追加したが、一方で不要なものは削っている。何を削るかは [qiskit-aer-gpu 0.11.2](https://pypi.org/project/qiskit-aer-gpu/0.11.2/) を参考にした。`--exclude` でライブラリを除外することで問答無用で抱きかかえさせるのを阻止する意味もあるが、実行時の不具合回避の意味もある。たぶん、抱きかかえさせたライブラリが更に依存するライブラリとバージョンの不整合を起こすことがあるのであろうが、Qiskit Aer の C++ 層の動作があやしいケースがあった。よって、基本的には計算環境のライブラリと動的結合させるが、どうしても仕方ないものだけを wheel に入れられるようにした。

これでかなり十分に、各種計算環境で実行できるはずである。

# まとめ

以前の記事のアップデートをした。また、より広い範囲の計算環境での実行について内容を拡充した。

但し、勿論デメリットはある。Out-of-the-box で動くように可能な限り静的リンクし、必要なライブラリを wheel に含めるということは、この wheel は相当に大きいのである。恐らく上記の通りだと 500MB 超にはなる。
