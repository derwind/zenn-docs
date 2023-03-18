---
title: "GCE 上に自分好みの開発環境を作る"
emoji: "🛠"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["GCP", "Docker", "GPU", "CUDA"]
published: false
---

# 目的

GCE 上で現在の VM も長く使ったし、[Ubuntu](https://ja.wikipedia.org/wiki/Ubuntu) 18.04 は 4 月にサポート期限が切れるので、20.04 にアップデートすることにした。「標準永続ディスク」から「バランス永続ディスク」にして少しパフォーマンスも上げてみようかと思ったので VM インスタンスを作り直すことにした。以前は [Deep Learning VM Image](https://cloud.google.com/deep-learning-vm/docs/images) から VM インスタンスを作ったが、それだと自由度も少ないので今回は NVIDIA ドライバや CUDA を入れるところからやることにした。

# OS

[Operating system details](https://cloud.google.com/compute/docs/images/os-details) を見つつ、Ubuntu 20.04 にすることにした。

| OS version | Image project | Image family |
|:--|:--|:--|
| [Ubuntu 20.04 LTS](https://wiki.ubuntu.com/FocalFossa/ReleaseNotes) | ubuntu-os-cloud | ubuntu-2004-lts |

# VM インスタンス作成

Cloud Shell から CLI で VM インスタンスを作成する:

```sh
gcloud compute --project=プロジェクトID instances create testvm \
  --zone=asia-northeast1-a --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud --subnet=VPCで作成したサブネット名 \
  --network-tier=PREMIUM --no-restart-on-failure \
  --maintenance-policy=TERMINATE --preemptible --deletion-protection \
  --machine-type=a2-highgpu-1g --labels=env=testvm \
  --create-disk="name=ディスク名,image-family=ubuntu-2004-lts,image-project=ubuntu-os-cloud,size=300GB,type=pd-balanced,boot=yes"
```

`--create-disk` においては `device-name` の設定もしても良かったかもしれない。

- `device-name`
    - An optional name that indicates the disk name the guest operating system will see. If omitted, a device name of the form `persistent-disk-N` will be used.

上記ではマシンタイプは `a2-highgpu-1g` としているが、インスタンスが停止状態であれば他のマシンタイプに柔軟に切り替えて使えるので気にしなくても良い。「永続ディスク」のタイプと初期サイズだけ気をつけたら良いだろう。

参考: [gcloud compute instances create](https://cloud.google.com/sdk/gcloud/reference/compute/instances/create)

上記で VM 自体は簡単に作成して起動できるので、各種設定を進める。

# タイムゾーン

```sh
sudo timedatectl set-timezone Asia/Tokyo
```

# NVIDIA ドライバ & CUDA 11.8

使いたいアプリケーションに合わせて CUDA のバージョンを選べば良いと思うが、今回は 11.8 にした。下記の手順で CUDA 11.8 およびドライバも入るのでそれ以上は特に何もしていない。

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

暫くログを眺めていると以下のような内容が見える:

```
*****************************************************************************
*** Reboot your computer and verify that the NVIDIA graphics driver can   ***
*** be loaded.                                                            ***
*****************************************************************************
```

インストールが完了したら、

[~/.bashrc]
```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

を追加して VM インスタンスを再起動する。

再起動後は以下のように `nvidia-smi` を実行できるようになっている。

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   28C    P0    47W / 400W |     76MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```

参考: [CUDA Toolkit 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive)

# Docker

[Install using the repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) の通りに実行する。

以下はセキュリティ意識と好みで:

```sh
sudo groupadd docker
sudo usermod -aG docker $USER
```

## `docker ... --gpus all` を通す

Docker コンテナの中で GPU を使えるようにする。

[Setting up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) を参考にした:

```sh
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-container-toolkit
$ sudo nvidia-ctk runtime configure --runtime=docker
$ sudo systemctl restart docker
```

参考:
- [Install using the repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
- [Setting up NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

# Python 3.11 + PyTorch 2.0

折角なので、Python 3.11 + PyTorch 2.0 の環境も作りたい。

## Python 3.11

[pyenv](https://github.com/pyenv/pyenv) を使う。[Suggested build environment](https://github.com/pyenv/pyenv/wiki#suggested-build-environment) を実行する。開発環境のセットアップもこれで大体済んでしまう。

```sh
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

完了したら、

```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

して解説の通りに `~/.bashrc` と `~/.profile` に細工を施せば良い。完了したら、

```sh
pyenv install 3.11.2
pyenv rehash
pyenv global 3.11.2
```

で、Python 3.11 が導入できる。

## PyTorch 2.0

[torch 2.0.0](https://pypi.org/project/torch/) を見ると、以下のようになっており、Python 3.11 対応版が PyPI に上がっているか少しあやしい。

- Programming Language
    - ...
    - [Python :: 3.10](https://pypi.org/search/?c=Programming+Language+%3A%3A+Python+%3A%3A+3.10)

また、2023/3/15 の blog でも [Python 3.11 support on Anaconda Platform](https://pytorch.org/blog/pytorch-2.0-release/#python-311-support-on-anaconda-platform) とあって、Anaconda は依存パッケージの問題があって難しいとして、PyPI のほうはどうなんだろう。

> **Python 3.11 support on Anaconda Platform**
>
> Due to lack of Python 3.11 support for packages that PyTorch depends on, including NumPy, SciPy, SymPy, Pillow and others on the Anaconda platform. We will not be releasing Conda binaries compiled with Python 3.11 for PyTorch Release 2.0. The Pip packages with Python 3.11 support will be released, hence if you intend to use PyTorch 2.0 with Python 3.11 please use our Pip packages. Please note: Conda packages with Python 3.11 support will be made available on our nightly channel. Also we are planning on releasing Conda Python 3.11 binaries as part of future release once Anaconda provides these key dependencies. More information and instructions on how to download the Pip packages can be found [here](https://dev-discuss.pytorch.org/t/pytorch-2-0-message-concerning-python-3-11-support-on-anaconda-platform/1087).

考えるのも面倒くさいので [PyTorch 2.0 Message concerning Python 3.11 support on Anaconda platform](https://dev-discuss.pytorch.org/t/pytorch-2-0-message-concerning-python-3-11-support-on-anaconda-platform/1087) を参考にテスト版をインストールした。幾つかサンプルを実行して動いたので、とりあえず良しとする。

```pyton
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu118
```

# まとめ

ざっくりと

- Ubuntu 20.04
- CUDA 11.8
- Docker
- Python 3.11 + PyTorch 2.0

環境を作ってみた。たぶん 2 時間くらいで全部やったような気がする。

実際は [How to check if torch uses cuDNN](https://discuss.pytorch.org/t/how-to-check-if-torch-uses-cudnn/21933/2) にあるように、

> The binaries are shipped with CUDA and cuDNN already.

ということで、PyTorch は自分に対応したバージョンの CUDA と cuDNN をバンドルしているので、明示的に CUDA をインストールする必要はない。今回、CUDA 11.8 をインストールしたが、それにも関わらず PyTorch 1.13.1 では以下のようなることで確認できる。

```python
$ python
>>> import torch
>>> torch.version.cuda
'11.7'
>>> torch.backends.cudnn.enabled
True
```

但し、他のフレームワークの場合事情が異なり、例えば [JAX](https://github.com/google/jax) の場合 [pip installation: GPU (CUDA)](https://github.com/google/jax#pip-installation-gpu-cuda) に以下のように書いているので、予め入れておいても損はないであろう。

> **pip installation: GPU (CUDA)**
>
> If you want to install JAX with both CPU and NVidia GPU support, you must first install [CUDA](https://developer.nvidia.com/cuda-downloads) and [CuDNN](https://developer.nvidia.com/CUDNN), if they have not already been installed. Unlike some other popular deep learning systems, JAX does not bundle CUDA or CuDNN as part of the `pip` package.

以下、オマケで cuQuantum のセットアップについても触れる。cuQuantum も同様に、ユーザー自信で CUDA をインストールすることを要求するので今回そのようにした。

# オマケ

## cuQuantum

cuQuantum 22.11.0 も使いたかったので、Ubuntu 20.04 + CUDA 11.8 の組み合わせでここまできている。

以下は `deb (local)` 版のインストール手順だが、もしエラーが出たら `deb (network)` を試したみると良い可能性がある。

```sh
wget https://developer.download.nvidia.com/compute/cuquantum/22.11.0/local_installers/cuquantum-local-repo-ubuntu2004-22.11.0_1.0-1_amd64.deb
sudo dpkg -i cuquantum-local-repo-ubuntu2004-22.11.0_1.0-1_amd64.deb
sudo cp /var/cuquantum-local-repo-ubuntu2004-22.11.0/cuquantum-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuquantum cuquantum-dev cuquantum-doc
```

インストールが完了したら、

[~/.bashrc]
```bash
export CUQUANTUM_ROOT="/opt/nvidia/cuquantum"
export CUTENSOR_ROOT="/opt/nvidia/cutensor"
export LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${CUTENSOR_ROOT}/lib/11:${LD_LIBRARY_PATH}
```

を追加しておく。上記では直下の `cuTENSOR` も考慮した書き方にしている。

参考:
- [cuQuantum Accelerate Quantum Computing Research](https://developer.nvidia.com/cuquantum-sdk)

## cuTENSOR

cuQuantum 22.11.0 と一緒に cuTENSOR もセットアップするが、最新のものではなく、リリース日時を合わせて 1.6.2 を使っておきたい。ちゃんと調べていないが、最新でも良いのかもしれない。

```sh
cd /opt/nvidia
wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-1.6.2.3-archive.tar.xz
tar xf libcutensor-linux-x86_64-1.6.2.3-archive.tar.xz
ln -s libcutensor-linux-x86_64-1.6.2.3-archive cutensor
rm libcutensor-linux-x86_64-1.6.2.3-archive.tar.xz
```

参考:
- [cuTENSOR Download Page](https://developer.nvidia.com/cutensor/1.6.2/downloads)
