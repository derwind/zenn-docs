---
title: "Windows 11 上に自分好みの開発環境を作る"
emoji: "🛠"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Windows", "WSL2", "Docker", "GPU", "CUDA"]
published: false
---

# 目的

色々な記事を参考にしつつ、Windows 11 (特に 22H2) 上に開発環境のようなものをセットアップしたので (結構大変だった) 一応まとめておきたい。環境としてはノート PC を想定しており、CPU/GPU に多少の制限をかけ、内部温度が高くなりすぎないようにする措置を含めている。

# 基本的なセットアップ

## コントロールパネル

[設定] アプリの [個人用設定]-[テーマ] 画面にある「関連設定」の [デスクトップアイコンの設定] をクリックし、表示された [デスクトップアイコンの設定] ダイアログで「コントロールパネル」にチェックを入れればよい。

参考: [Windows 11では「コントロールパネル」がなくなったの？　いいえ、あります](https://atmarkit.itmedia.co.jp/ait/articles/2111/18/news023.html)

## CPU の制限: プロセッサ パフォーマンスの向上モード (Processor Performance Boost Mode)

Turbo Boost の有効/無効に関する設定項目。「アグレッシブ」と「無効」で切り替えたら良い。やらなくても良いと思うが、ファンがうるさかったので。「無効」にしたら基本的には E コアで動作しているような振る舞いに見える。ChatGPT によるとこの状態でも必要に応じて P コアが使われるそうだが、本当のところはよく分からない。

`HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\be337238-0d82-4146-a960-4f3749d470c7` において Attributes (REG_DWORD) を `1` → `2`

参考: [Quickly turn turbo boost on or off in Windows](https://notebooktalk.net/topic/464-quickly-turn-turbo-boost-on-or-off-in-windows/)

### 電源プランの複製

`powercfg /list` で現在の電源プランを取得して、`powercfg /duplicatescheme scheme_GUID` で複製できる。複製後のプランで「プロセッサ パフォーマンスの向上モード」を「無効」にすれば、簡単に Turbo Boost の有効/無効が切り替えられるようになるはず。

参考: [Powercfg のコマンドライン オプション](https://learn.microsoft.com/ja-jp/windows-hardware/design/device-experiences/powercfg-command-line-options#option_duplicatescheme)

# GPU の制限

[MSI Afterburner](https://jp.msi.com/Landing/afterburner/graphics-cards) を使えば GPU にも制限をかけられるらしい。

参考: [夏のゲーミングPCは熱すぎる！「MSI Afterburner」でビデオカードを省電力・低発熱に](https://forest.watch.impress.co.jp/docs/serial/sspcgame/1432583.html)

# WSL2 のセットアップ

## 有効化

コントロールパネル - プログラムと機能 - Windows の機能の有効化または無効化 から「Linux 用 Windows サブシステム」と「仮想マシンプラットフォーム」をチェック。

Hyper-V を使うだけなら Windows 11 Home で問題ない。c.f. [WSL2とHyper-Vの関係](https://qiita.com/matarillo/items/ca1eecf8f9a3cd76f9ce)

参考:

[WSL を使用して Windows に Linux をインストールする](https://learn.microsoft.com/ja-jp/windows/wsl/install)

## Ubuntu をインストール

```sh
wsl --install -d Ubuntu-20.04
```

Ubuntu と Ubunt-20.04 の違いは “latest” かどうかの違いらしい。c.f. [ストアにある3つの「Ubuntu」の違いは？ ～WLSの「Ubuntu」をアップグレードする方法](https://forest.watch.impress.co.jp/docs/serial/yajiuma/1134055.html)

### 新しい kernel component のインストール

以下のようなメッセージが出ることがある:

```
Installing, this may take a few minutes...
WslRegisterDistribution failed with error: 0x800701bc
Error: 0x800701bc WSL 2 ???????????? ??????????????????????? https://aka.ms/wsl2kernel ?????????
```

メッセージにあるように `https://aka.ms/wsl2kernel` 或は [手順 4 - Linux カーネル更新プログラム パッケージをダウンロードする](https://learn.microsoft.com/ja-jp/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package) から「x64 マシン用 WSL2 Linux カーネル更新プログラム パッケージ」をダウンロードしてインストール。これをインストールしてスタートメニューから「Ubuntu 20.04 on Windows」を起動したら先へ進めた。

参考: [WslRegisterDistribution failed with error: 0x800701bc](https://qiita.com/hali/items/bf04a1e4012025a38d6b)

## WSL コンソールにおけるコピペ可能化

ちゃんと確認していないが、必要ならこの対応をすると楽かもしれない。

参考: [Linux/WSLコンソールがショートカットキーによるコピペに対応 ～Windows 10 RS5](https://forest.watch.impress.co.jp/docs/news/1117273.html)

## DNS の設定

`/etc/wsl.conf` と `/etc/resolv.conf` をいじる。

[/etc/wsl.conf]
```ini
[network]
generateResolvConf = false
```

[/etc/resolv.conf]
```ini
nameserver 8.8.8.8
```

Windows を再起動すると `/etc/resolv.conf` が削除されるので以下の対応も行う:

```sh
sudo chattr +i /etc/resolv.conf
```

参考:

- [WSL2 で dns の名前解決ができなくなって ネット接続できなくなった場合の対処方法](https://qiita.com/kkato233/items/1fc71bde5a6d94f1b982)
- [WSL2でresolv.confが消える問題の解決方法](https://zenn.dev/frog/articles/9ae2428be2825a)

## ssh-agent の設定

`~/.bashrc` に以下を追加:

[~/.bashrc]
```sh
eval `ssh-agent` > /dev/null
ssh-add  /home/xxx/.ssh/id_rsa >& /dev/null
 ```

## .ssh を Windows と共有

まず、`/etc/wsl.conf` に以下を追加:

```ini
[automount]
options = "metadata,umask=077,fmask=11"
```

その上で以下で、`.ssh` を共有:

```sh
ln -s /mnt/c/Users/xxx/.ssh ~/.ssh
```

上記の `/etc/wsl.conf` の設定によって、NTFS 下にある DrvFS のディレクトリ `.ssh` 下においても、`chmod 600 id_rsa` のような permission の設定が、WSL メタデータに書き込まれるようになる。

参考:

- [WSL での詳細設定の構成](https://learn.microsoft.com/ja-jp/windows/wsl/wsl-config)
- [WSL のファイルのアクセス許可](https://learn.microsoft.com/ja-jp/windows/wsl/file-permissions)
- [[小ネタ]Windows環境のcredentialファイルとconfigファイルをWSL環境と共有してみた](https://dev.classmethod.jp/articles/windows-wsl-share-files/)
- [windows10のssh鍵を使って、wsl2からgithubへssh -T](https://zenn.dev/keijiek/scraps/b03e1804d15f99)

## ディスクサイズ

色々作業をした結果、WSL2 のイメージが入った VHD が肥大化しているかもしれないので、`optimize-vhd` コマンドで最適化したほうが良いかもしれない。

参考: [WSL2 のディスクサイズを削減する](https://qiita.com/TsuyoshiUshio@github/items/7a745582bbcd35062430)

## 大量のメモリ消費への対応

現在は一定の制限がかかるようになっているようだが、必要なら以下のような対応を行う。

[%USERPROFILE%\.wslconfig]
```ini
[wsl2]
memory=6GB
swap=0
```

「PC搭載メモリの50%または8GBのうち、少ない方の値」ということになっているらしいが、実際にはなっていないらしく、`.wslconfig` で制限を設定しない場合には「PC搭載メモリの50%」が上限のようである。つまり、32GB のメモリを搭載している場合には、16GB まで消費される可能性がある。c.f. [WSL2 using more than "50% of total memory on Windows or 8GB, whichever is less" (my machine has 24GB)](https://github.com/microsoft/WSL/issues/9636)

参考: [WSL2によるホストのメモリ枯渇を防ぐための暫定対処](https://qiita.com/yoichiwo7/items/e3e13b6fe2f32c4c6120)

# Docker のセットアップ

基本的には普通に公式の手順に従って [Install using the repository](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) を実行。

以下はセキュリティ意識と好みで:

```sh
sudo groupadd docker
sudo usermod -aG docker $USER
```

[/etc/sudoers]
```sh
xxx ALL=NOPASSWD: /usr/sbin/service docker start, /usr/sbin/service docker stop, /usr/sbin/service docker restart
```

c.f. [sudo のパスワードを聞かれないようにする](https://qiita.com/inoko/items/09b9a15cb1a5c83fed34)

`sudo` でパスワードを訊かれないようにすることで、docker の daemon の起動が楽になる (WSL2 では `systemd` 周りが難しいためこの方法のほうが安定しているらしい):

[~/.bashrc]
```sh
if service docker status 2>&1 | grep -q "is not running"; then
  sudo service docker start
fi
```

参考:

- [Windows で Docker Desktop を使わない Docker 環境を構築する (WSL2)](https://blog.jp.square-enix.com/iteng-blog/posts/00024-wsl2-docker-ce/)
- [出荷状態の windows から wsl2 + Ubuntu + docker-ce の環境を整う](https://core-tech.jp/blog/tech_log/3655/#docker-ce)
- [DockerDesktopからWSL2上のみで動くDockerに移行する](https://zenn.dev/taiga533/articles/11f1b21ef4a5ff)

# CUDA のセットアップ

WSL2 に CUDA 11.8 を導入する場合、以下のようにすれば良いはず:

[CUDA Toolkit 11.8 Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive) からダウンロードする時に、Linux - x86_64 - WSL-Ubuntu を選んだ場合の URL を使っている。

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

この後、PowserShell を管理者権限で起動して以下を実行すると WSLg も有効になる。c.f. [Linux 用 Windows サブシステムで Linux GUI アプリを実行する](https://learn.microsoft.com/ja-jp/windows/wsl/tutorials/gui-apps)

```sh
wsl --update
wsl --shutdown
```

CUDA 対応アプリをビルドする時には以下のようにして使うことになる:

```sh
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

参考:

- [待ってました CUDA on WSL 2](https://qiita.com/ksasaki/items/ee864abd74f95fea1efa)

# まとめ

いきなり Windows マシンが壊れてまたもや再セットアップということにはならないと思うが、自動化しにくい部分ではあるのでまとめてみた。このような情報は賞味期限も短いだろうから、恐らく大して役には立たないが。
