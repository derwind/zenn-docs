---
title: "Dify + Llama 3 で遊んでみる (2) — Llama-3-ELYZA-JP-8B も使ってしまう"
emoji: "🤖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["LLM", "localllm", "Dify"]
published: true
---

# 目的

[Dify + Llama 3 で遊んでみる (1)](/derwind/articles/dwd-llm-dify01) から暫く経って、Dify が 0.6.13 に更新されていたので二番煎じで記事をまとめる。・・・というのは正直どうでも良くて、自分メモをまとめないとやり方がわからなくなるのでまとめるというだけ。

ついでにチャットですぐ英語で応答してくれて困っていたので、日本語に強いと噂の「Llama-3-ELYZA-JP-8B」に乗り換える。

# Dify 0.6.13 に更新する

Dify が [v0.6.13](https://github.com/langgenius/dify/releases/tag/0.6.13) になっていて、`git pull` したら結構差分が出たので不安になって調べた。

```sh
$ git clone https://github.com/langgenius/dify.git
$ git checkout 0.6.13
```

# 設定を見直す

[【 Dify 0.6.12 対応 】 n8n と Dify を VPS 上の Docker 環境で動かして連携させる。セキュリティや nginx 設定までのオマケつき](https://note.com/hi_noguchi/n/n27baed2357ea) を見ると 0.6.12 以降でちょっと仕様が変わった旨が記事の途中以降で書いてある。`docker-compose.yaml` の編集ではなく、`.env` というファイルで設定をオーバーライドする形になったそうだ。

## docker-compose.yaml

前回の [https://zenn.dev/derwind/articles/dwd-llm-dify01](/derwind/articles/dwd-llm-dify01) に引き続き、ローカルで `ollama` の Llama 3 を使いたいので、docker コンテナの中からホスト側のネットワークを参照できるようにしたい。よって、以下だけは `.env` ではなく `docker-compose.yaml` で引き続き設定した:

```yaml
$ git diff
diff --git a/docker/docker-compose.yaml b/docker/docker-compose.yaml
index 3d26ae2ad..883d06aee 100644
--- a/docker/docker-compose.yaml
+++ b/docker/docker-compose.yaml
@@ -177,6 +177,8 @@ services:
     networks:
       - ssrf_proxy_network
       - default
+    extra_hosts:
+      - "host.docker.internal:host-gateway"
```

## .env

まずは `.env.sample` をコピーするところから始まる。

```sh
$ cd dify/docker
$ cp .env.sample .env
```

そして `.env` を適当に書き換える。今回は欲張って SSL 暗号化通信も行ってみた:

```sh
$ diff -ub .env.example .env
--- .env.example        2024-07-10 00:51:44.335410700 +0900
+++ .env        2024-07-10 01:42:06.600874278 +0900
@@ -564,7 +564,7 @@
 # Environment Variables for Nginx reverse proxy
 # ------------------------------
 NGINX_SERVER_NAME=_
-NGINX_HTTPS_ENABLED=false
+NGINX_HTTPS_ENABLED=true
 # HTTP port
 NGINX_PORT=80
 # SSL settings are only applied when HTTPS_ENABLED is true
@@ -602,5 +602,5 @@
 # ------------------------------
 # Docker Compose Service Expose Host Port Configurations
 # ------------------------------
-EXPOSE_NGINX_PORT=80
-EXPOSE_NGINX_SSL_PORT=443
+EXPOSE_NGINX_PORT=8102
+EXPOSE_NGINX_SSL_PORT=8103
```

## SSL 暗号化通信

```sh
$ cd nginx/ssl/
$ ls
dify.crt  dify.key
```

という形で秘密鍵とサーバ証明書らしきものを配置すれば良い。ちょっと難のあるものの場合、ブラウザで接続しにいくと警告されるが、その辺は自己責任となる。

以上が 0.6.13 対応である。折角なので、LLM のエンジンもアップデート (？) してみたい。

# Llama-3-ELYZA-JP-8B を使う

[「よーしパパ、Ollama で Llama-3-ELYZA-JP-8B 動かしちゃうぞー」](https://qiita.com/s3kzk/items/3cebb8d306fb46cabe9f) を参考にする。

## Llama 3 を破棄する

[Dify + Llama 3 で遊んでみる (1)](/derwind/articles/dwd-llm-dify01) で Llama 3 を導入したが、これを破棄して入れ替える。

ollama を念のため更新:

```sh
$ docker pull ollama/ollama:latest
```

一旦古い Llama 3 を破棄するために現在のボリュームを破棄:

```sh
$ docker volume rm ollama
```

これで Llama 3 が入っていた 5GB くらいのボリュームが消える。

## Llama-3-ELYZA-JP-8B をセットアップする

Hugging Face の [elyza/Llama-3-ELYZA-JP-8B-GGUF](https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/tree/main) からダウンロードする。

```sh
$ mkdir Llama-3-ELYZA
$ cd Llama-3-ELYZA
$ curl -LO https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/resolve/main/Llama-3-ELYZA-JP-8B-q4_k_m.gguf
```

そして `Modelfile` を用意する。`ollama show` で取得できるらしいが、[「よーしパパ、Ollama で Llama-3-ELYZA-JP-8B 動かしちゃうぞー」](https://qiita.com/s3kzk/items/3cebb8d306fb46cabe9f) そのままでいく:

[Modelfile]
```
FROM ./Llama-3-ELYZA-JP-8B-q4_k_m.gguf
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
```

**ollama のコンテナをデタッチトモードで起動する**

`./Llama-3-ELYZA-JP-8B-q4_k_m.gguf` を取り込みたいのでカレントディレクトリを `/work` にマウントする形でコンテナを起動する:

```sh
$ docker run -d --rm --gpus=all -v $PWD:/work -v ollama:/root/.ollama \
             --name ollama ollama/ollama
```

**ollama のコンテナに入って Llama-3-ELYZA-JP-8B をセットアップする**

恐らく以下のような感じで良いはず:

```sh
$ docker exec -it ollama bash
root@bd8c5bc7fa9e:/# cd /work/
root@bd8c5bc7fa9e:/work# ls
Llama-3-ELYZA-JP-8B-q4_k_m.gguf  Modelfile
root@bd8c5bc7fa9e:/work# ollama create elyza:jp8b -f Modelfile
transferring model data
using existing layer sha256:91553c45080b11d95be21bb67961c9a5d2ed7556275423efaaad6df54ba9beae
creating new layer sha256:8ab4849b038cf0abc5b1c9b8ee1443dca6b93a045c2272180d985126eb40bf6f
creating new layer sha256:c0aac7c7f00d8a81a8ef397cd78664957fbe0e09f87b08bc7afa8d627a8da87f
creating new layer sha256:bc526ae2132e2fc5e7ab4eef535720ce895c7a47429782231a33f62b0fa4401f
writing manifest
success
root@bd8c5bc7fa9e:/work# exit
$ docker container stop ollama
$ cd ..
$ rm -rf Llama-3-ELYZA
```

最後にダウンロードしたモデルはもう不要なので（docker のボリュームに転送されたので）、これを破棄した。作業中はダウンロードしたモデル + docker のボリュームで、一時的に 10GB くらいストレージを消費していたはずだ。

**動作確認をする**

```sh
$ docker run -d --rm --gpus=all -v ollama:/root/.ollama -p 11434:11434 \
             --name ollama ollama/ollama
```

上記でコンテナをデタッチトモードで起動して、先ほどセットアップした `elyza:jp8b` を起動する:

```sh
$ docker exec -it ollama ollama run elyza:jp8b
>>> こんにちは～
こんにちは！お元気ですか？

>>> 今何時ですかね？
私はAIなので、現在の時間を把握することはできません。私が把握できるのは、会話中の情報や過去のデータまでです。

>>> /bye
```

うまくいったようである。

## Dify から Llama-3-ELYZA-JP-8B を使う

Dify 側の準備は既に済ませているので以下のようにして起動してブラウザからアクセスする。

```sh
$ cd dify/docker
$ docker compose up -d
```

**モデルプロバイダーの設定**

ユーザーアイコンをクリックすると「設定」があるので、そこから LLM の設定をする。既に「llama3」は消してしまったので、古い情報は削除して「elyza:jp8b」を設定する。

```
Model Name: elyza:jp8b
Base URL: http://host.docker.internal:11434
Completion mode: Chat
Model context size: 4096
Upper bound for max tokens: 4096
Vision support: No
```

![](/images/dwd-llm-dify02/001.png)

これで準備は完了である。

**アプリケーションからの設定**

中央上部の「スタジオ」から「チャットボット」を選んで「最初から作成」などを選択する。LLM モデルに早速「elyza:jp8b」を設定して適当なインストラクションでチャットアプリを準備する。

![](/images/dwd-llm-dify02/002.png)

最早わざわざ「日本語で」などとお願いせずとも、日本語で応答してくれていることが分かるだろう。

**終了処理**

```sh
$ docker compose down
```

で Dify のコンテナオーケストレーション（？）を終了させて、

```sh
$ docker container stop ollama
```

で ollama のコンテナを終了させれば良い。

# まとめ

やってみると案外簡単にアップデートと LLM の乗り換えができた上に、日本語のチャットが大変捗った。
