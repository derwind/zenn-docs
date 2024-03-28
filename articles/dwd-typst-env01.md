---
title: "typst の環境を WSL 上に作る (1)"
emoji: "🛠"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Windows", "WSL2", "typst"]
published: true
---

# 目的

1 回で終わりかもしれないが、[typst](https://typst.app/) の環境を WSL2 上に作ったメモを残しておく。

[github.com/typst/typst](https://github.com/typst/typst) の記述をそのまま DeepL に突っ込むと、以下のようになる。ざっくりとはポスト LaTeX といったところだろうか。

> Typstは新しいマークアップベースの組版システムであり、LaTeXと同じくらい強力でありながら、より簡単に学び、使えるように設計されている。Typstには以下がある:
> 
> - 最も一般的な組版作業のためのビルトインマークアップ
> - それ以外のすべてに対応する柔軟な機能
> - 緊密に統合されたスクリプトシステム
> - 数式組版、書誌管理など
> - インクリメンタル・コンパイルによる高速なコンパイル時間
> - 何か問題が発生したときのための親切なエラーメッセージ

とりあえずこれを使えるようにしてみたい。

# やること

ざっくり以下がやりたいことになる:

1. typst はインストールが面倒くさいので docker を使いたい
1. VS Code で書きたい
1. ホットリロードできる pdf ビューアを用意したい

これを踏まえ、以下のような構成にすることにした。

![](/images/dwd-typst-env01/001.png)

# 1. typst 環境構築

ローカル環境を汚して (？) セットアップを試行錯誤したくないのでプレビルドの docker イメージを使う。

```sh
docker pull ghcr.io/typst/typst:latest
```

で取得できる。

## 日本語フォント

```sh
$ docker run ghcr.io/typst/typst typst fonts
```

> DejaVu Sans Mono
> Linux Libertine
> New Computer Modern
> New Computer Modern Math

から分かるように、日本語フォントがデフォルトでは使えない。よって、ローカルにフォント用のディレクトリを作ってそこを参照させるようにする。今回は「fonts」ディレクトリを作って参照させる。つまり、[TeX Live](https://github.com/TeX-Live) にも 2020 以降入っている [原ノ味フォント](https://github.com/trueroad/HaranoAjiFonts) をダウンロードして

```sh
$ ls -1 fonts/
HaranoAjiGothic-Bold.otf
HaranoAjiGothic-Regular.otf
HaranoAjiMincho-Bold.otf
HaranoAjiMincho-Regular.otf
```

のように配置していると仮定して、

```sh
$ docker run -it --rm -v $(pwd):/root \
> -e TYPST_FONT_PATHS=/root/fonts ghcr.io/typst/typst typst SUBCOMMAND
```

のような使い方をする。例えば `fonts` サブコマンドを実行すると

```sh
$ docker run -it --rm -v $(pwd):/root \
> -e TYPST_FONT_PATHS=/root/fonts ghcr.io/typst/typst typst fonts
```

> DejaVu Sans Mono
> **Harano Aji Gothic**
> **Harano Aji Mincho**
> Linux Libertine
> New Computer Modern
> New Computer Modern Math

となるので、これで日本語対応 typst 環境の構築が完了したことになる。

# 2. VS Code を使う

これは特にやることがないのだが、VS Code で開いているファイルをリアルタイムで `typst` にコンパイルさせたいので、

```sh
$ code -n document.typ
```

しているなら、`typst watch` も以下のように合わせる。これでリアルタイムにコンパイルが走って「document.pdf」が作られる。

```sh
$ docker run -it --rm -v $(pwd):/root \
> -e TYPST_FONT_PATHS=/root/fonts ghcr.io/typst/typst typst watch document.typ
```

# 3. ホットリロードできる pdf ビューアを用意したい

Adobe Acrobat Reader では pdf の更新に対するリロードがかからない。macOS でいう「プレビュー」のようなものが欲しい。どうやらこれについては [Sumatra PDF](https://www.sumatrapdfreader.org/free-pdf-reader) が該当するらしい。ということで以上で本記事は完了である。

ところが個人的には、このためだけに専用のビューアを追加でインストールしたくなかったので別解を模索することにした。

## npm reload

ローカルサーバを立てて、Chrome をビューアにすれば良いのではないかと考えたのでこれを実現する。このためにはホットリロードに対応したサーバが必要であるが、[npm reload](https://www.npmjs.com/package/reload) が利用できる。

```sh
$ npm i reload --save-dev
```

結局これくらいはインストールする羽目になるが、npm 関連はセーフということにする。`reload` はデフォルトで `html`, `js`, `css` の更新を監視するが、今回 `pdf` も監視させたいので、package.json に記述を追加する。

```json
  ...
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "start": "reload -e 'html|js|css|pdf'"
  },
  ...
```

そしてサーバを起動する。

```sh
$ npm start
```

これでホットリロードに対応したサーバの準備ができた。

## pdf をブラウザで開く

実はこの状態で `http://localhost:8080/document.pdf` を開くと確かに pdf は表示されるが、pdf の再コンパイルがかかっても表示は更新されない。

`reload` はホスティングしている html に `reload.js` という JavaScript ファイルの読み込み処理を追加してから配信する動作をする。`reload.js` 内では WebSocket を用いて、ファイルの更新に応じてブラウザ側にページ再読み込みのためのイベントを送信する動作をしている。

このため、ブラウザで直接 pdf を開いてしまうと `reload.js` をロードして WebSocker が張られる処理が発生しない。よって、`document.pdf` を html でラップすることを考える。実はこれはとても簡単で以下で良い。

[document.html]
```html
<html>
  <embed src="document.pdf" type="application/pdf" width="100%" height="100%">
</html>
```

これをブラウザで開くと、実際には以下の内容で読み込まれる。

```html
<html>
  <embed src="document.pdf" type="application/pdf" width="100%" height="100%">
</html>


<!-- Inserted by Reload -->
<script src="/reload/reload.js"></script>
<!-- End Reload -->
```

これで漸くホットリロードに対応したビューアが手に入ったことになる。

# typst で書いてみる

公式サイトなどで記法についてチェックしながら備忘録メモのようなものを書く。

```tex
#set text(
    lang:"ja",
    font: "Harano Aji Mincho",
    11pt,
)
#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)

= Euler-Lagrange 方程式と Hamilton の正準方程式
\
$q$ を *一般化座標*、$dot(q)$ を *一般化速度* として、$L = L(q, dot(q))$ を Lagrangian とする。
\ \
$S = integral_0^1 L(q, dot(q)) d t$ なる作用積分を考える。$delta q(t)$ なる微小のパスを考え、$delta q(0) = delta q(1) = 0$ として、パス $q + delta q$ を考える。
```

こんな感じのものを書いてブラウザで `http://localhost:8080/document.html` を開くと以下のように表示される。勿論ホットリロードもされる。

![](/images/dwd-typst-env01/002.png)

# docker 周りの制御

`typst watch` を使う場合、docker コンテナを起動したコンソールで、理由はよく分からないが Ctrl+C がうまく受け付けてくれないように思える。よって、

```sh
$ docker run -it --rm --name typst-watch -v $(pwd):/root \
> -e TYPST_FONT_PATHS=/root/fonts ghcr.io/typst/typst typst watch document.typ
```

のように分かりやすいコンテナ名をつけて `typst watch` を起動しておき、終了したい場合には

```sh
$ docker stop `docker ps -q --filter "name=typst-watch"`
```

のようにして、コンテナ ID を解決してコンテナを終了すると良さそうである。終了には数秒かかる。

# まとめ

過剰とも言える処理を実行したことになるが、絶対に `Rust` でコンパイルなんかしないぞ、`winget` なんてわけの分からないものも使わないぞという気持ちで、心理的にハードルの低い Web 系ツールだけで済ませることにした。これで docker と npm だけで環境構築をして快適な (？) typst 環境が構築できることが分かった。

ビューアについては普通は [Sumatra PDF](https://www.sumatrapdfreader.org/free-pdf-reader) を使うのが大正解としか言えない。この場合、npm reload も不要となる。docker だけで済んでミニマル感があって更に良い。ただ今回は「**何か嫌だった**」という理由で npm reload を使っただけである。
