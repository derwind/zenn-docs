---
title: "LocalStack と CDK で遊んでみる (1)"
emoji: "🌨"
type: "tech"
topics: ["AWS", "ポエム", "TypeScript"]
published: true
---

# 目的

[CDK](https://aws.amazon.com/jp/cdk/)[^1] で作成した CloudFormation のスタックを [LocalStack](https://github.com/localstack/localstack)[^2] にデプロイして眺めた備忘録を残すこと。

[^1]: CloudFormation のスタックを JSON ではなく TypeScript で記述できる素敵ツール。
[^2]: AWS のモック環境。コミュニティ版は色々制約はあるが簡単なインフラの構築の模擬試験として活用できる。

本記事のスコープとしては、CDK の bootstrap を済ませ、スタックの開発に入っていける準備が済むところまでとする。

# 背景

普段仕事とかでちょこっと AWS を使うのだが、お手製の真心のこもった Lambda を沢山作ると管理が煩雑になるし、正直何を使っているのかすぐ分からなくなる。特に Step Functions を使い出すと、あまり好きではない JSON が絡んでくるし、マネジメントコンソールを右往左往する羽目になる。

つらくて仕方ないので一元管理したいと思い、IaC[^3] に手を出した。しかし、CloudFormation はまたまた JSON でつらく、[Terraform](https://www.terraform.io/) は DSL を覚えないとならないので潰しが効かない気がする（インフラが専門でないためというのもある）。Web は jQuery で時間が止まっているので、スキルのアップデートも一緒にするつもりで TypeScript を使って CDK に手を出した。

[^3]: Infrastructure as Code。環境構築もプログラムで記述して Git で管理しましょうというもの。

LocalStack を使う理由は無料だからというのも大きいが、ステージング/本番環境でうっかりやらかすとか気にしないで済むことや、直接 AWS にデプロイするよりは高速に処理が済むという点も大きい。

# 環境

- Docker イメージ

```sh
$ docker image ls | grep localstack
localstack/localstack       0.14.1              354d99d5680a        22 hours ago        1.5GB
```

- Node.js パッケージ

[aws-cdk-local](https://github.com/localstack/aws-cdk-local) を参考に

```sh
$ npm install -g aws-cdk-local aws-cdk
```

し、CDK プロジェクト下で

```sh
$ cdklocal init --language typescript
```

した結果、以下のようになっている:

```sh
$ npm list
├── aws-cdk-lib@2.16.0
├── aws-cdk@2.16.0
└── typescript@3.9.10
```

- AWS CLI version 2

[Installing past releases of the AWS CLI version 2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-version.html) で OS に合わせた AWS CLI version 2[^4] を使っていることを前提としている。

[^4]: `awslocal` というコマンドもあるが、AWS CLI version 1 ベースでつまらないので使わないことにした。

※ LocalStack はバージョンによって悩まされることがあったが、今のところ 0.14.1 ではやりたいことが実現できている。

# LocalStack

以下のようにして Docker イメージを取得できる:

```sh
$ docker pull localstack/localstack:0.14.1
```

docker-compose 用に [docker-compose.yml](https://github.com/localstack/localstack/blob/master/docker-compose.yml) を使うと以下のように LocalStack の起動と終了が楽になる。

- LocalStack 起動

```sh
$ docker-compose up -d
```

- LocalStack 終了

```sh
$ docker-compose down
```

## 補足

ちょっと凝った？スタックを `deploy` した時に `destroy` でもエラーが出てしまい、ハマったことがあった。このような場合、素直に `down` して再度 `up -d` すれば解消した。cdk の操作で何か変だと思ったら、

```sh
$ docker-compose logs
```

で LocalStack のログを確認するとわりとヒントが隠れていることがあった。

# CDK

```sh
$ mkdir cdk-demo
$ cd cdk-demo
$ cdklocal init --language typescript
```

のようにしてプロジェクトを作成した後、bootstrap なる一種の “おまじない” のような処理を実行する必要がある。

## bootstrap

すごく簡素なスタックなら bootstrap をしなくても良いが大抵必要なので、`docker-compose up -d` したタイミングで実行する。現時点での理解では「スタックのデプロイに必要なファイルの中間置き場として S3 バケットを作成する」というものである。

```sh
$ cdklocal bootstrap aws://000000000000/us-east-1
```

実際の AWS 環境の場合、アカウント内の同一リージョンでは 1 回のみ bootstrap すれば良いが、LocalStack の場合、`docker-compose down` すると bootstrap の結果ごと揮発するので、`docker-compose up -d` ごとに毎回 bootstrap する必要がある。

以上で、スタック開発の準備は整った。

# まとめ

何番煎じか知らないが、LocalStack + CDK について準備をするところまでをまとめた。ちょっと前まで jQuery 止まりであったところから、Node.js + TypeScript に手を出して、駆け足で CDK に突入してみた結果、散々辛い目にあったので、要所をまとめておきたいというものである。

この構成で以下のスタック（群）のデプロイと実行を確認している。

- 1 つのスタック: 単発の Lambda
- 1 つのスタック: Step Functions + 複数個の Lambda
- 3 つのスタック: IAM, VPC, EC2

LocalStack のコミュニティ版が対応していない機能[^5]については適当に削らないとならないが、これくらいのものがざっくりと動作するのはなかなか嬉しい。

取捨選択して記事化して残せたらと思う。

[^5]: 一例として EC2 に関連してネットワーク ACL 機能が引っかかるように見えた。
