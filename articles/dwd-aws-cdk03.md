---
title: "LocalStack と CDK で遊んでみる (3) — Web アプリに挑戦"
emoji: "🌨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["AWS", "LocalStack", "awscdk", "poem", "TypeScript"]
published: true
---

# 目的

何となく [SAA-C03](https://aws.amazon.com/jp/certification/certified-solutions-architect-associate/) を受けてみたいが、本だけ読んで勉強するのはつらいので手を動かしたい。ので、少しそれっぽいことを CDK + LocalStack でやりたいなというもの。

ちょうど [AWS CDK+localstackを使ってよくあるRESTなWebアプリ構成を作ってみる](https://zenn.dev/okojomoeko/articles/f4458e1efc8f7a) という素晴らしい記事があったので、内容を読みつつ少し読み替えて実装してみたい。

[LocalStack と CDK で遊んでみる (2)](/derwind/articles/dwd-aws-cdk02) 以来の久しぶりの CDK ネタでもある。

# 開発環境

- Ubuntu 20.04
- Docker version 24.0.5, build ced0996
- npm 9.8.1
- node v18.17.1
- LocalStack 2.2.0
- cdklocal 2.93.0 (build 724bd01)
- aws-cli 2.13.13

# 実装

基本的に [AWS CDK+localstackを使ってよくあるRESTなWebアプリ構成を作ってみる](https://zenn.dev/okojomoeko/articles/f4458e1efc8f7a) のままだが、少し変更する。

- Lambda の Python: 3.9 → 3.10
- 外部 PC からの実行を想定して CORS の設定を Lambda に追加

ソースコード: https://github.com/derwind/cdk-study/tree/rest-web-app

## docker-compose.yml

[Accessing localstack from another computer](https://stackoverflow.com/questions/73778062/accessing-localstack-from-another-computer) を参考に、以下を変更:

```yaml
version: "3.8"

services:
  localstack:
    container_name: "${LOCALSTACK_DOCKER_NAME-localstack_main}"
    image: localstack/localstack:2.2.0
    ports:
      - "4566:4566"            # ここを変更
      ...
```

## web/package.json

[Viteで起動したローカル開発サーバーにIPアドレスで外部からアクセスする方法【Vue3/Typescript編】](https://zenn.dev/jump/articles/9b863cfcf72eb7) を参考に、以下を変更:

```json
{
  "name": "web",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite --host --port 8080", // ここを変更
    ...
```

## web/.env

スタックのデプロイ後のログを参考に `VITE_REST_API_ROOT_URL` をリモートサーバの URL に変更:

```
VITE_REST_API_ROOT_URL=http://xxx.xxx.xxx.xxx:4566/restapis/yyyyyyyyyy/prod/_user_request_/
```

## CORS (Cross-Origin Resource Sharing) の設定

毎回よく分かっていないので宜しくはないのだが[^1]、今回のパターンでは、各 Lambda の戻り値に CORS の設定を追加すれば良さそうであった。内容は雑なので、もっと適切に設定すべきではある:

[^1]: [@aws_cdk/aws-apigatewayv2-alpha](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-apigatewayv2-alpha-readme.html) を使う場合や、Lambda 関数 URL を使う場合でそれぞれちょっとずつ設定が違うのだが、本質は何なんだろう？

```python
def handler(event, context):
    ...
    return {
        "statusCode": 200,
        "body": json.dumps({"res": results, "event": event}),
        "headers": {  # CORS の設定
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,GET,PUT,POST,DELETE,PATCH,HEAD",
            "Access-Control-Allow-Headers": "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Amz-User-Agent'",
        },
    }
```

# 動作確認

## ~/.aws/credentials

```
[localstack]
aws_access_key_id = test
aws_secret_access_key = test
```

という設定にしておいて、以下を順に実行していく:

## LocalStack 起動

```sh
docker compose up -d
```

## Bootstrap

```sh
cd app
cdklocal bootstrap aws://000000000000/us-east-1
```

## スタックのデプロイ

```sh
cd app
cdklocal deploy
```

## Web アプリの起動

S3 に Web アプリをデプロイした場合、LocalStack を動かしている PC 内からは `CfnOutput` した URL へのアクセスでいけたのだが、外部 PC からは名前解決できそうになかったので諦めて dev サーバ上で確認することにした:

```sh
cd web
npm run dev
```

画面イメージは以下の通り:

![](/images/dwd-aws-cdk03/001.png)

# まとめ

一番手こずったのは CORS の設定の部分なのだが、何とかこの構成でも適用して別 PC からアクセスすることができた。

今回の内容に近いことは既に何回かやったことはあるのだが、実際の AWS でやっていて LocalStack では初めてだったので動くかどうか試してみたかったのがあった。

これで、[SAA-C03](https://aws.amazon.com/jp/certification/certified-solutions-architect-associate/) に向けての “無料の AWS 環境” が用意できたような気がする[^2]。

[^2]: どうせマネジメントコンソールではほとんど操作しないから、基本機能が動けば本物の AWS でも LocalStack でもどちらでも良いのである。

# Appendix

## BucketDeployment の件

参考にした [AWS CDK+localstackを使ってよくあるRESTなWebアプリ構成を作ってみる](https://zenn.dev/okojomoeko/articles/f4458e1efc8f7a) では `BucketDeployment` が使えないとあるが、仮に使うとどうなるか試した。結論として、スタックのデプロイは通って、S3 にバケットも作成されるのだが、当該バケットに下に何もオブジェクトがデプロイされていないという状況になった

```typescript
import {
  ...
  aws_s3_deployment as s3deploy
} from 'aws-cdk-lib';

export class AppStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    ...
    new s3deploy.BucketDeployment(this, "DeployWebsite", {
      sources: [s3deploy.Source.asset("./website-dist")],
      destinationBucket: websiteBucket,
      destinationKeyPrefix: "web/static",
    });
    ...
  }
}
```

# 雑多なデバッグ用の色々 (メモ)

## Lambda 一覧

```sh
aws --profile localstack lambda list-functions
```

## API Gateway 一覧

```sh
aws --profile localstack apigateway get-rest-apis
```

## CloudWatch Logs

### ロググループを確認

```sh
aws --profile localstack logs describe-log-groups
```

### ロググループ内のログストリームを確認

```sh
aws --profile localstack logs describe-log-streams --log-group-name "/aws/lambda/RegisterData"
```

### ロググループ内の指定ログストリームの内容を確認

わりとハマりどころとして、`[$LATEST]` と表示される部分を `[\$LATEST]` にしないとならない。

```sh
aws --profile localstack logs get-log-events --log-group-name "/aws/lambda/RegisterData" --log-stream-name "2023/08/29/[\$LATEST]7ae57eaea31e16f0f00b724e8639425a"
```
