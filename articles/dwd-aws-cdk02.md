---
title: "LocalStack と CDK で遊んでみる (2)"
emoji: "🌨"
type: "tech"
topics: ["AWS", "ポエム", "TypeScript"]
published: false
---

# 目的

[前回](/articles/dwd-aws-cdk01)に引き続き、LocalStack と CDK で遊んでみる。

> - Step Functions + 複数個の Lambda

を題材に記事をまとめてみることにする。ゴールとしては、Fibonacci 数列の最初の 5 項を 2 乗してそれらの和をとった値 ― つまり、40 ― を求める過程を実装するところまでである。

# 参考資料

以下の資料に大いにお世話になった。

- [実践 AWS CDK - TypeScript でインフラもアプリも！](https://booth.pm/ja/items/1881928)
    - CDKv1 で書かれた本であるが、適宜読み替えることで参考になる。Step Functions を使ったスタックについてお世話になった。
- [[Step Functions]動的並列処理（Map）を使って60分×24時間=1440ファイルのバッチ処理を楽々実装してみた](https://dev.classmethod.jp/articles/step-functions-serverless-map-every-minute/)
    - AWS 界隈で有名なクラスメソッドさんの記事。Map ステートでお世話になった。
- [AWS Step Functions の Map ステートの挙動を調べてみた。](https://dev.classmethod.jp/articles/step-functions-map-behavior/)
    - 同じくクラスメソッドさんの記事。Map ステートでお世話になった。
- [実践！AWS CDK #1 導入](https://dev.classmethod.jp/articles/cdk-practice-1-introduction/)
    - クラスメソッドさんの連載物の CDK の記事群。途中までは CDKv1 向けだが、CDKv2 リリース以降は CDKv2 で再実装されている。

# この記事の存在意義は？

上記の参考文献で完結しているのだが、一応 CDKv2 + Step Functions の組み合わせの記事はなかったように思っていて、更に LocalStack 上で実行するサンプルは載っていないと思うので本記事に多少の存在意義はあるのではないかと思っている。ただ、やはり何番煎じかよく分からないが・・・。

# やること

いわゆる富豪的プログラミングを実施する。

- Fibonacci 数列を生成する Lambda を用意する
- Fibonacci 数列の各項を 2 乗する Lambda を用意する
- その結果を合算する Lambda を用意する
- これらの Lambda を統括する Step Functions を用意する

## Fibonacci 数列とは？

$a_1=1, a_2=1, a_3=2, a_4=3, a_5=5$ の数列 $\{a_n\}_{n=1}^\infty$ のこと。項はこれ以上使わないのでこれ以上は定義しない。
今回は、$a_1^2+a_2^2+a_3^2+a_4^2+a_5^2=1^2+1^2+2^2+3^2+5^2=40$ を計算させたい。

## ディレクトリツリー

以下のような構成にした。

```sh
$ tree
.
├── README.md
├── bin
│   └── step_functions_test.ts
├── cdk.json
├── jest.config.js
├── lambda
│   ├── default_values
│   │   └── index.py
│   ├── fibonacci
│   │   └── index.py
│   ├── pow
│   │   └── index.py
│   └── sum
│       └── index.py
└── lib
    └── step_functions_test-stack.ts
```

# ステートマシン

以下のようにした。途中、Map ステートを使って分散処理を行う形である。

![](/images/dwd-aws-cdk02/001.png)

# 実装

## Lambda

- default_values

Fibonacci 数列の項の数と指数のデフォルト値を設定する関数とした。

```python
default_values = {
    'n_terms': 5,
    'exponent': 1
}

def lambda_handler(event, context):
    for k, v in default_values.items():
        if k not in event:
            event[k] = v

    event['statusCode'] = 200

    return event
```

- fibonacci

Fibonacci 数列を与えられた項の数だけぶんのリストとして返す関数とした。

```python
def fibonacci(n_terms):
    terms = [1, 1]

    if n_terms < 2:
        return terms[:n_terms]

    for _ in range(n_terms - 2):
        num = terms[-2] + terms[-1]
        terms.append(num)
    return terms

def lambda_handler(event, context):
    n_terms = event['n_terms']

    event['fibonacci'] = fibonacci(n_terms)
    event['statusCode'] = 200

    return event
```

- pow

`base` の値を `exponent` 乗する関数とした。

```python
def pow(b, n):
    return b ** n

def lambda_handler(event, context):
    base = event['base']
    exponent = event['exponent']

    event['value'] = pow(base, exponent)
    event['statusCode'] = 200

    return event
```

- sum

複数のイベントを受け取って、それぞれのイベントに設定された値を合算する関数とした。

```python
def lambda_handler(event, context):
    total = 0
    for eve in event:
        total += eve['value']

    event = {}
    event['value'] = total
    event['statusCode'] = 200

    return event
```

## Step Functions

Step Functions を CDK を用いて以下のように実装した。JSON ベースの実装から解放されて気分が良い。
Map ステートでタスクを分散する部分と、分散処理が終わった後の結果をかき集めるところの実装がややこしいのだが、とりあえず以下のような形で動いた。[^1]

[^1]: 理屈で考えているというよりは、動くパターンを暗記してしまって “構文” のように使っている。

```typescript
import {
  Stack,
  StackProps,
  Duration,
  aws_lambda as lambda,
  aws_stepfunctions as sfn,
  aws_stepfunctions_tasks as tasks
} from 'aws-cdk-lib';
import { Construct } from 'constructs';

export class StepFunctionsTestStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    this.newStateMachine();
  }

  private newStateMachine(): sfn.StateMachine {
    const defaultValuesLambda = new lambda.Function(this, 'Default values', {
      functionName: 'DefaultValues',
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.lambda_handler',
      code: lambda.Code.fromAsset('./lambda/default_values')
    });

    const fibonacciLambda = new lambda.Function(this, 'Fibonacci', {
      functionName: 'Fibonacci',
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.lambda_handler',
      code: lambda.Code.fromAsset('./lambda/fibonacci')
    });

    const powLambda = new lambda.Function(this, 'Pow', {
      functionName: 'Pow',
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.lambda_handler',
      code: lambda.Code.fromAsset('./lambda/pow')
    });

    const sumLambda = new lambda.Function(this, 'Sum', {
      functionName: 'Sum',
      runtime: lambda.Runtime.PYTHON_3_9,
      handler: 'index.lambda_handler',
      code: lambda.Code.fromAsset('./lambda/sum')
    });

    const submitJob = new tasks.LambdaInvoke(this, 'Submit Job', {
      lambdaFunction: defaultValuesLambda,
      outputPath: '$.Payload',
    });

    const fibonacciJob = new tasks.LambdaInvoke(this, 'Fibonacci Job', {
      lambdaFunction: fibonacciLambda,
      outputPath: '$.Payload',
    });

    const map = new sfn.Map(this, 'Map State', {
      itemsPath: sfn.JsonPath.stringAt('$.fibonacci'),
      parameters: {
        'base.$': '$$.Map.Item.Value',
        'exponent.$': '$.exponent',
      },
    });

    const powJob = new tasks.LambdaInvoke(this, 'Power Job', {
      lambdaFunction: powLambda,
      inputPath: '$',
      outputPath: '$.Payload',
    });

    map.iterator(powJob);

    const sumJob = new tasks.LambdaInvoke(this, 'Sum Job', {
      lambdaFunction: sumLambda,
      inputPath: '$',
      outputPath: '$.Payload',
    });

    const definition = submitJob
      .next(fibonacciJob)
      .next(map)
      .next(sumJob);

    const stateMachine = new sfn.StateMachine(this, 'FibonacciStateMachine', {
      stateMachineName: 'FibonacciStateMachine',
      definition: definition
    });

    return stateMachine;
  }
}
```

# デプロイ

[bootstrap](/articles/dwd-aws-cdk01#bootstrap) が完了している前提のもとでは、普通に

```sh
$ cdklocal deploy
```

でスタックをデプロイできる。

# 動作確認

Step Functions を起動しても動作完了に暫くかかるので（試したら 5 秒くらいかかった）、ポーリングをするなどして結果を取り出すしかない。あまり美しくはないが、[Boto3](https://aws.amazon.com/jp/sdk-for-python/) を用いた以下のようなコードで動作確認ができる。

```python
import boto3
import json
import time

input = {
    'n_terms': 5,
    'exponent': 2
}

client = boto3.client('stepfunctions', endpoint_url='http://localhost:4566/')
response = client.start_execution(input=json.dumps(input), stateMachineArn='arn:aws:states:us-east-1:000000000000:stateMachine:FibonacciStateMachine')
max_try = 10
for _ in range(max_try):
    response = client.describe_execution(executionArn=response['executionArn'])
    if response['status'] != 'RUNNING':
        output = json.loads(response['output'])
        value = output['value']
        print(value)
        break
    else:
        time.sleep(1)
```

# まとめ

CDK を使って Step Functions を実装してみた。多数の Lambda と Step Functions の管理のために AWS マネジメントコンソールの中を行ったり来たりしなくて済むようになったので嬉しい。

開発も [Visual Studio Code](https://azure.microsoft.com/ja-jp/products/visual-studio-code/) だけで完結するし、[AWS Toolkit for Visual Studio Code](https://aws.amazon.com/jp/visualstudiocode/) を導入することで、`cdklocal deploy` 或は `cdklocal synth` 時に生成される CloudFormation のテンプレートから Step Functions のステートマシン図がプレビューできるので嬉しい。

なお、今回のお題の通りだと何も嬉しくない富豪的プログラミングに過ぎないが、例えば機械学習用の訓練データを大量に作成するための用途で使える。実際にその目的でこのスタックを開発した。[^2]

[^2]: より厳密には、お手製の Lambda と Step Functions を先行して書いたものの汚らしく散らかってしまったので、後から CDK で IaC 化した形である・・・。
