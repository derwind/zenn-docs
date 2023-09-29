---
title: "なろう小説API というものを呼んでみる"
emoji: "📖"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "poem", "Web", "API"]
published: false
---

# 目的

[小説家になろう](https://syosetu.com/) というサイトがあって、ユーザーが投稿した小説を読めるらしい。たぶん [転生したらスライムだった件](https://ncode.syosetu.com/n6316bn/) が有名？

たまたま [なろう小説API](https://dev.syosetu.com/man/api/) というものを見つけたので、普段は叩かない Web API を叩いてみようという内容。

# 実装

必要なモジュールを import する:

```python
import io
import json
import gzip
import requests
```

## 欲しい情報を指定する

検索項目を絞る。今回は「小説名」「作者名」「小説のあらすじ」「ジャンル」をターゲットにしてみたい。また、大域負荷を下げるために、`of` フィールドでターゲットの情報だけをフィルタリングすることにしよう。これは API の解説の中の「ofパラメータ」に書いてある。

```python
# 欲しい情報
attrs = ["title", "writer", "story", "genre"]

# 項目から of パラメータへの対応付け
attr2of = {
    "title": "t",
    "ncode": "n",
    "userid": "u",
    "writer": "w",
    "story": "s",
    "biggenre": "bg",
    "genre": "g",
    "keyword": "k",
    "general_firstup": "gf",
    "general_lastup": "gl",
    "noveltype": "nt",
    "end": "e",
    "general_all_no": "ga",
    "length": "l",
    "time": "ti",
    "isstop": "i",
    "isr15": "ir",
    "isbl": "ibl",
    "isgl": "igl",
    "iszankoku": "izk",
    "istensei": "its",
    "istenni": "iti",
    "pc_or_k": "p",
    "global_point": "gp",
    "daily_point": "dp",
    "weekly_point": "wp",
    "monthly_point": "mp",
    "quarter_point": "qp",
    "yearly_point": "yp",
    "fav_novel_cnt": "f",
    "impression_cnt": "imp",
    "review_cnt": "r",
    "all_point": "a",
    "all_hyoka_cnt": "ah",
    "sasie_cnt": "sa",
    "kaiwaritu": "ka",
    "novelupdated_at": "nu",
    "updated_at": "ua",
}
```

## API をラップする

生の Web API だと叩きにくい気がするので、Python の関数としてラップする。あまりやる気のない実装なのだが以下のようにした。

`gzip` というパラメータがあって、

> 転送量上限を減らすためにも推奨

とあったので、これも使うことにした。使わない場合には `json_data[0]["allcount"]` などの部分を `res.json()[0]["allcount"]` などにすれば良い。

```python
def get_allcount_for_keyword(url, keyword):
    payload = {
        "gzip": 5,
        "out": "json",
        "lim": 1,
        "order": "new",
        "keyword": 1,
        "word": keyword,
    }
    res = requests.get(url, params=payload)
    if res.status_code != 200:
        return None
    gzip_file = io.BytesIO(res.content)
    with gzip.open(gzip_file, "rt") as f:
        json_data = f.read()
    json_data = json.loads(json_data)
    return json_data[0]["allcount"]


def get_info(url, keyword, allcount, lim_per_page, page_no):
    # 1 ページあたり `lim_per_page` 件表示として、`page_no` ページ目の情報を表示
    st = (page_no - 1) * lim_per_page + 1

    # `allcount` すれすれの場合 `lim_per_page` を上書きしたほうが良い？
    # 今回は実装をさぼる。

    payload = {
        "gzip": 5,
        "out": "json",
        "of": "-".join([attr2of[attr] for attr in attrs]),
        "lim": lim_per_page,
        "st": st,
        "order": "new",
        "keyword": 1,
        "word": keyword,
    }
    res = requests.get(url, params=payload)
    if res.status_code != 200:
        return []
    gzip_file = io.BytesIO(res.content)
    with gzip.open(gzip_file, "rt") as f:
        json_data = f.read()
    json_data = json.loads(json_data)
    return json_data[1:]
```

# 検索してみる

どうやら「聖女」といったキーワードがたぶん人気のキーワードなのでこれで検索してみよう。

具体的な出力は割愛するとして以下のような感じで関数を呼べば小説のサマリが表示された。

```python
url = "https://api.syosetu.com/novelapi/api/"

keyword = "聖女"

allcount = get_allcount_for_keyword(url, keyword)
lim_per_page = 20
page_no = 1

for i, info in enumerate(get_info(url, keyword, allcount, lim_per_page, page_no)):
    print(f"[{(page_no - 1) * lim_per_page + 1 + i}]")
    for attr in attrs:
        value = info[attr]
        print(f"{attr}: {value}")
    print("-"*10)
```

# まとめ

[なろう小説APIサンプルプログラム「ページ分割対応小説簡易一覧(PHP)」](https://dev.syosetu.com/man/sample02/) を見ても PHP のことは分からないので、雰囲気で Python にしてみたが何とか動作した。

サイト固有の情報とか名詞が多くて正直よく分かっていないのだが、スクレイピングすることで何か情報が拾えるかもしれない。この辺は [なろう小説APIを試してみた](https://qiita.com/sola_wing529/items/bad361cdd0d11373b7be) で簡単な分析がなされているので、色々試してみても良いのかもしれない。

> IPアドレスごとに利用制限を行います。
> 1日の利用上限は80,000または転送量上限400MByteです。
> 利用制限は初回アクセスから24時間が適用範囲です。

という API 利用の注意書きがあったので一応気を付けたほうが良さそうだし、`gzip` を活用するのが良さそうだ。
