---
title: "Python でリストの中身を一時的に置き換える"
emoji: "🐍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python"]
published: false
---

# 目的

ふと Python の `list` で中身を一時的に置き換えながらループを回したいという謎めいたことがしたくなって、試したらできたので備忘録を残す。

# 実装

[contextlib --- with 文コンテキスト用ユーティリティ](https://docs.python.org/ja/3/library/contextlib.html) を参考に、例えば以下のようになりそうである。一時的に 1 個だけ要素を置換することもできるし、複数置換することもできる。

```python
from __future__ import annotations
import contextlib
from typing import Any
from collections.abc import Sequence


@contextlib.contextmanager
def temporarily_replaced_with(
    arr: list[Any],
    index: int | Sequence[int],
    by: Any | Sequence[Any]
):
    if isinstance(index, Sequence):
        backups = [arr[i] for i in index]
    else:
        backups = [arr[index]]
        index = [index]
        by = [by]

    try:
        for i, v in zip(index, by):
            arr[i] = v
        yield arr
    finally:
        for i, v in zip(index, backups):
            arr[i] = v
```

# 実験

- 指定位置の値を一時的に 3 倍にしたい:

```python
import numpy as np

a = np.arange(10)

for i in range(1, len(a)):
    with temporarily_replaced_with(a, i, by=a[i]*3):
        print(a)
print('original:', a)
```

> [0 **3** 2 3 4 5 6 7 8 9]
> [0 1 **6** 3 4 5 6 7 8 9]
> [0 1 2 **9** 4 5 6 7 8 9]
> [ 0  1  2  3 **12**  5  6  7  8  9]
> [ 0  1  2  3  4 **15**  6  7  8  9]
> [ 0  1  2  3  4  5 **18**  7  8  9]
> [ 0  1  2  3  4  5  6 **21**  8  9]
> [ 0  1  2  3  4  5  6  7 **24**  9]
> [ 0  1  2  3  4  5  6  7  8 **27**]
> original: [0 1 2 3 4 5 6 7 8 9]

- 指定位置とその隣の値を一時的に交換したい:

```python
a = np.arange(10)

for i in range(1, len(a)-1):
    with temporarily_replaced_with(a, [i, i+1], [a[i+1], a[i]]):
        print(a)
print('original:', a)
```

> [0 **2 1** 3 4 5 6 7 8 9]
> [0 1 **3 2** 4 5 6 7 8 9]
> [0 1 2 **4 3** 5 6 7 8 9]
> [0 1 2 3 **5 4** 6 7 8 9]
> [0 1 2 3 4 **6 5** 7 8 9]
> [0 1 2 3 4 5 **7 6** 8 9]
> [0 1 2 3 4 5 6 **8 7** 9]
> [0 1 2 3 4 5 6 7 **9 8**]
> original: [0 1 2 3 4 5 6 7 8 9]

一応期待通りに動作していそうである。

# まとめ

あまり普段 `@contextlib.contextmanager` を使わないので、ひょっとしたら 2 回目くらいか、下手すれば今回が初めてかもしれない。ただ、「そういう機能があるらしい」くらいは頭の片隅に置いておくと、10 年に 1 回くらいしか使うチャンスがなくても、ピンと来ることもあるかもしれない。
