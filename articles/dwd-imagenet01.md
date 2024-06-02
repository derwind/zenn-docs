---
title: "ImageNet について考える (1) — Tiny ImageNet"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Python", "機械学習", "ImageNet", "NLP"]
published: true
---

# 目的

ディープラーニングと言えば、MNIST か ImageNet という偏見があって、今回 ImageNet・・・ではなく、その簡易版？的な Tiny ImageNet というデータセットについて考えてみたい。

# Tiny ImageNet

[Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet/data) は Kaggle コンペでも用いられたことがあるデータセットのようだが、Stanford の [CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) 関連で作られたデータセットのようで “930.pdf” という pdf に詳細が書かれている。64x64 のサイズの画像で、200 クラスという構成である。

結構なめてかかった結果、適当な CNN で分類を試みるも過学習するばかりで val acc が伸びない。最終的には適当なモデルではなく VGG16 を用いた転移学習を試みたがそれでも簡単ではない様子。ということで情報集めをした。

# 研究プロジェクト

VGG16 ベースの取り組みを探したところ、[ImageClassificationProject-IITK](https://github.com/ayushdabra/ImageClassificationProject-IITK) というプロジェクトが見つかった。モデルのアーキテクチャのみならずレポートも添えて合って非常に良い。

レポートには大雑把には「解像度が低くて細部をとらえるのが難しいし、そもそもトラックとバスを厳密に区別するの難しいから 200 クラスといえど大変難しい」ということが書いてあって、実際の実験も E2E なものではなく、転移学習でベースモデルを作って、次に全体を訓練可能にしてファインチューニングをしている。ベースモデルの作成も学習率の制御に結構手間なことをしているようで、簡単なタスクではなかったことが窺えた。

ということで、データセットを眺めることに。

# データセット可視化

必要なモジュールを import:

```python
from __future__ import annotations

from torchvision import datasets
from torchvision.transforms import v2

import pprint
```

適当にデータローダを作成。各クラス 5 枚ずつ取り出してみる:

```python
train_transform = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]),
])

n_class = 200
n_image_per_class = 5

def pick_data(dataset, n_class, n_image_per_class):
    data_per_class = len(dataset) // n_class
    data = np.arange(len(dataset)).reshape(-1, data_per_class)
    data = data[:, :n_image_per_class]
    return data.flatten().tolist()

dataset = datasets.ImageFolder("tiny-imagenet-200/train", transform=train_transform)
dataset = torch.utils.data.Subset(
    dataset, pick_data(dataset, n_class, n_image_per_class)
)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=n_image_per_class*10, shuffle=False
)
```

可視化:

```python
for imgs, lbls in loader:
    imgs = imgs.numpy() * [[[[0.229]], [[0.224]], [[0.225]]]] + [[[[0.485]], [[0.456]], [[0.406]]]]
    imgs = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    lbls = [v.item() for v in lbls.numpy()]

    row = 10
    col = 5
    n_data = row * col

    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(8,10))
    for i, img in enumerate(imgs[:n_data]):
        r= i // col
        c= i % col
        ax[r, c].set_title(lbls[i], fontsize=8)
        ax[r, c].axes.xaxis.set_visible(False)
        ax[r, c].axes.yaxis.set_visible(False)
        ax[r, c].imshow(img)
```

![](/images/dwd-imagenet01/001.png =500x)

意外と何が何やらという状態だということが分かった・・・。

# 似たような画像クラスをくくり出す

`tiny-imagenet-200/wnids.txt` と `tiny-imagenet-200/words.txt` を使うと、画像の説明文を得られるので、説明文を使って文章ベクトルを作って類似度に基づくクラスタリングを行ってみることにした。画像なので

- HOG 特徴量
- 離散コサイン変換を用いた知覚ハッシュ

による分類も考えたが、上記で見た画像の可視化の感じだと、背景も多様性が多く、あまり期待できる結果にならない気がしたので、テキストベースで試すことにした。

## 画像の説明文

以下のような感じで、画像のクラスから説明文への辞書 `idx2explanation` と、その逆 `explanation2idx` を作った。後者を作るに当たっては、key を文章ベクトルにしようと考えているので、簡単のためカンマ区切りの先頭の文章だけとした。

```python
with open("tiny-imagenet-200/wnids.txt") as fin:
    idx2id = {i: val.strip() for i, val in enumerate(sorted(fin.readlines()))}


with open("tiny-imagenet-200/words.txt") as fin:
    id2explanation = dict(line.rstrip().split("\t") for line in fin.readlines())


idx2explanation = {idx: id2explanation[id_] for idx, id_ in idx2id.items()}
explanation2idx = {v.split(",")[0]: k for k, v in idx2explanation.items()}
```

先ほどの画像について説明を出してみると、以下のように分かるような分からないようなという内容である。

```python
pprint.pprint(idx2explanation)
```

> {0: 'goldfish, Carassius auratus',
>  1: 'European fire salamander, Salamandra salamandra',
>  2: 'bullfrog, Rana catesbeiana',
>  3: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
>  4: 'American alligator, Alligator mississipiensis',
> ...

## BERT を持ちいた文章ベクトルの作成

詳しくない領域なので、[BERTによる自然言語処理入門
Transformersを使った実践プログラミング](https://www.ohmsha.co.jp/book/9784274227264/) や Zenn の記事 [【自然言語処理】BERTの単語ベクトルで「王+女-男」を計算してみる](https://zenn.dev/schnell/articles/4acc48c49eb8eb) を参考にした。

後で使うモジュール類をざくっと import:

```python
from transformers import BertTokenizer
from transformers import BertModel
import torch

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
import holoviews as hv
from holoviews import opts

from collections import defaultdict
```

今回日本語を扱うわけではないので、よくある `tohoku-nlp/bert-base-japanese-v3` は使わない。Hugging Face の [BERT](https://huggingface.co/docs/transformers/ja/model_doc/bert) を確認して以下を用いることにした。

```python
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased")
bert_model.eval()
```

適当に予備実験をしたところ、`"man and woman"` みたいな文章はトークナイザを通すと `['[CLS]', 'main', 'and', 'woman', '[SEP]']` のようになっていた。`[CLS]` と `[SEP]` は不要なのだがという気持ちもあったが、文章ベクトルを作るにあたって [【🔰自然言語処理】単語の分散表現② Word2VecとBERT](https://tt-tsukumochi.com/archives/4770) を参考したところ、そんな細かい処理をしなくて良さそうだったので記事に倣った。多分アテンションマスクを掛けこむことで、十分に寄与が薄まるのだろう・・・。

```python
sentence_vecs = []


for words in explanation2idx.keys():
    with torch.no_grad():
        inputs = tokenizer(words, return_tensors="pt")
        outputs = bert_model(**inputs) 

        last_hidden_state = outputs.last_hidden_state

        attention_mask = inputs.attention_mask.unsqueeze(-1)
        valid_token_num = attention_mask.sum(1)
        sentence_vec = (last_hidden_state*attention_mask).sum(1) / valid_token_num
        sentence_vec = sentence_vec.detach().cpu().numpy().flatten()
        sentence_vecs.append(sentence_vec)

sentence_vecs = np.array(sentence_vecs).reshape(len(sentence_vecs), -1)
```

これを可視化したい。[【自然言語処理】BERTの単語ベクトルで「王+女-男」を計算してみる](https://zenn.dev/schnell/articles/4acc48c49eb8eb) を大いに参考にして t-SNE で次元削減をしてプロットした。

```python
tsne = TSNE(n_components=2, random_state=0)
reduced_vectors = tsne.fit_transform(sentence_vecs)

hv.extension('plotly')

points = hv.Points(reduced_vectors)
labels = hv.Labels(
    {
        ("x", "y"): reduced_vectors,
        "text": list(explanation2idx.keys()),
    }, ["x", "y"], "text")

(points * labels).opts(
    opts.Labels(xoffset=0.05, yoffset=0.05, size=14, padding=0.2,
                width=1200, height=800),
    opts.Points(color='black', marker='x', size=3),
)
```

![](/images/dwd-imagenet01/002.png)

右下のほうを見ると「mashed potato, confectionery, frying pan」と「sports car, go-kart」のクラスタ (？) が見えるのでまぁまぁそれっぽい気がしなくもない。

t-SNE だと分からない部分もあるので、ベクトルを次元削減せずに K-Means でクラスタリングしてみる。

## K-Means によるクラスタリング

もしも時間がかかりそうなら Meta の [Faiss](https://github.com/facebookresearch/faiss)、特に [faiss-gpu](https://pypi.org/project/faiss-gpu/) を使うことにして、まずは scikit-learn で試した。

```python
%%time

estimator = KMeans(n_clusters=30, n_init="auto")
estimator.fit(sentence_vecs)
```

> CPU times: user 57.3 ms, sys: 25.2 ms, total: 82.6 ms
> Wall time: 85.3 ms

一瞬だったので、scikit-learn で問題なかった。

## クラスタの表示

以下のような内容になった。英単語がよく分からない・・・。

```python
label2explanations = defaultdict(lambda: [])

for lbl, explanation in zip(estimator.labels_, explanation2idx.keys()):
    label2explanations[lbl].append(explanation)

pprint.pprint(label2explanations)
```

>             {0: ['gazelle'],
>              1: ['boa constrictor',
>                  'brain coral',
>                  'American lobster',
>                  'maypole',
>                  'moving van',
>                  'spider web',
>                  'turnstile'],
>              2: ['goose',
>                  'snail',
>                  'slug',
>                  'Chihuahua',
>                  'lion',
>                  'fly',
>                  'bee',
>                  'monarch',
>                  'hog',
>                  'ox',
>                  'altar',
>                  'apron',
>                  'barn',
>                  'barrel',
>                  'beacon',
>                  'broom',
>                  'bucket',
>                  'candle',
>                  'chain',
>                  'chest',
>                  'convertible',
>                  'crane',
>                  'lifeboat',
>                  'limousine',
>                  'nail',
>                  'projectile',
>                  'reel',
>                  'sock',
>                  'sunglasses',
>                  'teddy',
>                  'thatch',
>                  'torch',
>                  'tractor',
>                  'umbrella',
>                  'plate',
>                  'mushroom',
>                  'orange',
>                  'lemon',
>                  'banana',
>                  'cliff',
>                  'lakeside'],
>              3: ['miniskirt', 'neck brace', 'swimming trunks'],
>              4: ['Yorkshire terrier',
>                  'golden retriever',
>                  'Labrador retriever',
>                  'cougar'],
>              5: ['cash machine',
>                  'computer keyboard',
>                  'dining table',
>                  'freight car',
>                  'parking meter'],
>              6: ['pretzel', 'mashed potato', 'meat loaf'],
>              7: ['goldfish',
>                  'trilobite',
>                  'jellyfish',
>                  'sea slug',
>                  'sea cucumber',
>                  'African elephant',
>                  'cliff dwelling',
>                  'snorkel',
>                  'coral reef',
>                  'seashore'],
>              8: ['flagpole',
>                  'steel arch bridge',
>                  'suspension bridge',
>                  'trolleybus',
>                  'viaduct',
>                  'water tower'],
>              9: ['go-kart'],
>              10: ['sulphur butterfly',
>                   'baboon',
>                   'abacus',
>                   'dumbbell',
>                   'bell pepper',
>                   'acorn'],
>              11: ['guacamole'],
>              12: ['American alligator',
>                   'koala',
>                   'spiny lobster',
>                   'German shepherd',
>                   'standard poodle',
>                   'Persian cat',
>                   'Egyptian cat',
>                   'brown bear',
>                   'walking stick',
>                   'mantis',
>                   'guinea pig',
>                   'Arabian camel',
>                   'lesser panda',
>                   'cardigan',
>                   'drumstick',
>                   'fur coat',
>                   'kimono',
>                   'poncho',
>                   'sandal',
>                   'alp'],
>              13: ['basketball', 'dam', 'volleyball'],
>              14: ['barbershop', 'gondola'],
>              15: ['bullfrog',
>                   'black widow',
>                   'king penguin',
>                   'tabby',
>                   'ladybug',
>                   'dragonfly',
>                   'bannister',
>                   'beaker',
>                   'birdhouse',
>                   'bullet train',
>                   'hourglass',
>                   'jinrikisha',
>                   'stopwatch',
>                   'wok',
>                   'ice lolly',
>                   'potpie'],
>              16: ['European fire salamander',
>                   'tailed frog',
>                   'centipede',
>                   'black stork',
>                   'grasshopper',
>                   'cockroach'],
>              17: ['tarantula', 'cauliflower', 'pomegranate'],
>              18: ['albatross',
>                   'punching bag',
>                   'scoreboard',
>                   'comic book',
>                   'espresso'],
>              19: ["potter's wheel"],
>              20: ['orangutan', 'chimpanzee'],
>              21: ['lawn mower', 'picket fence', 'rocking chair'],
>              22: ['scorpion',
>                   'dugong',
>                   'bison',
>                   'academic gown',
>                   'beach wagon',
>                   'bikini',
>                   'binoculars',
>                   'brass',
>                   'cannon',
>                   'CD player',
>                   'iPod',
>                   'military uniform',
>                   'oboe',
>                   'pole',
>                   'pizza'],
>              23: ['bathtub',
>                   'beer bottle',
>                   'Christmas stocking',
>                   'gasmask',
>                   'lampshade',
>                   'pay-phone',
>                   'pill bottle',
>                   'pop bottle',
>                   'rugby ball',
>                   'syringe',
>                   'teapot',
>                   'water jug',
>                   'wooden spoon',
>                   'ice cream'],
>              24: ['obelisk', 'triumphal arch'],
>              25: ['sombrero'],
>              26: ['confectionery', 'frying pan'],
>              27: ['backpack',
>                   'butcher shop',
>                   'desk',
>                   'fountain',
>                   'magnetic compass',
>                   'organ',
>                   'plunger',
>                   'police van',
>                   'refrigerator',
>                   'remote control',
>                   'school bus',
>                   'sewing machine',
>                   'space heater',
>                   'sports car',
>                   'vestment'],
>              28: ['bighorn'],
>              29: ['bow tie']})

よく分からないので、ChatGPT-4o に翻訳をさせた。なんとなく分かるような気がするものから、よく分からないものまで色々だ:

>         {0: ['ガゼル'],
>          1: ['ボアコンストリクター',
>              '脳サンゴ',
>              'アメリカンロブスター',
>              'メイポール',
>              '引っ越し用トラック',
>              'クモの巣',
>              '回転柵'],
>          2: ['ガチョウ',
>              'カタツムリ',
>              'ナメクジ',
>              'チワワ',
>              'ライオン',
>              'ハエ',
>              'ミツバチ',
>              'オオカバマダラ',
>              'ブタ',
>              'オックス',
>              '祭壇',
>              'エプロン',
>              '納屋',
>              '樽',
>              'ビーコン',
>              'ほうき',
>              'バケツ',
>              'キャンドル',
>              '鎖',
>              'チェスト',
>              'コンバーチブル',
>              'クレーン',
>              '救命ボート',
>              'リムジン',
>              '釘',
>              '投射物',
>              'リール',
>              '靴下',
>              'サングラス',
>              'テディ',
>              'わらぶき屋根',
>              'トーチ',
>              'トラクター',
>              '傘',
>              '皿',
>              'キノコ',
>              'オレンジ',
>              'レモン',
>              'バナナ',
>              '崖',
>              '湖畔'],
>          3: ['ミニスカート', 'ネックブレース', '水着トランクス'],
>          4: ['ヨークシャーテリア',
>              'ゴールデンレトリバー',
>              'ラブラドールレトリバー',
>              'クーガー'],
>          5: ['現金自動預け払い機',
>              'コンピュータキーボード',
>              'ダイニングテーブル',
>              '貨車',
>              'パーキングメーター'],
>          6: ['プレッツェル', 'マッシュポテト', 'ミートローフ'],
>          7: ['金魚',
>              '三葉虫',
>              'クラゲ',
>              'ウミウシ',
>              'ナマコ',
>              'アフリカ象',
>              '崖住居',
>              'シュノーケル',
>              'サンゴ礁',
>              '海岸'],
>          8: ['旗竿',
>              '鉄製アーチ橋',
>              '吊り橋',
>              'トロリーバス',
>              '高架橋',
>              '給水塔'],
>          9: ['ゴーカート'],
>          10: ['硫黄蝶',
>               'ヒヒ',
>               'そろばん',
>               'ダンベル',
>               'ピーマン',
>               'ドングリ'],
>          11: ['ワカモレ'],
>          12: ['アメリカワニ',
>               'コアラ',
>               'イセエビ',
>               'ジャーマンシェパード',
>               'スタンダードプードル',
>               'ペルシャ猫',
>               'エジプト猫',
>               'クマ',
>               '歩行棒',
>               'カマキリ',
>               'モルモット',
>               'アラビアラクダ',
>               'レッサーパンダ',
>               'カーディガン',
>               'ドラムスティック',
>               '毛皮のコート',
>               '着物',
>               'ポンチョ',
>               'サンダル',
>               'アルプ'],
>          13: ['バスケットボール', 'ダム', 'バレーボール'],
>          14: ['理髪店', 'ゴンドラ'],
>          15: ['ウシガエル',
>               'クロゴケグモ',
>               'キングペンギン',
>               'トラ猫',
>               'テントウムシ',
>               'トンボ',
>               '手すり',
>               'ビーカー',
>               'バードハウス',
>               '新幹線',
>               '砂時計',
>               '人力車',
>               'ストップウォッチ',
>               '中華鍋',
>               'アイスキャンディー',
>               'ポットパイ'],
>          16: ['ヨーロッパファイアサラマンダー',
>               'シッポアマガエル',
>               'ムカデ',
>               'クロコウノトリ',
>               'バッタ',
>               'ゴキブリ'],
>          17: ['タランチュラ', 'カリフラワー', 'ザクロ'],
>          18: ['アルバトロス',
>               'サンドバッグ',
>               'スコアボード',
>               '漫画本',
>               'エスプレッソ'],
>          19: ['ろくろ'],
>          20: ['オランウータン', 'チンパンジー'],
>          21: ['芝刈り機', '杭柵', 'ロッキングチェア'],
>          22: ['サソリ',
>               'ジュゴン',
>               'バイソン',
>               '学位ガウン',
>               'ビーチワゴン',
>               'ビキニ',
>               '双眼鏡',
>               'ブラス',
>               '大砲',
>               'CDプレーヤー',
>               'iPod',
>               '軍服',
>               'オーボエ',
>               '柱',
>               'ピザ'],
>          23: ['バスタブ',
>               'ビール瓶',
>               'クリスマスストッキング',
>               'ガスマスク',
>               'ランプシェード',
>               '公衆電話',
>               'ピルボトル',
>               'ポップボトル',
>               'ラグビーボール',
>               '注射器',
>               'ティーポット',
>               '水差し',
>               '木製スプーン',
>               'アイスクリーム'],
>          24: ['オベリスク', '凱旋門'],
>          25: ['ソンブレロ'],
>          26: ['菓子', 'フライパン'],
>          27: ['バックパック',
>               '精肉店',
>               '机',
>               '噴水',
>               '方位磁針',
>               'オルガン',
>               'プランジャー',
>               'パトカー',
>               '冷蔵庫',
>               'リモコン',
>               'スクールバス',
>               'ミシン',
>               'スペースヒーター',
>               'スポーツカー',
>               '礼服'],
>          28: ['ビッグホーン'],
>          29: ['蝶ネクタイ']})

例えばクラスタ 7 は海洋関係であろうか？

結局のところ、Word2Vec と同様に、意味的な近さで説明文がクラスタリングされているので、画像的な類似度の観点では何とも言えない部分はある。一方で、純粋に画像だけで分類した場合、それはそれでなかなか厳しい結果になったような気もする。

# まとめ

100% 欲しい結果が得られたわけではないが、

- Tiny ImageNet の中身の確認
- 類似画像クラスのクラスタリング

ができた。各クラスタごとに適当なクラスを 1 つずつ選んで VGG16 で分類するモデルを作ればもう少し簡単に訓練できるのでは？という期待がある。
