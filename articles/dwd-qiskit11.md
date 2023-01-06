---
title: "Qiskit で遊んでみる (11) — Qiskit Advocate Mentorship Program"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem"]
published: true
---

# 目的

量子コンピュータ関連のプロジェクトに「Qiskit Advocate Mentorship Program」というものがあるのだが、これについて恐らく日本語の記事もないであろうし、一番乗りしてみようかという気持ちで記事を書いてみたい。

また、去年の秋に参加してみたので、簡単に振り返ってみたい。

# Qiskit Advocate Mentorship Program とは

結論から書くと **Qiskit Advocate Mentorship Program** とは「**Quantum Computing Mentorship Program にインスパイアされた Qiskit 版のプロジェクトであり、メンターとペアを組んで量子コンピュータ関連のプロジェクトに参加して、この分野の内側でスキルを磨くようなもの**」といったところだと考えている[^1]。これについて以下で簡単に触れてみたい。

[^1]: 歴史的経緯は正直知らないので憶測である。

この「Qiskit Advocate Mentorship Program」という用語であるが、例えば [Qiskit provider for Amazon Braket のご紹介](https://aws.amazon.com/jp/blogs/news/introducing-the-qiskit-provider-for-amazon-braket/) の文中に突然

> お客様からよくいただくリクエストである、Qiskit provider for Amazon Braket についてお伝えできることを嬉しく思います。  
> [...]  
> qiskit-braket-provider は、**[Qiskit Advocate Mentorship Program](https://github.com/qiskit-advocate/qamp-spring-21)** の一環として、オープンソースの寄稿者 David Morcuende によって開発されました。

として登場する。リンク (つまり [qamp-spring-21](https://github.com/qiskit-advocate/qamp-spring-21)) をクリックして一番下までスクロールすると

> their experiences and insights from [qosf mentorship program](https://qosf.org/qc_mentorship/)

というのがある。次に [Quantum Computing Mentorship Program](https://qosf.org/qc_mentorship/) に飛んで、DeepL で眺めると:

> このプログラムのアイデアは、多様なバックグラウンドを持つ多くの人々が、量子コンピューティングについてもっと知りたいと思いながらも、以下のような多くの課題を抱えていることに由来します。
>
> - 中級者向けの教材が不足している。
> - 中級者向けの教材が不足している。
> - 外から見ていると、この分野のことが歪んで見えてしまいがちだ。
>
> メンターとペアを組むことで、彼らがこれらのハードルを乗り越え、興味深いプロジェクトを生み出し、それがまた他の人を助けることになるよう支援したいと考えています。

ということで、このプログラムにインスパイアされたものではないかと思う。

# Qiskit Advocate とは？

文字列を分解すると `Qiskit Advocate` / `Mentorship` / `Program` となっている。すると「そもそも Qiskit Advocate とは？」となるのだが、これまたググっても日本語の情報は数件しか見つからない。その中から素晴らしい解説記事を 2 つ[^2]引用すると以下のようになる:

- [Qiskit Advocateになるために準備すべきこと徹底解説](https://www.investor-daiki.com/it/preparation_qiskit-advocate)
- [[2022年版] Qiskit Advocateについて](https://zenn.dev/bobo/articles/51fcd889693317)

[^2]: そして恐らくはこの瞬間は 2 記事のみで全てである。

一言で言うと「Qiskit のコミュニティの熱心な人たち」といった感じである。容易に想像がつくのだが「Qiskit Advocate Mentorship Program」に参加するには「Qiskit Advocate」に選ばれる必要があるというハードルがある[^3]。

[^3]: 「Qiskit Advocate Mentorship Program」に参加したいので、コミュニティに貢献して Qiskit Advocate になるという考え方もあると思う。

Qiskit Advocate になる方法は上記リンク先にすべて書いてあって、書いてある通りに実践すればなれるので詳細は割愛する。要件を勝手に引用すると

> 1. Qiskit Developer Certificationの取得[^4]
> 1. Qiskit Communityへの貢献活動(20pt以上)

[^4]: 歴史的な文書を探すと [Qiskit Advocate Applications Open Today — Here’s What You Can Expect from the Program](https://medium.com/qiskit/qiskit-advocate-applications-open-today-heres-what-you-can-expect-from-the-program-a1b7878f86b8) というのがあって、これを読むと最初は 3 時間くらいかかる独自の “the advocate test” なるテストがあったように思われる。その後、[IBM offers quantum industry’s first developer certification](https://www.ibm.com/blogs/research/2021/03/quantum-developer-certification/) という告知が一昨年の 3 月に公開されているので、[IBM Certified Associate Developer - Quantum Computation using Qiskit v0.2X](https://www.ibm.com/training/certification/C0010300) に置き換えられたのだと思う。90 分で済むので大分気楽である。

である。

貢献の一覧を見るとどれもこれもハードルが高そうで「うへぇ・・・」となるが、実はドキュメント類は新しい章がそれなりのペースで追加されがちなので、翻訳については当面は貢献し放題と思われる。「arXiv に論文を出してます」とかでなくても 2 ヶ月程度で Tier 1 の Gold Level Translator は簡単に狙えるので、~~DeepL~~ 腕の見せ所である。

# 参加してみた

と言う事で、Qiskit Advocate Mentorship Program Fall 2022 にメンティーとして参加してみたので記録を残してみたい。具体的には「[Enhancement of Aer-based `quantum_info`](https://github.com/qiskit-advocate/qamp-fall-22/issues/23)」というものに参加してみた。ちなみに参加時点では、

- かつかつの Qiskit の知識
- IBM Certified Associate Developer の試験対策でふわっと知った程度の純粋状態限定での密度行列の僅かな知識

しか持っていなかった。

課題は「Qiskit Terra が持っているクラス (下回りは `NumPy`) があるんだけど、[Qiskit Aer](https://github.com/Qiskit/qiskit-aer) に下回りを C++ でチューニングしたバージョンを追加したい。そのためガワにあたる Python 部分を実装してほしい」という内容である。今回は密度行列に対応する `DensityMatrix` クラスの Aer 版である `AerDensityMatrix` の対応 (Python 部分の実装) を行なった[^5]。

[^5]: `StabilizerState` までは手が出せなかったが、お気持ち表明程度に下調べしたものが [Qiskit で遊んでみる (9) — Shor の符号](/derwind/articles/dwd-qiskit09) などの記事の内容になる。

## 募集テーマと応募

「Qiskit Advocate Mentorship Program」は特に Qiskit の開発を行うものではなく、今回たまたまそういうプロジェクトを選んだだけで、実際には 40 程度のプロジェクト候補の中から選んで応募することになる。その時の募集のテーマにもよると思うが、一例として以下のようなものがある:

- 何かしらのプログラム・ツール開発、機能改善
- GitHub 上の issues への取り組み
- チュートリアル・ドキュメントの作成
- オリジナルコンテンツの作成
- ブログ記事の作成

また募集テーマに必ず参加できるわけではなく、取り組みたいプロジェクトを第 2 候補まで選んで、メンターに自分を売り込んでアピールするという形のものであった。興味や現状のスキルついてメンターと話し合って、マッチングするようなら募集メンティー人数の範囲で選ばれるということである。実際、方向性はマッチしたけど、人数の問題で「今回はごめんね」となってしまったプロジェクトが候補の中にあった。

## 日本人にとっての大きな課題 — 英語

さて・・・GitHub 上で格好をつけて英語でコメントを書いている・・・というわけではなく、どうしてもグローバルなプロジェクトであるので、日本人としてはつらいのだが、

- ドキュメントは英語
- プレゼンも英語 (2 回くらいある)

という最強のハードルがある。勿論下書きをカンニングペーパーに書いておいて、カンペ・ドリブン・スピーキングでなんとかするわけであるし、実際そうした。

結果としては [[Do not merge] Implement AerDensityMatrix](https://github.com/Qiskit/qiskit-aer/pull/1676) で draft PR を作成してプロジェクトとしては完了した。Credly の Digital Credentials も発行されたので、ようやく落ち着いてこのポエムを残している。

# あれれ？ドラフトのままでは？

まさにその通りなのだが、言い訳を書くと今回の課題はソフトウェア開発の性質が強かったので他の PR との依存性もあって、順に merge されないと噛み合わないというのがあった。よって、依存する PR がすべて merge されてくれないと `AerDensityMatrix` も取り込めなかったというのがある。仕方ないので、依存するコード類をローカルに取り込んでその上で開発して、それを draft PR したのである。結構依存関係に悩まされてつらいこともあったが、ソフトウェア開発者なら日常茶飯事の “あるある” だと思うので、それ程気になるものではないと思う[^6]。

[^6]: 逆に言えば、Labels に `type: code` がある課題は大なり小なりそういう傾向があるのかもしれない。

draft PR とは言え、コード的には完成のつもりで書いているので、手順は少々面倒臭いが PR ブランチをローカルに持ってきてインストールすると、以下のような内容が普通に実行できると思う。

```python
from qiskit_aer.quantum_info import AerDensityMatrix
from qiskit.circuit.library import QuantumVolume

qc = QuantumVolume(14, seed=1111)
dm = AerDensityMatrix(qc)
```

今回は課題の性質上、ソフトウェア開発に見られる困難に遭遇したが、どういった困難に遭遇するかは取り組む課題に依存するので一概にどうとは言えない。

# まとめ

Qiskit Advocate Mentorship Program Fall 2022 に参加した思い出をまとめてみた。小さいながら Qiskit Aer の機能開発に一部関われたのも良かったと思う。英語はつらかったけど、それも良い思い出かもしれない。

やはり日本の中で過ごしていると平気で何年間も英語を書きもせず話しもせずで過ごす感じになるので、普段 DeepL ばかり使っていると急に大変なことになるなという感想。早く「ほんやくコンニャク」AI が出てくることを望む。

# オマケ

「量子コンピュータ関連のプロジェクトに関わってみたいけど、なかなかないんだよね」ということもあると思う。完全日本語でいけるものなら恐らく [2023年 ナレッジモール研究： 枠を超えたワーキンググループでの共創研究活動](https://www.ibm.com/ibm/jp/ja/ibmcommunityjapan-wg-theme.html) の

- B-09 : 量子コンピューターの活用研究 －機械学習・量子化学計算・組み合わせ最適化への適用－

が該当すると思う。この活動事例については例えば [IBMのコミュニティ活動でQiskit Developer試験の受験虎の巻を作りました](https://qiita.com/w371dy/items/bc188f70ffdc2e7b9c7a) が参考になる。

IBM 関連以外でも量子プロジェクトを見かけないわけではないので、興味がある場合はアンテナを張っておけば良いと思われる。
