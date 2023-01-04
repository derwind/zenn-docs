---
title: "Qiskit で遊んでみる (11) — Qiskit Advocate Mentorship Program"
emoji: "🪐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["Qiskit", "poem"]
published: false
---

# 目的

「Qiskit Advocate Mentorship Program」というものについて恐らく日本語の記事もないであろうし、一番乗りしてみようかという気持ちで。

# Qiskit Advocate Mentorship Program とは

Conclusion first でも良いが年始早々にそういうのはしんどいので conclusion last でいってみたい。(少しスクロールしたところに結論を書いている。)

さて、例えば [Qiskit provider for Amazon Braket のご紹介](https://aws.amazon.com/jp/blogs/news/introducing-the-qiskit-provider-for-amazon-braket/) の文中に突然

> お客様からよくいただくリクエストである、Qiskit provider for Amazon Braket についてお伝えできることを嬉しく思います。  
> [...]  
> qiskit-braket-provider は、**[Qiskit Advocate Mentorship Program](https://github.com/qiskit-advocate/qamp-spring-21)** の一環として、オープンソースの寄稿者 David Morcuende によって開発されました。

という形で登場する “何か” であるものの、それ以上何も解説されていないものである。リンクを辿って [qamp-spring-21](https://github.com/qiskit-advocate/qamp-spring-21) に飛んで一番下までスクロールすると

> their experiences and insights from [qosf mentorship program](https://qosf.org/qc_mentorship/)

というのがある。次に [Quantum Computing Mentorship Program](https://qosf.org/qc_mentorship/) に飛んで、DeepL で眺めると:

> このプログラムのアイデアは、多様なバックグラウンドを持つ多くの人々が、量子コンピューティングについてもっと知りたいと思いながらも、以下のような多くの課題を抱えていることに由来します。
>
> - 中級者向けの教材が不足している。
> - 中級者向けの教材が不足している。
> - 外から見ていると、この分野のことが歪んで見えてしまいがちだ。
>
> メンターとペアを組むことで、彼らがこれらのハードルを乗り越え、興味深いプロジェクトを生み出し、それがまた他の人を助けることになるよう支援したいと考えています。

ということで、結論であるが、**Qiskit Advocate Mentorship Program** とは[^1]「**Quantum Computing Mentorship Program にインスパイアされた Qiskit 版のプロジェクトであり、メンターとペアを組んで量子コンピュータ関連のプロジェクトに参加して、この分野の内側でスキルを磨くようなもの**」といったところだと思う。

[^1]: 歴史的経緯は正直知らないので憶測だが

# Qiskit Advocate とは？

文字列を分解すると `Qiskit Advocate` / `Mentorship` / `Program` となっていそうである。すると「Qiskit Advocate とは？」となるのだが、これまたググってもあまり日本語の情報はなく、逆に特定の記事がひっかかる。その素晴らしい解説記事を 2 つ[^2]引用すると以下のようになる:

- [Qiskit Advocateになるために準備すべきこと徹底解説](https://www.investor-daiki.com/it/preparation_qiskit-advocate)
- [[2022年版] Qiskit Advocateについて](https://zenn.dev/bobo/articles/51fcd889693317)

[^2]: そして恐らくはこの瞬間は 2 記事のみである。

ここから容易に想像がつくのだが「Qiskit Advocate Mentorship Program」とは誰でも参加できるわけではなく「Qiskit Advocate」というものに選ばれる必要があるというハードルがある。

Qiskit Advocate になる方法は上記リンク先にすべて書いてあって、かつ書いてある通りに実践すればなれるので詳細は割愛する。勝手に要件を引用すると

> 1. Qiskit Developer Certificationの取得
> 1. Qiskit Communityへの貢献活動(20pt以上)

である。

貢献の一覧を見るとどれもこれもハードルが高そうで「うへぇ・・・」となるが、実はドキュメント類の翻訳は (新しいものがドカドカ追加されがちで) 常に微妙な進捗ではあるので、尽きる事なく貢献できる。「arXiv とかに論文を出してます」とかでなくても 2 ヶ月程度で Tier 1 の Gold Level Translator は簡単に狙えるので、~~DeepL~~ 腕の見せ所である。

# やってみた

と言う事で、Qiskit Advocate Mentorship Program Fall 2022 にメンティーとして参加してみたので記録を残してみたい。具体的には「[Enhancement of Aer-based `quantum_info`](https://github.com/qiskit-advocate/qamp-fall-22/issues/23)」というものに参加してみた。ちなみに参加時点では、かつかつの Qiskit の知識と、純粋状態限定での密度行列の僅かな知識しか持っていなかった。

課題は「Qiskit Terra が持っている下回りを `NumPy` で実装したクラスがあるんだけど、[Qiskit Aer](https://github.com/Qiskit/qiskit-aer) に C++ でゴリゴリにチューニングしたバージョンを追加したいんだけど」という内容である。今回は密度行列に対応する `DensityMatrix` クラスの Aer 版である `AerDensityMatrix` の対応を行なった。`StabilizerState` までは手が出せなかったが、お気持ち表明程度に下調べしたものが [Qiskit で遊んでみる (9) — Shor の符号](/derwind/articles/dwd-qiskit09) などの記事の内容になる。

さて・・・GitHub 上で格好をつけて英語でコメントを書いている・・・というわけではなく、どうしてもグローバルなプロジェクトであるので、日本人としてはつらいのだが、

- ドキュメントは英語
- プレゼンも英語 (2 回くらいある)

という最強のハードルがある。勿論下書きをカンニングペーパーに書いておいて、カンペ・ドリブン・スピーキングでなんとかするわけである。

結果としては [[Do not merge] Implement AerDensityMatrix](https://github.com/Qiskit/qiskit-aer/pull/1676) で draft PR を作成してプロジェクトとしては完了した。Credly の Digital Credentials も発行されたので、ようやく落ち着いてこのポエムを残している。

# ちゃんと merge されてないじゃないか！

まさにその通りなのだが、言い訳を書くと今回の課題はソフトウェア開発の性質が強かったので他の PR との依存性もあって、シーケンシャルに merge されないと噛み合わないというのがあった。よって、依存する PR がすべて merge されてくれないと `AerDensityMatrix` も取り込めなかったというのがある。仕方ないので、依存する PR や、場合によっては draft PR を引っ張ってきてローカルに merge して開発して、それを draft PR したのである。結構依存関係に悩まされてつらいこともあったが、ソフトウェア開発者なら日常茶飯事の “あるある” だと思う。

# まとめ

Qiskit Advocate Mentorship Program Fall 2022 に参加した思い出をまとめてみた。小さいながら Qiskit Aer の機能開発に一部関われたのも良かったと思う。英語はつらかったけど、それも良い思い出かもしれない。

やはり日本の中で過ごしていると平気で何年間も英語を書きもせず話しもせずで過ごす感じになるので、普段 DeepL ばかり使っていると急に大変なことになるなという感想。早く「ほんやくコンニャク」AI が出てくることを望む。
