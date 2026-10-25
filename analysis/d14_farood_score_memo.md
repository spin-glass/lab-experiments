# far-OOD スコア選定メモ — softmax を外すのが先、距離は補完信号に降格する

## 本リポジトリでの位置づけ（収載時の注記）

- 判断ID: **D14**（design.md §8）。G0（docs/g0_nonfood_plan.md）の検出器候補選定に対する外部証跡として収載する。
- 実測は本リポの構成（E1 凍結 CLIP のゼロショット）ではなく、別プロジェクトの学習済み 5 クラス線形ヘッド（手元 ckpt）上のもの。数値は本リポの期待値・判定基準には使わない（references.md 末尾の注記と同じ扱い。判定基準は事前登録の数値のみ）。
- G0 事前登録との関係: (1) 本メモの主批判「素の Mahalanobis は未正規化ゆえに負ける」は G0 では織り込み済み——主構成は Mahalanobis++（L2 正規化、[MahaPP25]）で、素の Mahalanobis はアブレーション。(2) MaxLogit（[MLS22]）・Energy（[Energy20]）・ViM 型合成（[ViM22]）は G0 主構成に含まれない。追加する場合は plan の改訂履歴（§改訂）に記録してから行う（凍結タグ g0-freeze-v1 は未打刻）。
- 「デッキへの反映」「claims.yaml」「次のアクション」は作成元プロジェクト側のタスクであり、本リポの作業項目ではない。

---

- 日付: 2026-09-08　作成: Claude（Devin 検証報告のレビュー＋文献調査）　状態: 案（本番 ckpt 未再現）
- 問い: 距離ベース（Mahalanobis・kNN）は捨てるべきか。負けたのは未チューニングのせいか。
- 対象: 手元 ckpt、5 クラス線形ヘッド `Linear(1024,1024) → Dropout(0.2) → Linear(1024,5) → Softmax`、埋め込み 1024 次元

## 結論

1. 距離ベースは捨てない。ただし「距離が主役」の筋立ては捨てる。今すぐ効く手当ては softmax を外して最大ロジット（MaxLogit）を使うこと。参照集合も距離計算も不要。
2. Mahalanobis が負けた主因は未チューニング（特徴の ℓ2 正規化未実施）と標本の小ささ。文献では素の Mahalanobis が MSP / MaxLogit に負けるのは通例で、正規化で一貫して改善する。
3. 「MaxLogit が距離より優れる」は現標本では有意でない。確定しているのは「softmax → MaxLogit の改善」だけ。距離は特徴残差として MaxLogit / Energy に足す（ViM 型）のが文献上の最善で、これが未実験。

## 実測（Devin、手元 ckpt、負例 130 件・1 枚で約 0.8pt）

R@FPRx% = 真の料理の誤棄却率 x% の閾値における非料理の検出率。

| スコア | AUROC | R@FPR1% | R@FPR5% | R@FPR10% | IQR（真の料理） |
|---|---|---|---|---|---|
| softmax 最大確率（現行） | 0.892 | 19.2 | 34.1 | 73.5 | 0.0000 |
| ロジット最大値（MaxLogit） | **0.903** | 21.0 | **56.9** | **78.6** | 4.1192 |
| ロジット差（1位−2位） | 0.890 | 19.3 | 33.8 | 71.7 | 6.7846 |
| Mahalanobis 距離 | 0.858 | 10.2 | 47.1 | 61.7 | 6.2487 |
| kNN（料理） | 0.852 | **22.3** | 41.6 | 61.9 | 0.0908 |

機構（Devin 検算：埋め込みからの再計算誤差 3.3e-06、argmax 一致 100%）

- ヘッドは活性化なしの厳密な affine。softmax は全ロジットへの定数加算に不変（実測 4.2e-07）なので、W の中心化行空間 D（特異値 [132.7, 113.9, 102.7, 95.8, 0] → 4 次元）しか見ず、共通成分 w̄·z + b̄ の 1 次元を構造的に捨てる。
- 捨てた 1 次元は補完的：共通成分のみ AUROC 0.673（符号反転後）、差の成分のみ 0.881 / R@FPR5% 27.0%、両方（= MaxLogit）0.903 / 56.9%。両者の Spearman 相関は −0.223。
- softmax は飽和：真の料理の 89.2% が 0.999 以上（IQR 0.0000）、そこに非料理の 26.3% も混在。AUROC は順位しか見ないので下がらないが、閾値操作では同点の塊を切り分けられない。
- 距離側：Mahalanobis 0.858 ＞ ユークリッド 0.807（効いているのは共分散による白色化）。far-OOD の変位は D⊥ に偏っていない（D 成分 0.826 / D⊥ 成分 0.801、D は変位エネルギーの 1.8%）。

## 文献照合

| 出典 | 知見 | 本件への含意 |
|---|---|---|
| OpenOOD v1.5（Zhang+ 2023） | ImageNet-1K far-OOD AUROC：MDS 74.25 / MSP 85.23 / MLS 89.57 / Energy 89.47 / ViM 92.68 / KNN 90.18 / ReAct 93.67 / ASH 95.74。全ベンチマークで勝つ単独手法なし。RMDS は ResNet より transformer に向く | 素の Mahalanobis が MaxLogit に負けるのは通例。far-OOD 上位は特徴×ロジットの合成型。CLIP 凍結特徴なら RMD を試す |
| Mahalanobis++（Müller & Hein, ICML 2025） | 素の Mahalanobis のモデル間ばらつきは特徴ノルム変動（ガウス仮定違反）が原因。ℓ2 正規化で 44 モデル一貫改善 | Devin の Mahalanobis は正規化未実施 → 未チューニングと判断。正規化版で再測定 |
| Deep kNN（Sun+, ICML 2022） | 正規化の有無で FPR95 が 61.05pt 変わる | kNN も正規化前提で再測定 |
| ViM（Wang+, CVPR 2022） | 特徴空間でしか見えない OOD とロジット空間でしか見えない OOD が併存。主部分空間の残差ノルムを仮想ロジットとしてロジットに足す（= Energy ＋ 残差）。特徴主成分ベースの残差は W 零空間（NuSA）より良い | Devin の D⊥ 成分（0.801）が残差スコアに相当。MaxLogit / Energy との合成が未実験 = 最も安い次の一手 |
| DML（Zhang & Xiang, CVPR 2023） | MaxLogit = cos 類似度 × ロジットノルム。中核は cos、ノルムが足を引っ張る場合がある | Devin の共通成分は主にノルム側。cos とノルムの重みは分離調整可 |
| Fort+（NeurIPS 2021）/ RMD（Ren+ 2021） | 事前学習 transformer 特徴では Mahalanobis・RMD が強い | 凍結 CLIP 特徴での距離は文献上は強いはず → 実装差を疑う |
| NegLabel（Jiang+, ICLR 2024） | CLIP ゼロショットに大量の負ラベルを足すだけで ImageNet-1K OOD で AUROC 94.21 / FPR95 25.40、多くの教師あり手法を上回る | 観覧車・ミシュランマン等「名前で言える」far-OOD は距離よりゼロショット側（既定の負プロンプト即日試験）が本命 |

## 留保

- 標本 ≈130 件：R@FPR5% の 95%CI は ±8.5pt（二項近似）。34.1 → 56.9 は実質的、MaxLogit 56.9 vs Mahalanobis 47.1 の 9.8pt は有意でない。AUROC 0.903 vs 0.858 も境界的。
- 動作点依存：R@FPR1% では kNN 22.3 ＞ MaxLogit 21.0 ＞ softmax 19.2。設計値が誤棄却率 1% なら MaxLogit 優位は消える。
- 未測定：Energy（logsumexp）。不明：Mahalanobis の実装（正規化・共分散 shrinkage・クラス条件付きか単一か・fit 標本数）。
- 手元 ckpt の結果であり、本番 `labeler.ckpt` では未再現。

## デッキへの反映

- 8 枚目：「確信度は隣のクラスとの差」→「softmax が捨てる絶対量（ロジット共通成分 = ノルム側）が効く」に書き換え。差だけでも AUROC 0.881。
- 10〜13 枚目：「距離を使う」の独立章を廃止。「特徴残差を補完信号として足す（ViM 相当）」に縮約。
- 数値は全て claims.yaml 経由で引用する。本メモの数値は暫定（手元 ckpt、CI 未付与）。

## 次のアクション（1〜2 時間、ここで二次目標側は打ち止め）

1. Devin の検証をスクリプト化。追加スコア：Energy、ℓ2 正規化 Mahalanobis、RMD、Energy ＋ α × 残差ノルム（ViM 相当）。
2. 全指標にブートストラップ 95%CI、動作点は FPR 1% と 5% の両方を出力。
3. 手元 ckpt と本番 `labeler.ckpt` で一括実行 → claims.yaml に登録。
4. 未決：本番の許容誤棄却率の設計値（1% か 5% か）。動作点が決まるまで距離ベースの採否は保留。

## 出典

- OpenOOD v1.5: https://arxiv.org/abs/2306.09301
- Mahalanobis++: https://arxiv.org/abs/2505.18032
- Deep Nearest Neighbors: https://arxiv.org/abs/2204.06507
- ViM: https://arxiv.org/abs/2203.10807
- DML: https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.html
- Exploring the Limits of OOD Detection: https://arxiv.org/abs/2106.03004　RMD: https://arxiv.org/abs/2106.09022
- NegLabel: https://arxiv.org/abs/2403.20078
