# references.md — カスケード設計の根拠文献

design.md および reports/ で数値・設計主張をする際の引用元。各項目に「支える主張」を付す。

## 1. 系譜・原型

- **[VJ01]** Viola, P. & Jones, M. "Rapid Object Detection using a Boosted Cascade of Simple Features." CVPR 2001. DOI: 10.1109/CVPR.2001.990517 / PDF: https://www.merl.com/publications/docs/TR2004-043.pdf
  — 主張: カスケードの原型。単純な分類器を複雑さの昇順に直列化し、大多数（背景）を初段で早期棄却して計算を難例に集中する構成。

## 2. コスト効率の実証

- **[Fru23]** Chen, Zaharia & Zou. "FrugalGPT." arXiv:2305.05176. https://arxiv.org/abs/2305.05176
  — 主張: カスケードで最良単一モデル同等性能を最大98%コスト減、または同コストで+4%精度。
- **[ABC24]** Kolawole et al. "Agreement-Based Cascading for Efficient Inference." TMLR 2024. https://arxiv.org/abs/2407.02348
  — 主張: 複数モデルの合意を委譲信号に使う変種（Stage3の2モデル一致確定の親戚）。
- **[Route24]** Ong et al. "RouteLLM." arXiv:2406.18665. https://arxiv.org/abs/2406.18665
  — 主張: ルーティング（入口で1回の並列分岐）との対比。カスケードは前段出力を委譲判断に使う直列構成。

## 3. 委譲規則と理論（selective prediction）

- **[Chow70]** Chow, C.K. "On Optimum Recognition Error and Reject Tradeoff." IEEE Trans. Information Theory, 16:41–46, 1970.（1957年論文が原典）
  — 主張: reject option（棄権つき分類）の原典。誤り率と棄却率のトレードオフ。
- **[EW10]** El-Yaniv & Wiener. "On the Foundations of Noise-free Selective Classification." JMLR 11:1605–1641, 2010.
  — 主張: selective classification の基礎理論（risk–coverage の枠組み）。
- **[GE17]** Geifman & El-Yaniv. "Selective Classification for Deep Neural Networks." NeurIPS 2017. https://arxiv.org/abs/1705.08500
  — 主張: 学習済みNNに目標リスクを保証する棄却規則を後付けできる。例: ImageNet top-5 誤り2%保証をカバレッジ約60%で達成。M2のしきい値設計の理論枠。
- **[Jit23]** Jitkrittum et al. "When Does Confidence-Based Cascade Deferral Suffice?" NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/file/1f09e1ee5035a4c3fe38a5681cae5815-Paper-Conference.pdf
  — 主張: 確信度ベース委譲（最大確率・エントロピー）が十分な条件と、後段の得意領域がずれる場合の限界。ペア別しきい値の根拠。
- **[CP-Intro23]** Angelopoulos & Bates. "Conformal Prediction: A Gentle Introduction." Foundations and Trends in Machine Learning 16(4):494–591, 2023.
  — 主張: 分布仮定なし・有限サンプルの被覆保証を任意の学習済み分類器に後付けできる枠組みの入門。委譲規則の保証付き化（D10）の基礎。
- **[RAPS21]** Angelopoulos, Bates, Malik & Jordan. "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR 2021.
  — 主張: 画像分類向けの予測集合構成（RAPS）。集合サイズ=1で確定・≥2で委譲＋secondary付与という保証付き委譲規則の道具。注意: 保証は周辺被覆であり、曖昧・困難な入力の層では被覆が崩れる報告があるため層別検証が必須。

## 4. 較正（前段の過信対策）

- **[Guo17]** Guo et al. "On Calibration of Modern Neural Networks." ICML 2017. https://arxiv.org/abs/1706.04599
  — 主張: 現代NNは過信傾向で較正が崩れており、temperature scaling（1パラメータ後処理）が大半のデータセットで有効。「高確信の誤りが委譲されず素通りする」失敗モードの根拠。
- **[Gate25]** Rabanser et al. "Gatekeeper: Improving Model Cascades Through Confidence Tuning." arXiv:2502.19335. https://arxiv.org/abs/2502.19335
  — 主張: 前段を「誤答時に自信が下がる」よう調整して委譲性能を改善（委譲特化の較正）。

## 5. 人間終端への拡張（learning to defer）

- **[Mad18]** Madras, Pitassi & Zemel. "Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer." NeurIPS 2018. https://arxiv.org/abs/1711.06664
  — 主張: 人間へ委譲する学習枠組み（L2D）の導入。
- **[MS20]** Mozannar & Sontag. "Consistent Estimators for Learning to Defer to an Expert." ICML 2020. https://arxiv.org/abs/2006.01862
  — 主張: 分類器＋rejector の2関数構成に一貫サロゲート損失を与えた理論的基礎。irreducible→人手キューの定式化に対応。
- **[CasHAI25]** "Cascaded Language Models for Cost-effective Human-AI Decision-Making." arXiv:2506.11887. https://arxiv.org/abs/2506.11887
  — 主張: モデルカスケードの終端に人間を置く構成のコスト効率。「既存の閾値＋人手レビュー運用＝1段カスケード＋人間終端」という再解釈の裏付け。

## 6. 失敗モードと監査

- **[CL21]** Northcutt, Jiang & Chuang. "Confident Learning." JAIR 70:1373–1411, 2021. 実装: https://docs.cleanlab.ai/
  — 主張: out-of-sample 予測確率からラベル誤りを理論保証つきで推定。高確信帯を含む監査サンプリングの道具。
- **[NAM21]** Northcutt, Athalye & Mueller. "Pervasive Label Errors in Test Sets." NeurIPS 2021 D&B. https://arxiv.org/abs/2103.14749
  — 主張: 主要10ベンチマークのテストセットに平均3.3%以上の誤ラベル。定期監査が標準実務である根拠。

## 7. 分析駆動の設計判断（decision matrix の根拠）

- **[MLTest17]** Breck, Cai, Nielsen, Salib & Sculley. "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." IEEE Big Data 2017. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
  — 主張: 本番ML実務では28項目のテスト・監視を定義し点数化して判断する。「実装前に測るべき項目を明文化し、測定で判断する」運用の実務標準。
- **[Casc21]** Sambasivan et al. "'Everyone wants to do the model work, not the data work': Data Cascades in High-Stakes AI." CHI 2021. DOI: 10.1145/3411764.3445518
  — 主張: データ品質の軽視が下流に複利で波及する Data Cascades は実務者の92%が経験。ラベル品質分析（D6）をモデル改善より先行させる根拠。
- **[Power20]** Card et al. "With Little Power Comes Great Responsibility." EMNLP 2020. https://arxiv.org/abs/2010.06595
  — 主張: 検定力不足の実験が蔓延しており、小さいテストセットではSOTA比較の大半が検出力不足になる。評価セット規模（D7）を検定力・CI幅から設計する根拠。
- **[CP34]** Clopper & Pearson. "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial." Biometrika 26(4):404–413, 1934.
  — 主張: 二項比率の正確信頼区間。ゴールドセットのクラス別サンプル数とCI幅の対応計算（D7）の原典。
- **[ECE15]** Naeini, Cooper & Hauskrecht. "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI 2015.
  — 主張: Expected Calibration Error（ECE）の定式化。較正要否判断（D3）の指標定義。

## 8. グラフ最適化・自己改善（M7 / D9 の根拠）

- **[Swarm24]** Zhuge et al. "GPTSwarm: Language Agents as Optimizable Graphs." ICML 2024 (Oral). https://proceedings.mlr.press/v235/zhuge24a.html
  — 主張: エージェント＝計算グラフとして統一記述し、ノード（プロンプト）最適化とエッジ（接続構造）最適化を自動化する枠組み。「グラフエンジニアリング」の代表定式化。
- **[ADAS24]** Hu, Lu & Clune. "Automated Design of Agentic Systems." arXiv:2408.08435. https://arxiv.org/abs/2408.08435
  — 主張: メタエージェントがコードとして新しいエージェント設計を発明・改良していく探索枠組み。
- **[AFlow25]** Zhang et al. "AFlow: Automating Agentic Workflow Generation." ICLR 2025 (Oral). https://arxiv.org/abs/2410.10762
  — 主張: ワークフロー最適化を「コード表現されたワークフロー空間の探索問題」として定式化し、MCTSと実行フィードバックで反復改良。Stage3サブワークフローの予算制約付き探索（M7副題・D4自動化）の直接の型。周辺にMaAS（arXiv:2502.04180、supernet探索）等の後続。
- **[ProTeGi23]** Pryzant et al. "Automatic Prompt Optimization with 'Gradient Descent' and Beam Search." EMNLP 2023. https://arxiv.org/abs/2305.03495
  — 主張: 誤り事例のミニバッチから自然言語「勾配」を生成し、その逆方向にプロンプトを編集、ビームサーチ＋バンディットで探索。rulebook自動改訂の基本機構。
- **[GEPA25]** Agrawal et al. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning." arXiv:2507.19457. https://arxiv.org/abs/2507.19457
  — 主張: 実行軌跡への自然言語内省で高水準ルールを学習し、GRPO比最大+19pt・ロールアウト最大1/35、MIPROv2を10pt超上回る。DSPy統合済み。M7の第一候補実装。
- **[ACE25]** Zhang et al. "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models." arXiv:2510.04618. https://arxiv.org/abs/2510.04618
  — 主張: Generator / Reflector / Curator の3役でコンテキストを「進化するプレイブック」として増分更新し、brevity bias と context collapse を構造化差分更新で防ぐ。ラベルなし・実行フィードバックのみで適応。境界事例集の維持規律（丸ごと書き換え禁止）の根拠。
- 注記: これらの公表成果は推論・コーディング課題中心で、知覚分類での証拠は薄い。採否は本リポジトリの実測（D9）で判断する。

## 9. 訓練不要適応（ゼロショットとFTの中間段）

- **[Tip22]** Zhang et al. "Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification." ECCV 2022. https://arxiv.org/abs/2207.09519
  — 主張: few-shot集合から key-value キャッシュを構築し、特徴検索でCLIPの事前知識を更新する訓練不要の適応。学習必須手法に匹敵。シルバーラベルをキャッシュ化する低コスト中間段（M6前段）の候補。

## 10. 分類体系・アノテーション・階層活用（taxonomy.md の根拠）

- **[CHiLS23]** Novack, McAuley, Lipton & Garg. "CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets." ICML 2023. https://arxiv.org/abs/2302.02551
  — 主張: 抽象クラス名の下にサブクラスを置き、子でスコアリングして親へ集約すると追加学習なしに zero-shot 精度が上がる。Stage1 の葉スコア→ブランチ max 方式（D11）の型。
- **[Poincare17]** Nickel & Kiela. "Poincaré Embeddings for Learning Hierarchical Representations." NeurIPS 2017. https://arxiv.org/abs/1705.08039
  — 主張: 階層構造は双曲空間に低次元で埋め込める。タクソノミー距離を学習信号に使う際の表現の選択肢。
- **[KG21]** Hogan et al. "Knowledge Graphs." ACM Computing Surveys 54(4), 2021. https://arxiv.org/abs/2003.02320
  — 主張: ナレッジグラフの総説。taxonomy.ttl（TBox）＋画像レコード（ABox）という本リポジトリの構造の位置づけ。
- **[KGAT19]** Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." KDD 2019. https://arxiv.org/abs/1905.07854
  — 主張: 店舗・ジャンル・メニュー等へエンティティを拡張した際の下流利用例。
- **[TIGER23]** Rajput et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023. https://arxiv.org/abs/2305.05065
  — 主張: RQ-VAE の残差量子化で階層的 Semantic ID を作る。埋め込みクラスタリングによるサブクラス発見（T10）と同型。
- **[WiSE22]** Wortsman et al. "Robust fine-tuning of zero-shot models (WiSE-FT)." CVPR 2022. https://arxiv.org/abs/2109.01903
  — 主張: zero-shot 重みと fine-tune 重みの補間で分布シフト頑健性を保ったまま精度を上げる。M6 の基本手順。
- **[LiT22]** Zhai et al. "LiT: Zero-Shot Transfer with Locked-image text Tuning." CVPR 2022. https://arxiv.org/abs/2111.07991
  — 主張: 画像塔を固定しテキスト塔のみ学習する対照チューニング。小規模 gold での微調整の選択肢。
- **[LoRA21]** Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. https://arxiv.org/abs/2106.09685
  — 主張: 低ランク差分での微調整。数千枚規模 gold でのフル微調整回避（M6）。
- **[SigLIP2-25]** Tschannen et al. "SigLIP 2." arXiv:2502.14786. https://arxiv.org/abs/2502.14786
  — 主張: sigmoid 損失の画像テキストエンコーダ。ペア独立のため、タクソノミー距離による負例重み付けを batch-softmax なしに書ける（M6）。
- **[LeGrad24]** Bousselham et al. "LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity." arXiv:2404.03214. https://arxiv.org/abs/2404.03214
  — 主張: ViT の帰属可視化。誤爆画像の「どこに反応したか」をルールの根拠として記録する道具。
- **[SKOS09]** W3C. "SKOS Simple Knowledge Organization System Reference" (Recommendation, 2009). https://www.w3.org/TR/skos-reference/ ／ "SKOS Primer." https://www.w3.org/TR/skos-primer/
  — 主張: 統制語彙の標準語彙（prefLabel / altLabel / definition / scopeNote / broader / changeNote）。taxonomy.yaml の項目名の出典（T7）。
- **[Hedden22]** Hedden, H. *The Accidental Taxonomist*, 3rd ed. Information Today, 2022.
  — 主張: 分類体系設計の実務原則（is-a のみ、単一分割基準、相互排他、網羅）。§2 設計原理の出典。

## 11. 対象外・非料理画像の検出（G0 / D14 の根拠）

分類スコアだけでは「用意されたクラスの中では food が最も近い」としか言えず、非料理画像を識別できない。G0 は料理画像の特徴分布からの距離（kNN / Mahalanobis++）が分類スコア（MSP）より多くの誤混入を検出できるかを、P0（ゲートなし）／P1（food 候補のみ再判定）／P2（全画像ゲート）の比較で実測する。以下はスコア定義・検出器・評価プロトコル・データ源の出典。

### スコアと検出器（主構成 8 = E1 × {MSP, kNN, Mahalanobis++, C3} × {P1, P2}）

- **[MSP17]** Hendrycks & Gimpel. "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks." ICLR 2017. https://arxiv.org/abs/1610.02136
  — 主張: 最大 softmax 確率を誤分類・OOD 検出の基準線とする。本リポの P0 / P1 が依拠する分類スコア基準。
- **[Energy20]** Liu, Wang, Owens & Li. "Energy-based Out-of-distribution Detection." NeurIPS 2020. https://arxiv.org/abs/2010.03759
  — 主張: energy score E(x) = −T·logsumexp(f/T) は softmax より OOD と ID の分離が良い。G0 ではアブレーション。
- **[OE19]** Hendrycks, Mazeika & Dietterich. "Deep Anomaly Detection with Outlier Exposure." ICLR 2019. https://arxiv.org/abs/1812.04606
  — 主張: 外部の負例（outlier）に対して既存クラス上の一様分布を目標とする補助損失（λ=0.5）を加えると未知の外れにも汎化する。副次構成 C1（5-way＋OE）の定義。
- **[MCM22]** Ming, Cai, Gu, Sun, Li & Li. "Delving into Out-of-Distribution Detection with Vision-Language Representations." NeurIPS 2022. https://arxiv.org/abs/2211.13445
  — 主張: CLIP のクラスプロンプト類似度に softmax（τ=1）を取った最大値（Maximum Concept Matching）で zero-shot OOD 検出。CLIP 系の画像テキストエンコーダに MSP を適用したものであり、本リポの MSP 検出器の実体。τ=0.01 はアブレーション。
- **[kNN22]** Sun, Ming, Zhu & Li. "Out-of-Distribution Detection with Deep Nearest Neighbors." ICML 2022. https://arxiv.org/abs/2204.06507
  — 主張: L2 正規化した特徴の k 近傍距離による非パラメトリック検出。正規化が本質的で、分布仮定を置かない。本リポの kNN 検出器（k=50、tune_fit バンク）。
- **[MahaPP25]** Mueller & Hein. "Mahalanobis++: Improving OOD Detection via Feature Normalization." ICML 2025 (PMLR 267). https://arxiv.org/abs/2505.18032
  — 主張: 特徴ノルムのばらつきがガウス仮定を壊す。L2 正規化後にクラス別平均＋共有共分散を推定すると 44 モデルで一貫して改善。本リポの Mahalanobis++（リッジ 1e-6）。素の Mahalanobis はアブレーション。

### ロジット系スコアと合成型（analysis/d14_farood_score_memo.md の出典。G0 主構成には含まれない）

- **[MLS22]** Hendrycks, Basart, Mazeika et al. "Scaling Out-of-Distribution Detection for Real-World Settings." ICML 2022. https://arxiv.org/abs/1911.11132
  — 主張: 多クラス・大規模設定では最大ロジット（MaxLogit）が MSP を上回る。softmax 正規化が捨てるロジットの絶対量（共通成分）を保持する基準線。
- **[ViM22]** Wang, Li, Feng & Zhang. "ViM: Out-Of-Distribution with Virtual-logit Matching." CVPR 2022. https://arxiv.org/abs/2203.10807
  — 主張: 特徴空間でしか見えない OOD とロジット空間でしか見えない OOD が併存する。特徴主部分空間からの残差ノルムを仮想ロジットとしてロジットに連結（= Energy ＋ 残差）する合成型の代表。距離信号を補完信号として足す構成の型。
- **[DML23]** Zhang & Xiang. "Decoupling MaxLogit for Out-of-Distribution Detection." CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Decoupling_MaxLogit_for_Out-of-Distribution_Detection_CVPR_2023_paper.html
  — 主張: MaxLogit は cos 類似度×特徴ノルムに分解でき、両者の重みを分離調整すると改善する。
- **[Fort21]** Fort, Ren & Lakshminarayanan. "Exploring the Limits of Out-of-Distribution Detection." NeurIPS 2021. https://arxiv.org/abs/2106.03004
  — 主張: 事前学習 transformer（ViT）特徴上では Mahalanobis 系の距離検出が強い。凍結 CLIP 特徴で距離が負ける場合は実装差（正規化等）を疑う根拠。
- **[RMD21]** Ren, Fort, Liu et al. "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection." arXiv:2106.09022. https://arxiv.org/abs/2106.09022
  — 主張: クラス条件付き Mahalanobis から全体分布への距離を引く相対 Mahalanobis（RMD）。事前学習特徴と相性が良い。

### 評価プロトコル

- **[OpenOOD23]** Zhang et al. "OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection." arXiv:2306.09301（DMLR 採録）. https://arxiv.org/abs/2306.09301
  — 主張: AUROC / FPR@95 を標準指標とし、OOD を near / far に区分して報告する。区分は ID との意味差・難易度に基づく便宜的なものであり、本リポでは収集画像を near / far と決めつけず距離で実測する（型ごとに「遠く分離できる／料理側に入り込む」を記録）。

### 追加比較の候補（G0 では実施しない。A / D / E は design.md §6 G0 節）

- **[NegLabel24]** Jiang et al. "Negative Label Guided OOD Detection with Pretrained Vision-Language Models." ICLR 2024. https://arxiv.org/abs/2403.20078
  — 主張: WordNet から ID ラベルと意味的に遠い負ラベル約 1 万語を選び、正負ラベルとの類似度比でスコア化する。追加比較 A。
- **[CSP24]** Chen, Gao & Xu. "Conjugated Semantic Pool Improves OOD Detection with Pre-trained Vision-Language Models." NeurIPS 2024. https://arxiv.org/abs/2410.08611
  — 主張: 負ラベルの語彙プールを上位語×形容詞の組合せに拡張して被覆を上げる。A の派生。
- **[AdaNeg24]** Zhang & Zhang. "AdaNeg: Adaptive Negative Proxy Guided OOD Detection with Vision-Language Models." NeurIPS 2024. https://arxiv.org/abs/2410.20149
  — 主張: 固定の負ラベルではなく、テスト時に実際の OOD 画像分布に合わせた負プロキシを適応的に構成する。A の派生。
- **[NegRefine25]** Ansari, Wang & Xiong. "NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection." ICCV 2025. https://arxiv.org/abs/2507.09795
  — 主張: 負ラベル集合から ID の下位概念・固有名詞を除去し、複数ラベルに合致する画像のスコアリングを改良する。A を実装する場合の除外規則の根拠。
- **[EOE24]** Cao et al. "Envisioning Outlier Exposure by Large Language Models for Zero-Shot OOD Detection." ICML 2024. https://arxiv.org/abs/2406.00806
  — 主張: LLM に ID クラスに近い外れクラス名を生成させ（β=0.25、L=500）、実画像なしで outlier exposure を模す。文献値は Food-101 等が対象で、本リポの期待値には使わない。
- **[WOODS22]** Katz-Samuels, Nakhleh, Nowak & Li. "Training OOD Detectors in their Natural Habitats." ICML 2022. https://arxiv.org/abs/2202.03299
  — 主張: 運用環境の未ラベル wild 混合（ID＋OOD）を制約付き最適化で活用し、外れ候補を抽出して検出器を学習する。追加比較 D の原型。
- **[SCONE23]** Bai et al. "Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection." ICML 2023. https://arxiv.org/abs/2306.09158
  — 主張: wild データで OOD 汎化と OOD 検出を同時に扱う。D の系譜。
- **[SAL24]** Du, Fang, Diakonikolas & Li. "How Does Unlabeled Data Provably Help Out-of-Distribution Detection?" ICLR 2024. https://arxiv.org/abs/2402.03502
  — 主張: 未ラベル混合から勾配空間の特異値分解で外れ候補を分離し、二値分類器を学習する（SAL）。誤り保証つき。追加比較 D（簡略版・独自実装）の直接の型。
- **[Medix25]** Abbas, Falahati, Goli & Amiri. "Medix: Out-of-Distribution Detection from Unlabeled Wild Data via Robust Gradient Statistics." TMLR. https://arxiv.org/abs/2510.06505
  — 主張: SAL の勾配統計を中央値ベースで頑健化し、混入率が高い wild 集合でも安定させる。D の改良候補。
- **[ReGuide25]** Kim, Lee & Hwang. "Reflexive Guidance: Improving OoDD in Vision-Language Models via Self-Guided Image-Adaptive Concept Generation." ICLR 2025. https://arxiv.org/abs/2410.14975
  — 主張: VLM 自身が画像適応的に近傍・外れの概念候補を生成し、それを手がかりに OOD 判定する。追加比較 E（灰色域の VLM 再判定）。
- **[LLMVLM25]** Lee, Chen & Wu. "Harnessing Large Language and Vision-Language Models for Robust Out-of-Distribution Detection." ACML 2025 (PMLR 304). https://arxiv.org/abs/2501.05228
  — 主張: LLM 生成の負概念と VLM の視覚特徴を組み合わせ、far と near の OOD 検出を両立させる。A と E の接続例。

### 評価用データ源（外部の非料理候補・ソース対照。ライセンスはファイル単位で記録）

- **[OpenImages20]** Kuznetsova et al. "The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale." IJCV 2020. https://storage.googleapis.com/openimages/web/index.html
  — 主張: 人手検証済みの画像レベルラベルからクラス（MID）別に候補抽出できる。画像は CC BY 2.0、注釈は CC BY 4.0。層 A（人形・像、遊具、看板、人物・動物、乗り物）とソース対照正例（Food / Dish）の主な源。
- **[COCO14]** Lin et al. "Microsoft COCO: Common Objects in Context." ECCV 2014. https://cocodataset.org
  — 主張: 物体注釈（面積・iscrowd・カテゴリ）で person / vehicle 主体の画像を条件抽出できる。注釈は CC BY 4.0、画像は Flickr のライセンス ID を画像ごとに記録する。
- **[Places365-17]** Zhou, Lapedriza, Khosla, Oliva & Torralba. "Places: A 10 Million Image Database for Scene Recognition." TPAMI 2017. http://places2.csail.mit.edu/
  — 主張: シーンカテゴリ（street 等）の画像源。G0 では任意（時間が無ければ落とす）。
- **[DTD14]** Cimpoi, Maji, Kokkinos, Mohamed & Vedaldi. "Describing Textures in the Wild." CVPR 2014. https://www.robots.ox.ac.uk/~vgg/data/dtd/
  — 主張: テクスチャ画像。補助 far（指標は food 誤受理率のみ）の任意源。
- **[CORD19]** Park et al. "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing." NeurIPS 2019 Workshop on Document Intelligence. https://huggingface.co/datasets/naver-clova-ix/cord-v2
  — 主張: 領収書画像 1,000 枚（CC BY 4.0、HF `naver-clova-ix/cord-v2`、revision 固定）。補助 far の源。
- **[Commons]** Wikimedia Commons. "Commons:Licensing" および MediaWiki API（`imageinfo` の `extmetadata`）. https://commons.wikimedia.org/wiki/Commons:Licensing
  — 主張: ファイル単位のライセンス・帰属を API から取得できる。カテゴリ探索は非再帰（depth ≤ 2）、User-Agent ポリシー遵守、500px サムネイル。人形・マスコット・食品サンプル・玩具の食品など near-food 候補の源。
- **[YelpTOU23]** Yelp. "Yelp Dataset Terms of Use" (2023-07-07). https://s3-media0.fl.yelpcdn.com/assets/srv0/engineering_pages/f64cb2d3efcc/assets/vendor/Dataset_User_Agreement.pdf
  — 主張: 学術利用限定、Data の再配布・表示禁止、公開前の Yelp 審査。design.md §2 の対応（ID 行を git に置かない、reports は集計値のみ、結果 PR は draft）の根拠。規約の版と DL 日を reports フッタに記録する。
- **[YelpBlog15]** Yelp Engineering Blog. "How We Use Deep Learning to Classify Business Photos at Yelp." 2015-10-19. https://engineeringblog.yelp.com/2015/10/how-we-use-deep-learning-to-classify-business-photos-at-yelp.html
  — 主張: Yelp の 5 クラス写真ラベル（food / drink / menu / inside / outside）は CNN 分類器が付与したもの。本リポで Yelp ラベルを「名目ラベル」として扱い、棄却側の全数目視と通過側の無作為抽出で確認済み損失を別に推定する根拠。

- 注記: 上記文献の性能値（AUROC・FPR@95 等）はベンチマーク固有であり、本リポの期待値・判定基準には使わない。判定基準は事前登録（docs/g0_nonfood_plan.md、D14）の数値のみ。

## 引用規約

reports/ 内で数値主張をする際は上記キー（[VJ01] 等）で引用する。本ファイルにない主張を導入する場合は、一次文献を確認してからここに追記する。
