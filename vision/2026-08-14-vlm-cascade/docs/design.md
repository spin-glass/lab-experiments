# design.md — lab-vlm-cascade

本書がリポジトリ設計の正本。実装との矛盾が生じた場合は本書を改訂してから実装を変更する。

## 0. 目的

曖昧クラスを含む画像分類（例: inside と food の境界事例）を、コスト・可用性・ルール変更即応性の3制約下で運用する「カスケード＋判定層＋監査ループ」アーキテクチャの一般実装と数値検証。

- 公開ポートフォリオであると同時に、GCP実務へ部品単位で転用可能なリファレンス実装とする
- 設計パラメータは事前に固定せず、定義済みの分析で決定する（§8 decision matrix）。同一分析を実データで再実行できる移植性を持つ
- データは公開の Yelp Open Dataset のみ。数値はすべて本リポジトリで実測したものだけを掲載する
- 特定企業の事例・数値・環境情報は一切含めない

## 1. アーキテクチャ

```
photos ─▶ Stage1: encoder zero-shot ─▶ Stage2: decision layer ─▶ 確定 (clear / rule_resolved)
              │ probs, margin              │ 低margin・指定confusion pair のみ
              ▼                            ▼
        predictions (parquet)        Stage3: VLM escalation ─▶ 確定 or irreducible
                                           │
offline: 監査ループ (cleanlab / VLM合議 / κ定点観測) ◀──┘
single source (語彙):     taxonomy/taxonomy.yaml (versioned) ─▶ クラス定義 / 階層 / 葉プロンプト
single source (ポリシー): rulebook.md (versioned)            ─▶ 優先規則 / 判定表 / VLMプロンプト
```

- Stage1: 画像エンコーダのゼロショット分類。プロンプトは taxonomy の第3階層（葉）単位で持ち、ブランチ内 max で第1階層（food / non-food）と第2階層（運用5クラス）へ集約する。第1階層の確信度 `branch_margin = S_food − S_non` を Stage2 に渡す。二値の直列ハードゲートは置かない（詳細は [taxonomy.md](taxonomy.md) §5、採否は D11 / D12）
- Stage2: しきい値＋優先順位ルールの決定的判定層。モデルなし、コードとconfigのみ。品質規則（ボケ等）と運用写像（unjudgeable の扱い）もこの層の管轄
- Stage3: 委譲スライスのみ VLM（Gemini）で構造化出力判定。自己一致または2モデル一致で確定、不一致は irreducible
- 監査: バッチでラベル・予測を検品（cleanlab、VLM合議、gold κ）
- 対象外画像の扱い: 対象外（5 クラスのどれにも該当しない）画像は残余クラスへ流さず、処理方式の比較で決める。P0 ゲートなし／P1 Stage1 が food と予測した画像のみ距離信号で再判定／P2 Stage1 前の全画像ゲート。P2 は T4/D12 が退ける直列ハードゲートの構造であり、G0 では誤ゲートの回収不能コストを実測する比較対象。採否と方式は D14（G0）

## 2. データ

- Yelp Open Dataset photos: 約20万枚、label ∈ {food, drink, menu, inside, outside}
- 取得は公式ページから各自ダウンロード。**画像・生データはリポジトリ非同梱**（scripts/ に手順とチェックサム検証のみ）。README等にYelp画像を掲載しない。同梱の利用規約PDFの確認を M0 の最初の工程とする
- サンプリング: work set = 層化 25,000枚 / eval set = 凍結 2,000枚（work と非交差。クラス別400目安＋natural分布スライス併設）。eval は photo_id リストのハッシュで凍結し、W&B reference artifact で版管理
- **分割と用途の分離**（M0 で確定し、以後のマイルストーンで変更しない）。探索に使ったデータで探索結果を評価することを構造的に防ぐ
  - `tune`: work set 内の層化部分集合（目安 5,000枚、規模は D7 で決める）。しきい値探索（Optuna / Vizier）、較正（D3）、conformal の calibration、プロンプト文の修正、判定表の調整など**パラメータを fit する処理はすべてここだけ**で行う
  - `work`（tune を除く残余）: cleanlab の out-of-sample 予測、food 予測の precision 監査（M1）、パイロットアノテーション（M4）、蒸留の soft label（M6）、M7 holdout rotation の fold 元
  - `eval`: 最終比較と reports の数値報告のみ。eval 上で fit したパラメータは採用しない。eval の結果を見て設計を変えた場合は、その事実を意思決定ログに明記する（暗黙の再利用を禁止）
  - 分割は photo_id の非交差に加え、同一店舗の写真が分割をまたがないよう店舗ID（`business_id`）単位で行う。非交差は M0 のテストで検証し、`eval_sets` に役割・定義・ハッシュを記録する
  - gold（M4）は監査対象の work 側と、κ・内容精度の評価用に eval 側の両方へ付与する。gold の付与は分割の用途を変えない（eval に gold が付いても探索には使わない）
- ラベルの扱い: Yelp付与ラベルは「来歴の混在した既存ラベル」とみなす（実務で頻出する状況の一般形）。gold は M4 で目視スポットチェックした部分集合のみとする
- gold の記録粒度: 運用クラス（第2階層）のみを記録し、第1階層（food / non-food）は導出する。内容（content_label）・判定可能性（quality）・判定者の迷い（boundary_flag）・属性（food_visible）は別フィールドで直交に持ち、残余クラスへ混載しない。アノテーションは 50〜100 枚のパイロット → ガイドライン改訂 → 本番の2パス制（[taxonomy.md](taxonomy.md) §4）
- 外部の公開画像データ（Open Images, Wikimedia Commons, COCO, CORD 等）を非料理の評価候補として使う（G0）。画像は非同梱。公開 ID・sha256・ファイル単位のライセンスを `configs/ood_sources.lock` に記録し、reports に画像を載せない
- Yelp 規約への対応: Data（photo_id・business_id・caption・ラベル行）は git に置かず、sha256 と config・seed のみを置く。reports は集計値のみ。公開前の扱い（学術利用該当性・公開前審査の要否）は結果 PR を draft にして確認後にマージする。規約の版と DL 日を reports のフッタに記録する
- eval の natural スライス（上記の層化 2,000 枚と併設する側）は G0 の要件（eval の food ≥ 3,000。1% 損失 = 30 枚）を満たす規模に拡大する（D7）
- 分割キーは `business_id` と近重複群（sha256 完全一致 → pHash Hamming ≤ 8 → 埋め込み余弦 ≥ 0.95）の連結成分。同一店舗・同一群は分割を跨がない
- 外部画像の前処理は Yelp と同分布（長辺サイズ・JPEG 品質・EXIF 除去）に統一する

## 3. 技術スタックと実務転用マップ

| 本リポジトリ | GCP実務での対応物 |
|---|---|
| uv + Python 3.12 | 同一 |
| OpenCLIP / SigLIP (Hugging Face) | 同一コンテナを Vertex AI Custom Job で実行 |
| DuckDB + parquet 三層 (raw / core / marts) | BigQuery 三層 (ds_raw / ds_core / ds_marts) |
| Gemini API 直呼び | Vertex AI Gemini Batch inference（50%割引） |
| Tracker fan-out: W&B Free ＋ Vertex AI Experiments へ二重送出 | Vertex AI Experiments（承認が出れば W&B を追加送出） |
| Optuna（本探索）＋ managed Vizier（無料枠内のパリティ検証1本） | Vertex AI Vizier（後退線 Optuna） |
| cleanlab | 同一 |
| reports/ の md 自動生成 | 同一（GitHub共有・PRレビュー） |

**runs正本パターン（fan-out）**: 実験記録の正本はローカル parquet の runs テーブルに置き、Tracker アダプタ（config: `tracking: [wandb, vertex]`、任意の組合せ・空も可）が各バックエンドへ投影する。実験ごとにバックエンドを排他切替すると履歴が分断され横断比較が壊れるため、**切替ではなく二重送出**とする。どのバックエンドが不通でも run 本体は完走する。同一 run を W&B と Vertex Experiments の両UIで比較できること自体が、トラッカー可搬性の実証＝本リポジトリの主張の一つ。

**Vertex 利用条件**: Vertex AI Experiments は run 自体に追加課金がなくメタデータ保存の従量のみ。Vertex AI TensorBoard はログ保存 $10/GiB/月 の課金があるため本リポジトリでは**有効化しない**（学習曲線は W&B と素の TensorBoard で代替）。managed Vizier は無料枠（月100トライアル）内でのみ使用し、超過する探索は Optuna（または OSS 版 google-vizier）で行う。

**W&B 利用条件**: Free プランは個人プロジェクト限定のため、本リポジトリ（個人ポートフォリオ）に限定して使用する。W&B には画像・生データをアップロードしない（メトリクス・小テーブル・reference artifact のみ）。ストレージ 5GB/月の制限内に収まり、「画像はストレージ参照のみ」という実務パターンと同型になる。

## 4. 単一ソース: taxonomy と rulebook

単一ソースは「語彙」と「ポリシー」の2ファイルに分け、手書きの重複定義を禁止する。語彙＝何があるか、ポリシー＝どう写像するか。

**taxonomy（語彙）** — `taxonomy/taxonomy.yaml`。設計の詳細は [taxonomy.md](taxonomy.md)。
- 内容: 2階層＋プロンプト用サブクラスの is-a 木。SKOS 借用項目（prefLabel / altLabel / definition / scopeNote / broader / changeNote）と構造制約（排他・網羅・導出式）
- 版管理: `scheme.version`（`taxonomy_version`）。runs・reports に刻印する
- 生成: `taxonomy/taxonomy_build.py` が検証のうえ `prompts.json`（Stage1 葉プロンプト）・`label_master.csv`（アノテーション用マスタ、第1階層は導出列）・`taxonomy.ttl` を生成。検証に1件でも失敗したら生成しない

**rulebook（ポリシー）** — `rulebook.md`。
- 構成: (a) 分類意図＝閲覧者がどのタブにその写真を期待するか (b) クラス定義の参照（本文は taxonomy の definition。重複記載しない） (c) 曖昧ペアごとの優先順位ルール＝多ファセット空間から単一ラベルへの射影。**アノテーション開始前に確定** (d) 境界事例の文例 (e) 品質規則と運用写像（unjudgeable の扱い、Laplacian 分散・OCR 被覆率等の決定的信号）
- 版管理: semver（`rulebook_version`）。runs・reports に必ず刻印する
- 生成: `scripts/gen_from_rulebook.py` が taxonomy の `prompts.json` と rulebook から Stage1 クラスプロンプト・Stage2 判定表（YAML）・Stage3 プロンプト（taxonomy の definition / scopeNote ＋ rulebook の優先規則・否定・例外を全部渡す）を生成する
- 判定出力: `primary`, `secondary`（任意）, `flag ∈ {clear, rule_resolved, irreducible}`

結合キーは taxonomy の `node_id` に統一する。プロンプト・ルールの対象ノード・評価スライス・レポートを同一 id で結合する。

## 5. スキーマ（DuckDB / parquet、BigQuery互換型のみ使用）

- `labels(photo_id, label_source, label, provenance, ts)` — label は第2階層クラス。gold は `gold_annotations` から `content_label` を射影して `label_source='gold'` で投入
- `gold_annotations(photo_id, content_label, quality, boundary_flag, food_visible, secondary, annotator, guideline_version, ts)` — 第1階層は持たない（導出のみ）。`content_label ∈ 第2階層 ∪ {unjudgeable}`、`quality ∈ {ok, degraded, unjudgeable}`
- `scope_annotations(photo_id, source, content, operational_truth, subject_type, boundary_flag, quality, dup_group_id, role, sampling_prob, license_short, attribution, source_url, sha256, annotator, guideline_version, ts)` — `gold_annotations` の拡張（G0）。`content ∈ {food, drink, non_food, mixed, unjudgeable}` は被写体の内容（属性）。`operational_truth ∈ 第2階層 ∪ {out_of_scope}` は運用上の正解で、第1階層は `operational_truth` から導出し `content` からは導出しない。`out_of_scope` はアノテーション専用の残余値で Stage1 のクラスではない。迷いは `boundary_flag`（クラスへ流さない）。`subject_type ∈ {doll_character_statue, amusement_tourist, signage_decor_goods, person_animal, vehicle_street, near_food_nonfood, texture_document_screen, food_contrast, other}`、`source ∈ {yelp, open_images, commons, coco, places365, cord, dtd, screenspot}`（1 画像 1 源）、`role ∈ {tune, work, eval}`。`annotator` は仮名 ID。ライセンス列は外部源のみ
- `predictions(run_id, photo_id, stage, probs_json, margin, branch_margin, primary, secondary, flag, model_id, cost_tokens, ts)` — `probs_json` は第2階層の確率（葉スコアを含めてよい）、`branch_margin = S_food − S_non`
- `runs(run_id, git_sha, taxonomy_version, rulebook_version, eval_set_id, model_id, config_json, metrics_json, wandb_url, ts)`
- `eval_sets(eval_set_id, role, photo_ids_hash, definition, created_at)` — `role ∈ {tune, work, eval}`（§2 の用途分離を記録。runs は評価に使った `eval_set_id` を刻印する）
- `ood_sets(ood_set_id, source, subject_type, role, n, ids_sha256, lock_path, created_at)` — 非料理評価集合の定義（G0）
- `ood_scores(run_id, photo_id, source, mode, method_id, encoder_id, score, role, fold_axis, fold_id, ts)` — 検出器スコア。向きは大＝対象内（`mode ∈ {P1, P2}`）
- `gate_decisions(run_id, photo_id, mode, method_id, threshold_id, decision, stage1_pred, final_label, ts)` — 動作点ごとの棄却／通過と最終ラベル
- 推移閉包用に `taxonomy_nodes(node_id, level, parent_id, ancestors)` を marts に持ち、「food の全子孫」を1クエリで取れるようにする

評価は2系統: 内容精度（`quality='unjudgeable'` を除外）と運用出力精度（運用写像による導出込み）。

## 6. マイルストーン

### M0 セットアップ
uv環境、データ取得と規約確認、サンプリングと tune / work / eval の分割確定（§2。`business_id` 単位の非交差テストを含む）と eval 凍結、三層テーブル初期化、Tracker アダプタ実装と W&B project／GCP プロジェクト（Vertex AI Experiments）の初期化、`tracking` config の縮退動作テスト（wandb のみ／vertex のみ／none で完走すること）、Gemini の現行モデルIDを確認して config にピン留め（"latest" 系エイリアス禁止）。`taxonomy/` の導入（taxonomy.yaml v0.1、build / viz の検証通過、負のテスト、Makefile または pre-commit への束ね）。G0 に必要な最小構成（環境・taxonomy v0.1・埋め込み・分割・tracking）を先行し、rulebook 未導入の間は reports のフッタに rulebook_version を「未導入」と記す。
完了条件: `reports/m0_setup.md`（データ統計・クラス分布・分割定義と非交差検証結果・eval定義・taxonomy_version）

### G0 非料理画像の距離信号による識別（M0 後。M1 と並行可）
中心の問い: 分類スコアでは food に見える非料理画像が、料理画像の特徴分布（kNN／Mahalanobis 距離）からは外れているか。正しい料理の棄却を同じ水準に揃えたとき、距離は分類スコアより多くの誤混入を検出できるか。
比較する処理方式（既定なし）: P0 ゲートなし（Stage1 argmax）／P1 Stage1 = food の画像のみ検出器で再判定（他クラスの出力は不変。T4 と整合）／P2 Stage1 前の全画像ゲート（T4/D12 が退ける直列ハードゲート。他クラス損失・層 B の誤棄却というコストを実測する比較対象）。
主構成 8 = E1 × {MSP, kNN, Mahalanobis++, C3 二値ヘッド} × {P1, P2}。E1 = open_clip ViT-B-16 / datacomp_xl_s13b_b90k（hf_hub laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K）、副次 E2 = google/siglip2-base-patch16-224。revision ハッシュは B-a でピン留め。主構成 E1×kNN×P1 は α=0.05、残り 7 構成は Holm。
主指標: 非料理→food 誤受理率、誤混入削減率（層 A 合算・型別、および層 B。両層とも必須評価で別々に判定）、距離の追加価値 Δ（同じ料理損失に較正した MSP との同一画像上の対応比較）、料理損失（名目／確認済み。確認済みの正解は operational_truth = food）、層 B の P2 通過率（§7）。動作点は tune_cal の「正しく food」のスコア昇順 1% 位置（副: 0.5%、2%）。
成立条件: 層 A eval の合算 n_before ≥ 40（型別は ≥ 10 のみ評価）、層 B eval の n_before_B ≥ 30。未満の層は評価不能とし「今回の標本では削減効果を評価する誤分類件数が不足した」と報告する。主検定は (1) ゲートが有効か: H0: p ≤ 0.25 の片側正確二項検定、(2) 距離の追加価値: MSP との不一致対に対する片側正確 McNemar と Δ の片側 95% 下限（画像単位ブートストラップ）。帰結は層ごとに採用候補／保留／不採用／評価不能を出し、総合は両層の弱い方（`docs/g0_nonfood_plan.md` §8）。
検証命題: 投稿されそうな非料理の food 誤分類に対し、料理損失を抑えながら、距離信号が food スコア閾値（MSP）を上回るか。(i) ゲートが有効か、(ii) 距離を使う追加価値があるか、(iii) P1 で足りるか（P1 − P2 の削減率差の非劣性、許容差 10 pt）の 3 つに分けて答える。C 系の LOTO / LOSO は held-out の eval 行のみで評価し、fold ごとの閾値は共通の tune_cal で取り直す。
事前登録: `docs/g0_nonfood_plan.md`（凍結タグ `g0-freeze-v1`。以後の変更は逸脱記録のみ）。eval は最後の 1 回だけ使う。VLM（Gemini）は G0 では使わない（評価ラベルにも灰色域再判定にも使わない）。追加比較（NegLabel、SAL 簡略版、灰色域 VLM、母集団の混入率）は G1 以降。
完了条件: `reports/g0_nonfood_gate.md`（冒頭に記述分析の要約・主指標・帰結表。フッタに Yelp 規約の版・DL 日を追記）

### M1 ゼロショットベースライン
最初にプロンプト埋め込みの余弦類似度行列（画像不要・テキストのみ）で兄弟プロンプトの過接近や誤爆吸収プロンプトの food 側偏りを診断し、プロンプト文を修正する。そのうえでエンコーダ2種以上でクラス別 P/R/F1、混同行列、margin 分布を、フラット5クラス方式と階層方式（葉スコア→ブランチ max）の両方で測る（D11）。food 予測画像 100 枚程度の precision 監査で誤爆の出所内訳を取り、誤爆吸収クラスの初期列挙を taxonomy v0.x に反映する（D13）。
検証命題: 混同ペア（inside×food 等）が定量的に特定できる。階層方式がフラット方式に対して food precision を落とさずに改善するか。
完了条件: `reports/m1_baseline.md`（類似度行列、方式別比較、precision 監査の内訳表）＋W&B run

### M2 判定層と risk–coverage
risk–coverage 曲線からクラス別・ペア別しきい値を設計する。探索・較正・conformal の calibration はすべて tune のみで行い、採用したしきい値の risk–coverage と委譲率は eval で報告する（§2）。第1階層の `branch_margin` 閾値方式と直列二値ハードゲート方式を同一 eval で比較し、誤ゲートによる food 取りこぼしを実測する（D12）。しきい値探索の本体は Optuna（キャッシュ済み確率に対する純関数評価、数千トライアル）。加えて managed Vizier で同一探索空間のスタディを**無料枠内（≤100トライアル）で1本**実行し、Optuna との到達解を突き合わせて work-parity を記録する。さらに conformal prediction（split conformal / RAPS）による予測集合ベースの委譲規則——集合サイズ=1なら確定、≥2なら委譲＋secondary付与——をしきい値方式と比較し、**層別（クラス別・margin帯別）の被覆充足**まで検証する（D10。周辺被覆のみの保証は曖昧例の層で崩れうるため）。優先順位ルール v1.0 を判定層に実装しユニットテストを付ける。
検証命題: カバレッジを X% 委譲すると残存誤り率が Y% 下がる、の定量化。
完了条件: `reports/m2_decision_layer.md`（risk–coverage 図、採用しきい値、委譲率）

### M3 カスケード vs 全量VLM
委譲スライスのみ Gemini 構造化出力で判定し、全量VLM（同一eval）と精度・コストを実測比較。呼び出しは config 上限内（既定: eval 2,000枚＋委譲分、MAX_ESCALATION=5,000）。
あわせて**プロンプト4条件アブレーション**を委譲スライスで実施し、D4の下位判断「Stage1確率をStage3に渡すか」を実測で決める:
- A: 縮小画像＋ペア限定ルールのみ（既定案）
- B: A＋margin帯ラベル（low / mid のみ、数値なし）
- C: A＋Stage1全確率
- D: ペア限定なしの5クラス判定＋全確率
測定: 精度、2モデル一致率、irreducible率、コスト。Cで一致率だけ上がり精度が伸びない場合はアンカリング（合議の独立性喪失）の兆候としてAを維持し、Cで精度が有意に上がる場合のみ確率注入を採用する。
検証命題: カスケードが全量VLM比コスト Z% で同等精度圏に入る。
完了条件: `reports/m3_cascade_vs_vlm.md`（精度・コスト表、1万枚あたり試算、アブレーション表）

### M4 ラベル監査
cleanlab（out-of-sample 予測確率）＋VLM 2系合議で既存ラベルの疑義を検出し、上位N件を目視スポットチェックして的中率（precision@N）を測定。gold 部分集合で人×VLM の κ を算出。gold 作成は `gold_annotations` スキーマ（§5）で 50〜100 枚のパイロット → boundary_flag の収穫とガイドライン改訂 → 本番の2パス制。パイロット結果で未決2件（food×drink 同格時の優先順位、unjudgeable の運用出力先。D13）を確定し、taxonomy v0.2 と rulebook に反映する。VLM 合議のシルバーは葉で不一致でも第1階層で一致していれば二値シルバーとして保持する。
検証命題: 既存ラベルの誤り率推定と検出的中率。
完了条件: `reports/m4_label_audit.md`（パイロットの boundary_flag 率・unjudgeable 率を含む）

### M5 ルール変更即応
rulebook を v1.1 に改版（例: inside×food の優先規則変更）→ 再学習なしで判定表・プロンプトを再生成 → eval 再実行 → ラベルフリップ数と影響差分レポートを自動生成。
検証命題: ルール変更が判定層・プロンプト層だけで即日反映できる。
完了条件: `reports/m5_rulebook_change.md`（版間diff、フリップ表）

### M6（任意）蒸留
VLM合議の soft label で linear probe → LP-FT / LoRA、WiSE-FT 補間。階層を負例の難度マップとして使い（第1階層境界＝food 葉 × 誤爆吸収クラスを hard negative、ブランチ内兄弟を細粒度用）、SigLIP 系の sigmoid 損失ではタクソノミー距離で負例を重み付ける。委譲率の低下を実測。クラス内画像埋め込みのクラスタリング（k-means / 残差量子化）によるサブクラス候補の発見と、LeGrad 等の帰属可視化による誤爆根拠の記録も本Mで行う（[taxonomy.md](taxonomy.md) §5）。
完了条件: `reports/m6_distill.md`

### M7（任意）self-improving rulebook loop
M4の監査不一致集合を訓練信号に、GEPA / ProTeGi 型のテキスト最適化で rulebook 改訂案を自動生成 → frozen eval＋**holdout rotation**（work 側の fold を回転させ過適合を検出。eval は凍結のまま最終報告にのみ使う）で評価 → 影響差分レポート → **人の承認ゲート** → M5 の再評価パイプラインへ接続する閉ループを実装する。境界事例集の節は ACE 流の増分キュレーション（丸ごと書き換え禁止、構造化差分更新のみ）で維持する。副題として、Stage3 昇格サブワークフローの予算制約付き構造探索（単発VQA／自己一致／2モデル一致／OCRツール併用／記述→ルール適用の候補空間を AFlow 流に探索）で D4 を自動化する。

**承認ゲートの実装（PRを承認プリミティブとする）**: 改訂案はブランチ `rulebook/vX.Y-rc` ＋自動PR（本文＝影響差分レポート）として提出する。CIが gen_from_rulebook → frozen eval＋holdout rotation を再実行し、結果をPRコメント＋status check化（基準未達はマージ不能）。`rulebook.md` に CODEOWNERS＋required approving review（保護ブランチ、publicリポは無料）を設定し、**マージ＝承認確定**とする。蒸留・一括再ラベル等の高コスト下流処理は Actions environments の required reviewers で二段目ゲートを設ける。Slack は通知・催促・議論面（`/github subscribe <owner>/<repo> pulls reviews`、scheduled reminders）に限定し、承認行為は置かない（diffの見えない場所での承認と監査痕跡の分散を避ける）。

**頻度制御（PR洪水の防止）**: ①改訂案生成は週次監査バッチに従属させ、イベント駆動にしない。②発火条件: ペア別不一致の蓄積が閾値超のときのみ生成を起動。③改善ゲート: holdout rotationで最小改善幅（CI下限>0）を満たさない候補は**PR化せず自動棄却**し、runsにログのみ残す。④レート上限: 1サイクル最大1PR、かつ `max_open_prs=1`（未処理の自動PRがある間は新規PRを作らない）。頻度は成り行きでなく発火閾値で制御するガバナンス変数とし、PR発生頻度・却下率・レビュー所要時間をD9の計測項目に含める。

**変更の3層ポリシー**: Tier 1＝境界事例集への追記（最頻・最軽。週次PRに同梱）／Tier 2＝ルールの新規追加（minor bump、同梱可）／Tier 3＝既存ルールの修正・優先順位変更（既存判定のフリップ発生源。**単独PR＋フリップ表必須**、フリップ率>基準で目視サンプルレビュー。D8と接続）。

**撤退条件**: 却下率の高止まり、またはレビュー負担が改善価値を上回ると計測された場合はループを停止し、手動M5運用（人が起点、機械は影響分析のみ）へ戻す。この条件をD9の採否基準に含める。

検証命題: 自動改訂が同一入力に対する人手改訂と比べて、改善幅・コスト・安定性で優位か（D9）。
完了条件: `reports/m7_self_improving_loop.md`（人手改訂との対照、holdout間の安定性、探索コスト、承認ゲート通過率、PR頻度・却下率）

## 7. 指標定義

- selective risk / coverage: 判定層が確定した集合における誤り率と、その集合の被覆率
- escalation rate: Stage3 へ委譲した割合
- cost/image: 入力トークン＝258（両辺≤384pxへリサイズ）＋プロンプト分。モデル単価×トークンで算出し、レポートに必ず概算を記載
- macro-F1 / クラス別 P/R/F1、混同行列
- κ: gold 部分集合における人×VLM の一致度（監査モデルの較正指標）
- 非料理→food 誤受理率: 層 A（対象外の非料理）のうち、ゲートを通過し food として確定した枚数の割合。ゲート前（P0）と後で報告（G0）
- 誤混入削減率: r / n_before（n_before = P0 が food 確定した層 A 枚数、r = そのうちゲートが除去した枚数）。合算と型別（G0）
- 料理損失（名目／確認済み）: 名目 = eval の「正しく food」（ラベル food ∧ P0 = food）のうち棄却された率（Clopper–Pearson）。確認済み = R_true / (R_true + N_pass·q̂)（R_true: 棄却された名目 food のうち目視で operational_truth = food、q̂: 通過側無作為抽出の operational_truth = food 率。比推定量なので区間は画像単位ブートストラップ）（G0）
- 層 B の削減率・P2 通過率: 対象内の非料理（層 B）について、削減率 = P0 が food と判定した層 B のうちゲートが除去した率（層 A と同じ形の必須評価）、P2 通過率 = 全画像ゲートを通過した率（片側 95% 下限で判定）（G0）
- 距離の追加価値 Δ: 同じ料理損失（1%）に較正した距離検出器と MSP の、同一画像上の削減率の対応差 Δ = (b − c) / n_before（b: 距離のみ除去、c: MSP のみ除去）。検定は不一致対の片側正確 McNemar、区間は画像単位ブートストラップ。層 A・層 B で別々に算出（G0）

## 8. 設計判断を決める分析（decision matrix）

設計パラメータは事前に決め打ちしない。以下の各判断は、対応する分析の結果と判断基準によって決定し、reports/ に**意思決定ログ**（基準値・実測値・採否）として記録する。運用は二段構えとする: まず Yelp データで分析の方法論と基準を確立し（各M）、次に**同一の分析スクリプトを実データ（業務環境の予測ログ・ラベル）で再実行して本番の設計判断を下す**。この「方法論は公開データで実証済み、判断は自データの数値で行う」構図そのものが、アーキテクチャ提案の説得の中核になる。

| ID | 設計判断 | 決める分析 | 判断基準の型 | Yelp側 | 実データ側 |
|---|---|---|---|---|---|
| D1 | カスケード採否（vs 単独強化 / 全量VLM） | margin分布の偏り、risk–coverage、コスト分岐点 | 委譲率≤X%で目標精度、全量VLM比コスト<Y% | M1–M3 | 予測ログに同一スクリプト |
| D2 | しきい値の粒度（大域/クラス別/ペア別） | 粒度別の risk–coverage 比較 | 同カバレッジでの誤り削減幅 > 保守コスト | M2 | 同左 |
| D3 | 較正の要否 | reliability diagram、ECE、高確信帯の誤り率 | ECE>基準、または高確信誤りが残存誤りの過半 | M2 | 同左 |
| D4 | Stage3構成（単発/自己一致/2モデル一致、確率注入の有無） | 委譲スライスでの一致率×正解率×コスト、プロンプト4条件アブレーション | 精度とコストのパレート比較。確率注入は精度向上時のみ採用（一致率のみ上昇なら不採用） | M3 | 昇格ログ |
| D5 | irreducible境界（三値フラグ閾値） | VLM合議一致度と人間κ（gold二重判定）の突合 | 不一致帯の人間κ<基準なら分布保持が妥当 | M4 | gold二重判定 |
| D6 | 既存ラベルの再利用可否 | cleanlabスコア分布、来歴別疑義率、スポットチェック的中率 | 来歴別疑義率<基準の部分集合のみ再利用 | M4（Yelpラベルを来歴混在の代役に） | ラベル来歴検証と併走 |
| D7 | 評価セット規模と分割（tune / work / eval の用途分離、§2） | Clopper–Pearson CI幅、ブートストラップ分散、分割の非交差検証 | クラス別±CI目標を満たす最小n、探索と報告のデータが非交差 | M0で設計、M1で実測検証 | 同一計算 |
| D8 | ルール変更ゲート | 版間フリップ率、ペア別影響 | フリップ率>基準で人手レビュー必須 | M5 | rulebook運用 |
| D9 | rulebook自動最適化の採否 | 同一不一致集合を入力にした自動改訂 vs 人手改訂の対照比較、PR頻度・却下率・レビュー所要時間の計測 | 人手比の改善幅/コスト、holdout rotation間の安定性、承認ゲート通過率。却下率高止まり・レビュー負担>改善価値なら撤退（手動M5へ復帰） | M7 | 監査ループに接続 |
| D10 | 委譲規則のconformal化の採否 | 予測集合ベース委譲 vs しきい値委譲の比較、層別被覆検証 | 同一被覆での委譲率減、全層で被覆充足 | M2 | 同左 |
| D11 | Stage1 の階層スコアリング採否（フラット5クラス vs 葉スコア→ブランチ max） | プロンプト類似度行列、方式別の P/R/F1・混同行列・margin 分布 | food precision を落とさず macro-F1 または food F1 が改善 | M1 | 同一スクリプト |
| D12 | 第1階層の委譲方式（branch_margin 閾値 vs 直列二値ハードゲート） | 同一 eval での risk–coverage、誤ゲートによる food 取りこぼし率 | 同一委譲率で残存誤りが少なく、取りこぼしが基準以下 | M2 | 同左 |
| D13 | taxonomy / rulebook の未決事項（誤爆吸収クラスの列挙、food×drink 同格時の優先順位、unjudgeable の運用出力先） | food 予測の precision 監査内訳、パイロットアノテーションの boundary_flag 率・unjudgeable 率 | 誤爆源の累積被覆率、同格衝突頻度、unjudgeable 率で運用先を決定 | M1・M4 | 自データで再監査 |
| D14 | 非料理画像の除外方式（ゲートなし／food 候補のみ再判定／全画像ゲート）と検出器の候補選定、距離信号の追加価値 | 同一料理損失（1%）での誤混入削減率の片側正確二項検定（Holm）を層 A・層 B で別々に、MSP との同一画像上の対応比較（McNemar＋対応差の区間）、料理損失の確認済み上限、層 B の P2 通過率、LOTO/LOSO（held-out eval のみ）、P1 − P2 の非劣性、ソース対照 | 層ごとの採用候補／保留／不採用の帰結表と「追加価値あり」ラベル（docs/g0_nonfood_plan.md §8）。結論は候補選定まで | G0 | 同一スクリプトを自データの予測ログへ |

**移植性要件**: 分析コードは `analysis/` に判断ID付きで置く。SQLはDuckDB/BigQueryの両方で動く書き方に限定し（方言依存の関数を避ける）、接続とテーブル名はconfig注入とする。これにより実データ側では接続先の差し替えだけで同一分析が走る。

**自己改善ループの統制（D9 / M7）**: 自動最適化は次の4つの失敗モードを前提に統制する。(a) frozen evalへの過適合 — holdout rotationを必須とし、分割間で改善が再現しない改訂は棄却。(b) Goodhart化 — 指標だけ上がる無意味な規則を防ぐため、rulebook改訂は必ず人の承認ゲートを通し、意味的妥当性をレビューする。(c) 再現性 — 最適化の軌跡・シード・使用モデル版をrunsに記録する。(d) 証拠の偏り — ワークフロー自動最適化の公表成果は推論・コーディング課題中心で知覚分類での証拠は薄いため、採否は本リポジトリでの実測（D9）でのみ判断する。なお、パイプラインのコード自体をエージェントが書き換える自動ML工学系（AIDE等）は、統制・監査可能性と衝突するため検討のうえ不採用とする。

## 9. 非スコープ

リアルタイム推論、Web UI、Yelp データの再配布・画像表示、特定企業事例への言及。README の最終整備（英語版含む）は M5 完了後に別途判断する。
