# g0_nonfood_plan.md — G0 非料理画像の距離信号による識別（事前登録）

本書は G0 の事前登録文書。凍結（A3）以後に記入するのは §13（結果）と §14（逸脱記録）のみで、それ以外の節（§0〜§12、§15）は変更しない。変更が必要になった場合は §14 逸脱記録に残してから変更する。設計の正本は [design.md](design.md)（§1 の P0/P1/P2、§6 G0 節、§8 D14）、分類体系は [taxonomy.md](taxonomy.md)、クラス定義は `taxonomy/taxonomy.yaml`。本書と design.md が矛盾したら design.md を先に改訂する。

## 0. 版と状態

| 版 | 日付 | 状態 | 内容 |
|---|---|---|---|
| v1 草案 | 2026-09-07 | A1 | 中心の問い・定義・評価集合・検出器・指標・判定規則・目視設計を固定内容として記載。§11〜§14 は空 |
| v1.1 草案 | 2026-09-07 | A1 | ユーザレビュー反映: MSP との対応比較（追加価値）、層 B の必須評価、散布図の境界を閾値に、P1/P2 の非劣性判定、LOTO/LOSO を eval のみ・fold ごと較正、確認済み損失の正解を operational_truth に統一し区間法を判定量ごとに固定、成立条件の文言 |
| v1 調査記入 | （A2 で記入） | A2 | §11 データ調査を記入。ゲート性能は見ない（P0 の誤受理数のみ例外） |
| v1 凍結 | （A3 で記入） | A3 | §12 凍結を記入し git タグ `g0-freeze-v1` を打つ。以後に記入するのは §13・§14 のみ |
| v1 結果 | （実験後に記入） | E | §13 結果と §14 逸脱記録を記入。design.md §8 D14 の意思決定ログへ反映 |

状態の遷移: 草案（A1）→ データ調査を記入（A2）→ 凍結（A3）→ 結果を記入（D の実行後、E で反映）。

## 1. 背景と中心の問い

既存のクラス分類は対象外画像を識別できない。クラス別 accuracy は集計値であり、個々の入力に対する判定信号にならない。分類スコアが高くても「用意されたクラスの中では food が最も近い」というだけで、その画像が対象内である保証はない。そこで、対象外画像を分類の前段ではなく **分類とは別の対象内判定** として扱い、その価値を「料理への混入をどれだけ減らし、正しい料理画像を何枚失うか」で判定する。判定信号には料理画像の特徴分布からの距離（kNN／Mahalanobis）を用い、分類スコアとの差を実測する。

**中心の問い**: 分類スコアでは food に見える非料理画像が、料理画像の特徴分布からは外れているか。正しい料理の棄却を同じ水準に揃えたとき、距離は分類スコアより多くの誤混入を検出できるか。

**主結論の形**: 投稿されそうな非料理の food 誤分類に対し、料理損失を抑えながら、距離信号が food スコア閾値を上回るか。結論は 2 つに分ける。(i) **ゲートが有効か**: 各構成の誤混入削減率が絶対基準を満たすか。(ii) **距離を使う追加価値があるか**: 同じ料理損失に較正した MSP（food スコア閾値）と距離検出器を同一画像上で対応比較し、削減率の差が正か。(ii) は (i) と独立に判定し、MSP が距離を上回る場合は「ゲートは有効だが距離の追加価値はない」と結論する。

| 観測結果 | 意味 |
|---|---|
| food スコアが高いが料理の参照集合から遠い | 距離による追加判定で救える可能性 |
| food スコアが高く参照集合にも近い | その埋め込み・距離だけでは救いにくい |

前提としないこと:

- 「far」を前提にしない。収集画像は「非料理の評価候補」であり、遠いかどうかは距離で実測し、型ごとに「遠く分離できる群」と「料理側に入り込む群」に分ける（§6）。far / near の語は、先行研究のベンチマーク区分（OpenOOD の far-OOD 区分など）を指すときと、それに対応させた補助集合の名称（§3.5「補助 far」）にのみ使い、収集画像の性質の記述には使わない
- VLM（Gemini）は G0 に含めない。評価ラベルにも灰色域の再判定にも使わない。VLM を用いる比較は §10 の追加比較として G0 の外に置く
- 既定の方式を置かない。P0 / P1 / P2（§4）は比較対象であり、採否は D14（§8）で決める

## 2. 定義

### 2.1 対象内と対象外

- **対象内**: 飲食店の写真として投稿されたと見なせる画像。運用真値は第 2 階層 5 クラス（food / drink / menu / inside / outside）のいずれか
- **対象外（out_of_scope）**: 飲食店の文脈が無い画像。アノテーション専用の残余値であり、Stage1 のクラスではない。taxonomy.md §3 の「残余 other は拡張オプション」を scope 用途に限定して採用する
- **層 A**: `operational_truth = out_of_scope` の非料理画像。主評価の対象
- **層 B**: 対象内の非料理画像。`content ∈ {non_food, mixed}` かつ `operational_truth ∈ {inside, outside, menu}`（店内の人形・店頭の像・看板・食事中の人物など）。P2 で棄却してはならず、P1 では food から外すべき画像。今回の誤分類（店内の置物・店頭の装飾）はこの層でも起こるため、**food に誤分類された層 B の除去率は層 A と並ぶ必須評価**とし（§8）、層ごとに帰結を出す。任意の比率で層 A と合算しない

### 2.2 属性の定義と値域

| フィールド | 値域 | 意味 |
|---|---|---|
| content | food / drink / non_food / mixed / unjudgeable | 被写体の内容（属性）。分割ではない |
| operational_truth | 第 2 階層 5 クラス ∪ {out_of_scope} | 運用上の正解。迷いは boundary_flag に置き、クラスへ流さない（T5 / T6） |
| subject_type | doll_character_statue / amusement_tourist / signage_decor_goods / person_animal / vehicle_street / near_food_nonfood / texture_document_screen / food_contrast / other | 被写体の型。層 A の型別集計と LOTO の fold 軸 |
| boundary_flag | bool | 判定者の迷い。付けたら仮ラベルのまま先へ進む |
| quality | ok / degraded / unjudgeable | 判定可能性（taxonomy.md §4） |

第 1 階層（food / non-food）は `operational_truth` から導出し、`content` からは導出しない（T2、整合条件 level1 = parent(level2)）。`content = food` かつ `operational_truth = menu`（料理写真入りメニュー）のような組合せは正当であり、矛盾ではない。

### 2.3 記録スキーマ

design.md §5 の `gold_annotations` の拡張として定義する。

| テーブル | 列 |
|---|---|
| scope_annotations | photo_id, source, content, operational_truth, subject_type, boundary_flag, quality, dup_group_id, role, sampling_prob, license_short, attribution, source_url, sha256, annotator, guideline_version, ts |
| ood_sets | ood_set_id, source, subject_type, role, n, ids_sha256, lock_path, created_at |
| ood_scores | run_id, photo_id, source, mode, method_id, encoder_id, score, role, fold_axis, fold_id, ts |
| gate_decisions | run_id, photo_id, mode, method_id, threshold_id, decision, stage1_pred, final_label, ts |

- `source ∈ {yelp, open_images, commons, coco, places365, cord, dtd, screenspot}`（1 画像 1 源）。`role ∈ {tune, work, eval}`（外部集合も同じ役割名）
- `mode`: ood_scores では {P1, P2}（P0 は検出器スコアを持たない）、gate_decisions では {P0, P1, P2}（P0 は棄却なし）。`fold_axis ∈ {none, loto, loso}`
- `annotator` は仮名 ID。ライセンス列（license_short / attribution / source_url）は外部源のみ
- Yelp 側の ID 一覧は `data/`（git 管理外）に置き、リポジトリには sha256・config・seed のみ。外部源は公開 ID・画像 sha256・ライセンスを `configs/ood_sources.lock` としてコミットする

## 3. 評価集合

### 3.1 Yelp（対象内、母集団）

- 母集団 = photos.json の全画像（ラベル欠損を含む）。サイズは A2 でクラス別枚数・ラベル欠損率・近重複率・画像サイズ分布を確認してから確定する（クラス別 5,000 を前提にしない。menu は少ない可能性がある）
- 分割キー = `business_id` と dup 群（§3.7）の連結成分のハッシュ。同一店舗・同一群は分割を跨がない（テストで保証。design.md §2）
- 役割と最小 n:

| 役割 | 用途 | 最小 n |
|---|---|---|
| tune_fit | kNN バンク・クラス平均と共有共分散・ヘッド学習 | food ≥ 2,000、他クラス各 ≥ 500 |
| tune_cal | 動作点（閾値）の決定 | food ≥ 3,000 |
| work | 開発、追加手法の発動判断、記述分析の予備 | 残余 |
| eval | 最終報告のみ | food ≥ 3,000 |

- tune_fit と tune_cal は design.md の `tune` の内部分割。eval の food ≥ 3,000 は 1% 損失 = 30 枚を数えられる規模として D7 の要件に含める（natural スライスの拡大）
- **人手確認済み部分集合（eval）**: 主 8 構成（§5）の 1% 動作点で棄却された名目 food は全数を目視する（P2 で 300 枚を超える構成は (b) 不合格の根拠として目視を省略してよい）。通過側の名目 food から無作為 200（標準 300）を目視し、名目ラベルの精度 q̂ を推定する。名目損失（Yelp ラベル基準）と確認済み損失（§7.2）を分けて報告する。「真の FRR」「gold」とは呼ばない

### 3.2 層 A: 対象外の非料理（主評価）

候補は既存ラベルで抽出し、画像単位の目視で `operational_truth = out_of_scope` を確定する。MID とカテゴリは config に固定する。

| subject_type | 候補抽出 | 目標 n（縮小案） | 標準案 |
|---|---|---|---|
| doll_character_statue（人形・キャラクター・像） | Open Images: Doll /m/0167gd, Teddy bear /m/0kmg4, Mascot /m/0153bq, Statue /m/013_1c, Sculpture /m/06msq, Figurine /m/01xmqj。Commons: Stuffed animals, Plush toys, Mascot costumes, Kigurumi, Statues of mascots, Maneki neko, Shigaraki ware Tanuki statues | 120 | 150 |
| amusement_tourist（遊具・観光施設） | Open Images: Amusement park /m/010jjr, Ferris wheel /m/017rgb, Roller coaster /m/010l12, Carousel /m/01pwdc, Playground /m/01h_1n, Swing (Seat) /m/02y929, Fountain /m/0220r2。Commons: Ferris wheels / Carousels（国別 subcat を depth ≤ 2） | 120 | 150 |
| signage_decor_goods（看板・装飾・雑貨） | Open Images: Billboard /m/01knjb, Poster /m/01n5jq, Balloon /m/01j51, Lantern /m/01jfsr, Christmas tree /m/025nd, Signage /m/0bkqqh。Commons: Toy balloons, Paper lanterns in Japan, Christmas decorations, Shop windows with Christmas decorations | 120 | 150 |
| person_animal（人物・動物） | Open Images: Person / Dog / Cat（Food 正例を含む画像は除外）。COCO val2017: person 主体（最大 box 面積比 ≥ 0.2、iscrowd 除外、person ≤ 3、食関連カテゴリ {44, 46–61, 67} を含む画像は除外）、animal。Commons: Selfies taken indoors, Group selfies | 120 | 150 |
| vehicle_street（乗り物・街並み） | Open Images: Car / Bus / Train / Building / Skyscraper。COCO: vehicle 主体。Places365 street 等は任意 | 120 | 150 |
| near_food_nonfood（食に近い非料理、別集計） | Commons: Food samples（食品サンプル）配下, Toy food, Empty plates, Dirty dishes, Leftovers。Open Images: Plate / Bowl / Tableware かつ Food 明示 negative | n_min 80 | 150 |

- 「店頭のマスコット・キャラクター像」型は doll_character_statue に含める。固有名は書かない
- near_food_nonfood の帰属例: 玩具の食品・料理のイラスト・食品パッケージ単体・料理形のオブジェは out_of_scope。店頭の食品サンプル陳列は outside、料理写真入り掲示は menu として層 B へ
- 5 型の合算目標 600（標準 750）に near-food を別集計で加える。標準案への拡大は凍結前に限る

### 3.3 層 B: 対象内の非料理

- Yelp のキャプション検索で候補を抽出する。語彙は config に固定（mascot, statue, doll, figure, character, sign, mural, decor, balloon, christmas, dog, cat, car, selfie, "me and", toy 等）
- 目視で `content ∈ {non_food, mixed}` かつ `operational_truth ∈ {inside, outside, menu}` を確定し、subject_type を付与する。外部候補のうち目視で inside / outside / menu と判定されたものもここへ入れる
- 目標: 合算 200（標準 300）

### 3.4 ソース対照正例（交絡診断）

対象内は全て Yelp、層 A は全て外部源のため、ゲートが「非 Yelp」を検出してしまう恐れがある。外部源の料理画像（Open Images Food /m/02wbm・Dish /m/02q08p0、Commons の料理カテゴリ）100（標準 150）を目視で `content = food` に確定し、同じゲートに通す。

- 報告: 対照正例の棄却率と Yelp food の損失の差。埋め込み上の Yelp vs 外部の 2 値 AUROC（tune で logistic）
- **ソース検出フラグ**: 対照正例の棄却率が Yelp food の損失の 3 倍以上かつ 5pt 以上高い構成は「ソース検出の疑い」とし、採用候補になれず保留（§8）

### 3.5 補助 far（別集計）

CORD-v2（CC BY 4.0、レシート文書）200。指標は food 誤受理率のみ。DTD / ScreenSpot は任意で、時間が無ければ落とす。subject_type は texture_document_screen。先行研究のベンチマーク区分（OpenOOD の far-OOD 区分など）に対応させた参照点として置くもので、層 A の主評価には含めない。実際に遠いかどうかは §6 で実測する。

### 3.6 候補抽出規則

- 候補は目標の 2 倍を決定的順序（config に固定した並びと seed）で抽出する
- 型内でクラス別上限 40 枚を設け、単一クラス（Flower・Person・Poster 等）の独占を防ぐ
- 目視確定後に 120 未満（near-food は 80 未満）なら、同一源から 1 回だけ追加抽出する。それでも届かなければ、その型は「評価不能」として記録し、主検定の合算からは外さない（型別評価のみ不能）
- 抽出の歩留まり（看板→outside、人物→mixed 等）は §11 に記録する

### 3.7 近重複と前処理

- 近重複は全源横断で判定する: sha256 完全一致 → pHash Hamming 距離 ≤ 8 → E1 余弦類似度 ≥ 0.95。union-find で連結成分を作り `dup_group_id` とする
- 群は role を跨がない（同一群の画像は同じ role に入る。テストで保証）
- 外部画像の前処理は Yelp と同分布に統一する。A2 で測る Yelp の長辺サイズと JPEG 品質に合わせてリサイズ・再エンコードし、EXIF を除去する。数値は config に固定し §12 に記録する

### 3.8 外部集合の role 配分

外部集合（層 A・対照正例・補助 far）は subject_type × source で層化し、`tune` 30%（C 系の負例学習）／`work` 10%（発動判断・記述分析の予備）／`eval` 60%（報告）に配分する。層 B（Yelp 由来）は Yelp の分割に従う。

## 4. 処理方式

比較対象であり、既定は置かない。

| 方式 | 内容 | 位置づけ |
|---|---|---|
| P0 ゲートなし | Stage1 argmax のみ（分類スコアのみ） | 基準 |
| P1 food 候補のみ再判定 | Stage1 = food の画像だけを検出器で再判定。棄却 = food として確定しない。他クラスの出力は不変 | 中心の問いに直接対応。T4（誤ゲートは回収不能）と整合し、design.md の branch_margin 方式と両立する |
| P2 全画像ゲート | Stage1 の前で棄却。棄却された画像はどのクラスにも確定しない | T4 / D12 が退ける直列ハードゲートの構造。G0 では「他クラス損失・層 B の誤棄却というコストを実測する比較対象」 |

Stage1 は taxonomy v0.1 のフラット 5 クラス zero-shot（素朴版のプロンプトで固定）。階層方式（D11）は G0 では使わない。

## 5. 検出器

凍結埋め込み上で動作する。主構成 8 = E1 × {MSP, kNN, Mahalanobis++, C3} × {P1, P2}。

### 5.1 主構成

| 検出器 | P2 スコア（対象内全体） | P1 スコア（food） | 固定値 |
|---|---|---|---|
| MSP（MCM） | max softmax、τ = 1 | p(food) | τ = 0.01 はアブレーション |
| kNN | L2 正規化、tune_fit 全体バンク、k = 50、負の k 近傍距離 | food バンク | k ∈ {10, 200} は work のみ |
| Mahalanobis++ | L2 正規化 → クラス平均＋共有共分散（リッジ 1e-6）、クラス間 min 距離 | food クラス距離 | 素の Mahalanobis はアブレーション |
| C3 二値ヘッド | logistic（ID tune_fit vs 外部 tune）、C = 1.0、class_weight balanced | food vs 負例 | 副次: C1・C2 |

### 5.2 副次構成（記述のみ。採否に使わない）

| 構成 | 内容 | 評価範囲 |
|---|---|---|
| MSP τ = 0.01 | 温度アブレーション | eval |
| 素の Mahalanobis | L2 正規化なし、共有共分散 | eval |
| kNN k ∈ {10, 200} | k 感度 | work のみ |
| C1 5-way + OE | 既存 5 クラス上の一様目標、λ = 0.5 | eval |
| C2 6-way | 第 6 クラス = 対象外 | eval |
| E2 | エンコーダ差の再現用。時間が無ければ落とす | eval |

### 5.3 エンコーダ

| ID | モデル | 備考 |
|---|---|---|
| E1 | open_clip `ViT-B-16` / `datacomp_xl_s13b_b90k`（hf_hub `laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K`） | 主。revision ハッシュは B-a でピン留め |
| E2 | `google/siglip2-base-patch16-224` | 副次。revision ハッシュは B-a でピン留め |

- float32。`model_key` に前処理と dtype を含める。config の "latest" / "main" は拒否する（テスト）
- スコアの向きは大 = 対象内で統一する。OOD スコアは s_ood = −s_in
- 検出器は fit 前の score 呼び出しで raise し、seed で決定的に動く。P1 / P2 の両モードを同一クラスで持つ

### 5.4 C 系の交差検証

C1 / C2 / C3 は外部負例で学習するため、学習に使った型・源への過適合を切り分ける。

- **LOTO**（leave-one-type-out）: subject_type 6 fold（層 A 5 型 + near_food_nonfood）。held-out 型の学習負例（tune）と発動判断（work）を外して再学習し、**held-out 型の eval 行だけ**で評価する（work で見た画像を最終判定に入れない）
- **LOSO**（leave-one-source-out）: source ∈ {open_images, commons, coco} の 3 fold。型は pooled。評価は held-out 源の eval 行だけ
- **fold ごとの較正**: C 系は fold ごとに再学習するとスコアの尺度が変わるため、各 fold の閾値は共通の Yelp tune_cal（正しく food の 1% 損失位置、§7.1）で取り直す
- **報告**: LOTO と LOSO は別々に報告する。LOTO 内では各 eval 画像がちょうど 1 回 held-out になるので fold 合算は二重計上にならない（LOSO も同様）。LOTO と LOSO を足し合わせない
- fold の n_before < 10 は評価不能。型と源の交絡表を併記する

## 6. 記述分析

判定規則（§8）の前に必ず出す。判定に使う数値の背後にある分布を先に見せる。

- **ヒストグラム**: 料理参照集合（tune_fit の food）からの kNN 距離・Mahalanobis++ 距離。料理（eval food）vs 層 A（型別）vs 層 B
- **散布図と 4 象限表**: 横軸 p(food)、縦軸 距離（kNN 距離。Mahalanobis++ は別図）。象限の境界は、横軸 = MSP 方式（P1）の tune_cal で決めた food 閾値 t_msp、縦軸 = 距離方式（P1）の閾値 t_dist（いずれも §7.1 の 1% 動作点）。argmax は同じ p(food) でも他クラスの値で結果が変わるため境界に使わず、P0 = food かどうかは点の形で別表示する。「高スコア×遠い／高スコア×近い／低スコア×遠い／低スコア×近い」の枚数を料理・層 A・層 B で表にし、P0 で food に誤分類された層 A・層 B は別に出す
- **分布差の要約**: AUROC（主）、重なり係数、KL（ビン幅は事前固定し §12 に記録）
- 型ごとに「遠く分離できる」「料理側に入り込む」を、型別のヒストグラムの重なりと AUROC（eval food vs 型）に基づいて記録する。これは記述であり判定には使わない

## 7. 動作点と指標

### 7.1 動作点

- tune_cal の「正しく food」（Yelp ラベル food かつ P0 = food）のスコアを昇順に並べ、1% 位置の値 t（food 3,000 なら 31 番目の最小値）を動作点とする。同値の扱いは実装で固定し、テストで解析解と突き合わせる
- 棄却は s < t。達成した cal 損失（tune_cal で棄却された率）を報告する
- 副: 0.5%、2%
- P2 は同一 t で他 4 クラスの FRR も報告し、採否条件（§8 (d)）に含める

### 7.2 主指標（eval）

| 指標 | 定義 | 備考 |
|---|---|---|
| 非料理→food 誤受理率 | （層 A eval で、ゲートを通過し food として確定した枚数）／（層 A eval の総数） | P0（ゲート前）と P1 / P2（ゲート後）を並記 |
| 誤混入削減率 | r / n_before。n_before = P0 が food 確定した層 A eval 枚数、r = そのうちゲートが除去した枚数 | 合算（主）と型別 |
| 料理損失（名目） | eval の「正しく food」のうち棄却された率 | Yelp ラベル基準 |
| 料理損失（確認済み） | R_true / (R_true + N_pass · q̂)。R_true = 棄却された名目 food のうち目視で **operational_truth = food** の枚数、N_pass = 通過した名目 food の枚数、q̂ = 通過側無作為 200（標準 300）の operational_truth = food 率。守るべき正解は food 出力なので content ではなく operational_truth で判定する（料理写真入りメニュー = content food ∧ operational_truth menu は「正しく food」に数えない） | 比推定量。CI は画像単位ブートストラップ（§7.4） |
| 層 B の削減率（P1 除去率） | r_B / n_before_B。n_before_B = P0 が food と判定した層 B eval 枚数、r_B = そのうちゲートが除去した枚数 | 層 A と同じ形の必須評価。高いほど良い |
| 距離の追加価値 Δ | 同一画像上で Δ = 削減率（距離検出器）− 削減率（MSP）。両者とも tune_cal の同じ料理損失（1%）に較正。層 A・層 B それぞれで算出。不一致対（距離のみ除去 b、MSP のみ除去 c）から Δ = (b − c) / n_before | 対応比較。CI は画像単位ブートストラップ（§7.4） |
| 層 B の P2 通過率 | 層 B eval のうち P2 を通過した率 | 片側 95% 下限で判定 |
| P2 の他クラス FRR | 他 4 クラスの「正しく分類」のうち P2 で棄却された率（名目） | P2 のみ |

### 7.3 副指標

- AUROC・FPR@95（P1: 陽性 = 正しく food、P2: 陽性 = 対象内全体。陰性 = 層 A eval）
- near-food 誤受理率、補助 far の food 誤受理率
- ソース対照正例の棄却率、Yelp vs 外部の 2 値 AUROC
- 埋め込み時間（枚/秒）と実行環境

### 7.4 CI の統一

判定に使う区間の計算法は判定量ごとに 1 つに固定する。

| 判定量 | 推定量の型 | 区間法（判定） | 検定 |
|---|---|---|---|
| 削減率（層 A・層 B・fold）、通過率、名目料理損失、他クラス FRR | 二項比率 | Clopper–Pearson 片側 95%（両側 90%） | 片側正確二項検定 |
| 料理損失（確認済み） | 比推定量（通過側の標本推定 q̂ を含む） | 画像単位ブートストラップ（棄却側の目視結果と通過側無作為標本を独立に再抽出、B = 2,000、パーセンタイル）片側 95% | 区間のみ |
| 距離の追加価値 Δ（距離 − MSP） | 対応のある比率差 | 画像単位ブートストラップ（同一画像の対を再抽出、B = 2,000）片側 95% 下限 | 不一致対の正確 McNemar（片側） |
| P1 − P2 の削減率差（§8.4） | 対応のある比率差 | 同上 | 非劣性（§8.4） |

- 報告用の区間は両側 95%（同じ方法）
- Yelp 側の指標には店舗（business_id）単位ブートストラップを感度分析として添付する。判定には使わない

注: 人工的に集めた層 A からは「実際の food 出力に占める非料理の割合」は推定できない。母集団の混入率は G1（§10）で扱う。

## 8. D14 判定規則

A3 で凍結する。以後は逸脱記録（§14）なしに変えない。

### 8.1 成立条件

- 層 A eval の合算 n_before ≥ 40、層 B eval の n_before_B ≥ 30。未満の層は「評価不能」とし、「今回の標本では、削減効果を評価する誤分類件数が不足した」と報告する（Stage1 の性質についての主張はしない）
- 型別（層 A）は n_before ≥ 10 の型のみ評価する
- 層 A・層 B は別々に判定する。片方が標本不足なら、その層への効果は保留とする

### 8.2 主検定

主検定は 2 系統。いずれも層 A と層 B で別々に行う。

**(1) ゲートが有効か（絶対基準）**
- 構成ごとに r ～ Binom(n_before, p)、H0: p ≤ 0.25 の片側正確二項検定
- 主構成 **E1 × kNN × P1** は α = 0.05。残りの 7 構成は Holm（m = 7）で調整する
- 同値の表現: 「調整水準での片側 Clopper–Pearson 下限 ≥ 25%」

**(2) 距離を使う追加価値があるか（MSP との対応比較）**
- 同じ料理損失（1%）に tune_cal で較正した MSP と距離検出器（kNN、Mahalanobis++、C3）を、同一の eval 画像（P0 = food の非料理）上で対応比較する。不一致対 b（距離のみ除去）・c（MSP のみ除去）に対する正確 McNemar（片側、H0: Δ ≤ 0）と、Δ の片側 95% 下限（画像単位ブートストラップ）
- 主比較 **E1 × kNN × P1 vs E1 × MSP × P1** は α = 0.05。残り 5 比較（kNN P2、Mahalanobis++ P1/P2、C3 P1/P2 それぞれ vs 同モードの MSP）は Holm（m = 5）
- MSP 自身にはこの検定は無い（MSP は「ゲートが有効か」のみ判定する）
- 副次構成（§5.2）は検定しない

### 8.3 構成ごとの帰結表（網羅・排他）

| 帰結 | 条件 |
|---|---|
| 採用候補 | 以下 (a)〜(g) のうち該当するものを全て満たす（層 A・層 B とも） |
| 不採用 | いずれかの層で削減率の片側 95% 上限 < 50%、または (b) の点推定が基準の 2 倍超（確認済み > 4% または名目 > 5%） |
| 保留 | それ以外（点推定は満たすが限界が満たさない、片方の層が標本不足、ソース検出フラグ）。**G0 では最終**。データ追加・再検定はしない。後続は G1 |
| 評価不能 | 層 A の n_before < 40 かつ層 B の n_before_B < 30（全構成共通） |

帰結は層ごと（層 A・層 B）に出し、総合の帰結は両層の帰結の弱い方とする（採用候補 > 保留 > 不採用 の順で弱い方。評価不能は保留として扱う）。「距離の追加価値」は帰結とは別のラベルとして、(g) を満たす構成に「追加価値あり」を付す。

採用候補の条件:

| 条件 | 内容 | 適用 |
|---|---|---|
| (a) | 主検定 (1) 合格かつ削減率の点推定 ≥ 50%（層 A 合算・層 B それぞれ） | 全構成 |
| (b) | 料理損失（確認済み）の片側 95% 上限 ≤ 2% かつ名目の片側 95% 上限 ≤ 2.5% | 全構成 |
| (c) | 評価可能な全型（n_before ≥ 10）で削減率の点推定 ≥ 20% | 全構成 |
| (d) | 他 4 クラスの FRR 片側 95% 上限 ≤ 3% かつ層 B 通過率の片側 95% 下限 ≥ 90% | P2 のみ |
| (e) | LOTO と LOSO のそれぞれで、held-out eval の fold 合算（各画像 1 回）の削減率の片側 Clopper–Pearson 下限 ≥ 25%。LOTO と LOSO は別々に判定し足し合わせない | C 系のみ |
| (f) | ソース検出フラグ（§3.4）なし | 全構成 |
| (g) | 主検定 (2) 合格（McNemar 片側 p が調整水準未満）かつ Δ の片側 95% 下限 > 0（層 A・層 B それぞれ） | 距離検出器（kNN・Mahalanobis++・C3）のみ。MSP は対象外 |

判定順序: 評価不能 → 不採用 → 採用候補 → 保留。同一構成・同一層が複数に該当することはない。(g) は「ゲートが有効か」の帰結に影響しない（(g) 不合格の距離検出器は「採用候補（追加価値なし）」になり得る）。

### 8.4 「P1 で足りるか」の対応比較

検出器ごとに同一 eval 層 A（P0 = food の画像）上で P1 と P2 を対応比較し、削減率の対応差 Δ₁₂ = 削減率(P1) − 削減率(P2) に対する**非劣性判定**を行う。許容差 δ = 10 pt を事前に定める。「Δ₁₂ の片側 95% 下限（画像単位ブートストラップ、§7.4）≥ −δ かつ P1 の対象内総損失 ≤ P2 の対象内総損失」を満たせば「全画像ゲート不要（P1 で足りる）」と結論する。P2 側の推定誤差は対応差の区間に含まれる。注: MSP は P0 = food の画像上で max softmax = p(food) なので、同じ較正条件では P1 と P2 の food 誤混入削減が一致し Δ₁₂ = 0 になる（不一致対は生じない）。この比較が情報を持つのは距離検出器と学習ヘッドである。層 B については P1 の除去率と P2 の通過率という別の量を見るため、この非劣性判定の対象外とする。対象内総損失 = 対象内 eval（Yelp の正しく分類された 5 クラス＋層 B）のうち、ゲートで棄却されて本来のクラスに確定しなかった枚数の割合（名目）。P1 では正しく food の棄却のみが該当し（層 B を food から外すのは損失に数えない）、P2 では他 4 クラスと層 B の棄却も含む。

### 8.5 結論の範囲

結論は採用候補集合（空集合可）と「P1 で足りるか」まで。実運用の採用は自データで同一スクリプトを再実行して判断する（design.md §8 の二段構え）。

## 9. 目視確認の設計

- **ツール**: 単一ファイルのローカルサーバ（stdlib http.server、`scripts/review_server.py`）。`data/` から画像を配信し、`data/review/*.jsonl` に追記保存する（中断再開可）
- **盲検**: source・候補ラベル・スコア・split を表示しない。表示順はシード付きシャッフル
- **2 パス制**: 1 パス目はキー入力で keep / drop / boundary。2 パス目で content・operational_truth・subject_type を付与する。確認済み料理損失（§7.2）と通過側 q̂ の判定は operational_truth = food で行う
- **ガイドライン**（外部画像の operational_truth）: 「飲食店の写真として投稿されたと見なせる場合のみ 5 クラス。飲食店の文脈が無い画像は out_of_scope」。near-food の帰属例（§3.2）を列挙する
- **パイロット**: 50 枚 → boundary_flag の収穫 → guideline_version 確定 → 本番。guideline_version は §12 に記録する
- **再判定**: 本番の 10% を 1 日以上空けて再判定し、自己一致 κ と boundary_flag 率を報告する
- **作業量（縮小案）**:

| 対象 | 目標枚数 | 候補（2 倍） |
|---|---|---|
| 層 A 5 型 | 600 | 1,200 |
| near-food | 80 | 160 |
| 層 B | 200 | 400 |
| ソース対照正例 | 100 | 200 |
| パイロット | 50 | — |
| 棄却された名目 food（全数） | 構成に依存（P1 で約 30／構成） | — |
| 通過側無作為 | 200 | — |

候補約 2,000 枚を 5〜8 秒/枚で 3〜4.5 時間。時間があれば凍結前に限り標準案（6 型×150、層 B 300、対照 150、通過側 300）へ増やす。

## 10. 追加比較（G0 では実施しない）

| ID | 内容 | 位置づけ |
|---|---|---|
| A | NegLabel（負のテキストラベルによる zero-shot OOD） | G0 の主 8 構成に対する追加比較。実施は G0 の結果を見てから判断 |
| D | SAL 簡略版（独自実装・変更点明記。適応用と評価用の画像を分離した別実験） | 同上 |
| E | 灰色域の VLM 再判定 | ゲート単体の残存誤りに追加判定の価値があると分かってから |
| G1 | 母集団の混入率（H4）。2 段抽出確率・下位スコア側を厚く・残り層 ≥ 100 | 層 A からは推定できない量。G0 から繰り延べ |

## 11. データ調査（A2 で記入）

ゲート性能は見ない。P0（Stage1 argmax のみ）の誤受理数だけを例外として計測する。

### 11.1 Yelp

| クラス | 枚数 | ラベル欠損率 | 近重複率 | 長辺サイズ（中央値／範囲） | JPEG 品質（中央値） |
|---|---|---|---|---|---|
| food | | | | | |
| drink | | | | | |
| menu | | | | | |
| inside | | | | | |
| outside | | | | | |
| 欠損 | | — | | | |

確定サイズ: tune_fit food ＝ ／ tune_cal food ＝ ／ eval food ＝ ／ 他クラス ＝ 。

### 11.2 外部（源 × 型）

| source | subject_type | 候補数 | 除去数（dedup） | 除去数（目視） | 確定数 | ライセンス分布 |
|---|---|---|---|---|---|---|
| | | | | | | |

層 B（Yelp キャプション由来）: 候補 ／ 確定 ／ subject_type 内訳 。ソース対照正例: 候補 ／ 確定 。

### 11.3 P0 誤受理数の見込み

| subject_type | n（work + eval） | P0 が food 確定 | 評価可否（n_before ≥ 10） |
|---|---|---|---|
| | | | |

合算 n_before（eval）＝ 。成立条件（≥ 40）の見込み: 。

### 11.4 目視の一致

パイロット 50 枚の boundary_flag 率 ／ 10% 再判定の自己一致 κ ／ guideline_version 。

## 12. 凍結（A3 で記入）

| 項目 | 値 |
|---|---|
| `configs/ood_sources.lock` の sha256 | |
| Yelp 分割（tune_fit / tune_cal / work / eval）の photo_id リスト sha256 | |
| eval_set_id | |
| guideline_version | |
| 前処理 config（長辺・JPEG 品質・EXIF 除去）の sha256 | |
| 記述分析のビン幅 | |
| git タグ | g0-freeze-v1 |

判定規則の最終数値:

| 項目 | 値 |
|---|---|
| 成立条件 合算 n_before | ≥ 40 |
| 型別 n_before | ≥ 10 |
| H0 | p ≤ 0.25 |
| 主構成 α | 0.05（E1 × kNN × P1） |
| 残り 7 構成 | Holm |
| (a) 削減率点推定 | ≥ 50% |
| (b) 確認済み損失 上限／名目 上限 | ≤ 2%／≤ 2.5% |
| (c) 型別削減率点推定 | ≥ 20% |
| (d) 他クラス FRR 上限／層 B 通過率 下限 | ≤ 3%／≥ 90% |
| (e) held-out 合算 下限 | ≥ 25% |
| (f) ソース検出フラグ | 3 倍以上かつ 5pt 以上 |
| 動作点 | tune_cal 正しく food の 1% 位置（副 0.5% / 2%） |

## 13. 結果（実験後に記入）

### 13.0 主結論

(i) ゲートが有効か（層 A／層 B、構成別）。(ii) 距離を使う追加価値があるか（MSP との対応比較、層 A／層 B）。(iii) P1 で足りるか。

### 13.1 記述分析の要約

型ごとの分離（遠く分離できる／料理側に入り込む）と 4 象限表の要点。

### 13.2 主 8 構成の主指標

| 構成 | 層 A n_before | r | 削減率 A（点／片側下限） | Δ vs MSP A（点／片側下限） | 層 B n_before_B | 削減率 B（点／片側下限） | Δ vs MSP B（点／片側下限） | 料理損失 名目（点／上限） | 料理損失 確認済み（点／上限） | 層 B P2 通過率 | 他クラス FRR | ソース対照 棄却率 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 × MSP × P1 | | | | — | | | — | | | — | — | |
| E1 × MSP × P2 | | | | — | | | — | | | | | |
| E1 × kNN × P1 | | | | | | | | | | — | — | |
| E1 × kNN × P2 | | | | | | | | | | | | |
| E1 × Mahalanobis++ × P1 | | | | | | | | | | — | — | |
| E1 × Mahalanobis++ × P2 | | | | | | | | | | | | |
| E1 × C3 × P1 | | | | | | | | | | — | — | |
| E1 × C3 × P2 | | | | | | | | | | | | |

### 13.3 帰結表

| 構成 | 層 | 主検定 (1) p（調整後） | 主検定 (2) p（調整後） | (a) | (b) | (c) | (d) | (e) | (f) | (g) | 帰結（層） | 総合 | 追加価値 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | A | | | | | | | | | | | | |
| | B | | | | | | | | | | | | |

### 13.4 P1 で足りるか

| 検出器 | 削減率 P1 | 削減率 P2 | Δ₁₂ 点推定 | Δ₁₂ 片側 95% 下限 | 許容差 −δ | P1 対象内総損失 | P2 対象内総損失 | 結論 |
|---|---|---|---|---|---|---|---|---|
| | | | | | −10 pt | | | |

### 13.5 副指標

AUROC・FPR@95、near-food 誤受理率、補助 far の food 誤受理率、Yelp vs 外部 AUROC、副次構成の記述、埋め込み時間。

## 14. 逸脱記録

凍結後に §0〜§12 または §15 を変更した場合に記入する。空のままなら逸脱なし。

| 日付 | 箇所 | 変更前 | 変更後 | 理由 |
|---|---|---|---|---|
| | | | | |

## 15. Yelp 規約と公開の扱い

- Data（photo_id・business_id・caption・ラベル行）は git に置かない。リポジトリには sha256・config・seed のみ
- reports と本書は集計値のみ。画像・ID 行・キャプション原文を載せない。README・reports に Yelp 画像を掲載しない（CLAUDE.md 絶対制約）
- 結果 PR（branch `g0-nonfood-gate`）は draft のまま置き、学術利用該当性と公開前審査の要否をユーザが規約を確認してからマージする
- 規約の版と DL 日を `reports/g0_nonfood_gate.md` のフッタに記録する（run_id / git sha / taxonomy_version / rulebook_version（未導入表記）/ eval_set_id / 概算 API コスト / W&B run URL と併記）
- 外部源はファイル単位のライセンスを `configs/ood_sources.lock` に記録し、帰属を保持する
