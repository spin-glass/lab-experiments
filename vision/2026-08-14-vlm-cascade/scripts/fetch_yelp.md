# Yelp Open Dataset（photos）の取得手順 — ユーザ作業

Yelp Open Dataset は規約同意が必要なため、ダウンロードは各自が公式ページから行う（design.md §2）。画像・生データはリポジトリに含めない（CLAUDE.md 絶対制約）。

## 1. 規約の確認（M0 の最初の工程）

1. 公式ページ <https://business.yelp.com/data/resources/open-dataset/> でフォームに入力し、Dataset Terms of Use に同意する
2. TAR に同梱される規約 PDF（Dataset_User_Agreement.pdf）を読み、少なくとも次を確認して `reports/m0_setup.md` に記録する
   - 利用目的の制限（学術利用）と、その該当性の判断
   - 公開前の審査・提出に関する条項の有無と、本リポジトリの結果公開（reports）への影響
   - Data の表示・配布の禁止範囲（photo_id 等の一覧を git に置かない根拠）
   - 有効期間（ダウンロード日を記録する）
3. 規約の版（PDF 内の "Last updated" 等）とダウンロード日を控える。両方を reports のフッタに刻印する

## 2. 配置

```
data/yelp/
  photos/            # 約 20 万枚の JPEG（TAR を展開したもの）
  photos.json        # 1 行 1 写真: photo_id, business_id, caption, label
  Dataset_User_Agreement.pdf
  DOWNLOAD.txt       # 自分で作る: ダウンロード日 (YYYY-MM-DD) と TAR の sha256
```

`data/` は `.gitignore` 済み。別の場所に置く場合は環境変数 `CASCADE_DATA_DIR` で `data/` の位置を変える。

TAR の sha256 は取得直後に控える:

```bash
shasum -a 256 <取得した photos の tar> | tee -a data/yelp/DOWNLOAD.txt
```

## 3. 検証

配置後に次を実行する（B-b で実装。photos.json のスキーマ、画像枚数、クラス別枚数、ラベル欠損率、画像サイズ分布、TAR の sha256 の記録を行う）:

```bash
uv run python -m cascade.m0_setup --config configs/m0.yaml --verify-yelp
```

検証に通ったら、tune / work / eval の分割と eval の凍結に進む（design.md §2、§6 M0）。
