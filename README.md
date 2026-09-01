# EEG・EMG・モーションを用いた運動意図推定

脳波（EEG）・筋電（EMG）・モーションキャプチャの複合信号から、動作の遷移前後で
**どの運動を行おうとしているか**を推定するパイプラインの実装です。
九州大学での卒業研究として開発しました。

---

## 概要

Brain–Computer Interface（BCI）や歩行支援・神経リハビリテーションへの応用を念頭に、
EEG・EMG・モーションの特徴量を組み合わせた**7 クラス分類**で運動意図を推定します。

| 項目 | 内容 |
|------|------|
| 入力信号 | EEG（32ch）・表面 EMG（8ch, 下肢筋）・モーションキャプチャ（下肢マーカー） |
| タスク | 運動意図の 7 クラス分類：立位 / 座位 / 昇段 / 降段 / 歩行 / 立ち上がり / 座り |
| 手法 | 帯域分割 → スライディング窓 → 特徴量抽出（Hjorth / RMS / 運動学） → 全結合 NN |
| 主眼 | 動作遷移点の前後 ±400 ms で意図をどれだけ早く読み出せるか |
| ベースライン | チャンスレベル 1/7 ≈ 14.3% |

分類精度などの定量結果は卒業論文を参照してください（本リポジトリでは数値は未記載）。

---

## リポジトリ構成

```
eeg-emg-motion-intention/
├── src/motion_intent/          # パイプライン共通ロジック（pip install -e . でインポート可能）
│   ├── config.py               #   パス・チャンネル群・周波数帯・窓長の一元管理
│   ├── io_xdf.py               #   XDF の LSL ストリームを 1 つの時間軸に統合
│   ├── labeling.py             #   EMG 包絡・マーカー高さからの動作ラベル付けと平滑化
│   ├── preprocessing.py        #   帯域通過・リサンプル・EEG アーティファクト除去・軸統合
│   ├── windowing.py            #   時系列ブロック分割とスライディング窓
│   ├── features.py             #   Hjorth / RMS / 平均位置 / 環境 特徴量
│   ├── datasets.py             #   特徴量行列の torch Dataset ラッパ
│   ├── models.py               #   全結合意図分類器（EEG単独 / 非EEG / 融合）
│   └── training.py             #   学習・評価ループ（Early Stopping）
├── notebooks/                  # ステージごとに分割し、出力が次段の入力になる構成
│   ├── 01_xdf_to_csv.ipynb
│   ├── 02_labeling.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_extraction.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_evaluation.ipynb
├── docs/pipeline.md            # パイプライン図（Mermaid）と各ステージの入出力仕様
├── tests/test_features.py      # 合成配列による特徴量抽出のスモークテスト
└── data/                       # 各ステージの入出力（.gitignore 対象）
```

パイプラインの全体像は [`docs/pipeline.md`](docs/pipeline.md) を参照してください。

### パイプライン

| # | ノートブック | 入力 | 出力 |
|---|---|---|---|
| 01 | `01_xdf_to_csv`      | `data/raw/<subject>/*.xdf` | `data/csv/<subject>/<session>.csv` |
| 02 | `02_labeling`        | `data/csv/…` | `data/formatted/<subject>/*.csv`（`label` 列付与） |
| 03 | `03_preprocessing`   | `data/formatted/…` | `data/clean/<subject>.csv` |
| 04 | `04_feature_extraction` | `data/clean/…` | `data/features/<subject>_{train,val,test}.npz` |
| 05 | `05_model_training`  | 特徴量行列 | `data/checkpoints/<subject>/*.pth` |
| 06 | `06_evaluation`      | 学習済みモデル | 精度・混同行列・遷移オフセット別精度の図表 |

---

## 手法の要点

- **マルチモーダル融合** — EEG・EMG・モーションの特徴量を結合し、単一モダリティより
  早く・安定して意図を推定する。
- **Hjorth パラメータ** — 分散ベースの軽量な EEG 特徴量（activity / mobility）を
  α・β 帯域それぞれに適用し、リアルタイム推論に必要な計算コストを抑える。
- **セッション単位の処理** — 帯域通過フィルタと時系列 train/val/test 分割は
  録画境界をまたがずセッションごとに適用し、窓のリークを防ぐ。
- **遷移オフセット解析** — テスト窓を遷移点に対して −400〜+400 ms ずらして評価し、
  「意図がどれだけ先読みできるか」を特徴量セット間で比較する。

---

## セットアップ

```bash
git clone https://github.com/TODO_USER/eeg-emg-motion-intention.git
cd eeg-emg-motion-intention
pip install -e .          # src/motion_intent をインストール
# もしくは: pip install -r requirements.txt
```

動作確認環境: Python 3.9+ / PyTorch（CPU wheel で可、CUDA は PyTorch 公式手順参照）

```bash
pytest        # 合成配列で特徴量抽出の形状を検証
```

---

## データについて

> **実験データは参加者のプライバシー保護のため公開していません。**
> ノートブックは各ステージの出力が次段の入力になる構造を示すもので、
> 実行には `data/` 以下に自前の記録データを配置する必要があります。
> データフォーマットは各ノートブック冒頭と `docs/pipeline.md` に記載しています。

---

## 参考文献

- （使用した論文・手法の参考文献をここに追記）

---

## Author

**TODO: Your Name**
Kyushu University
[GitHub](https://github.com/TODO_USER)

## License

MIT License — see [`LICENSE`](LICENSE).
