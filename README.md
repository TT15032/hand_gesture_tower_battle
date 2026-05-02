

https://github.com/user-attachments/assets/3ae3f609-a247-4045-bf11-165d7ceaf40a





# Hand Gesture Tower Battle

カメラに映した手の輪っかの形をリアルタイムにセグメンテーションし、その輪郭をそのまま物理オブジェクトとしてゲーム内に落とす積み上げゲームです。

## 概要

1. カメラに向けて手で輪っかを作る
2. YOLOv8 がリアルタイムで輪っかの形状をセグメンテーション
3. 検出した輪郭が物理エンジン（pymunk）のブロックとしてフィールドに降ってくる
4. 高く積み上げるほどスコアが上がる

ウィンドウは3段構成です：

| 段 | 内容 |
|----|------|
| 上 | ライブカメラ映像 |
| 中 | セグメンテーション結果 |
| 下 | ゲーム画面 |

## 必要環境

- Python 3.10 以上
- Webカメラ

## セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/<your-username>/opencv.git
cd opencv

# 依存パッケージをインストール（uv 推奨）
uv sync

# または pip
pip install opencv-python ultralytics pymunk
```

## 実行

```bash
python3 game.py
```

| キー | 操作 |
|------|------|
| `R`  | リスタート |
| `ESC` | 終了 |

## モデルについて

`model.pt` は YOLOv8n-seg をベースにカスタムデータで学習したセグメンテーションモデルです。  
自分で学習し直す場合は `train_seg.ipynb` を参照してください。

## 依存ライブラリ

- [OpenCV](https://opencv.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [pymunk](http://www.pymunk.org/)
