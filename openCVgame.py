import cv2
import time
from ultralytics import YOLO
import numpy as np

# モデル読み込み
model = YOLO('runs/segment/train14/weights/best.pt')

# カメラの設定（デバイスID = 0）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラが開けませんでした。")
    exit()

last_time = time.time()
segmentation_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました。")
        break

    frame = cv2.flip(frame, 1)  # 左右反転

    # 2秒おきにセグメンテーション実行
    if time.time() - last_time >= 5:
        last_time = time.time()

        results = model(frame)
        result = results[0]

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            # 最初のマスクだけ処理
            mask = masks[0]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            # ラベリング（領域分離）
            num_labels, labels = cv2.connectedComponents(mask_bin)

            # 最大領域を探す（背景はラベル0）
            max_area = 0
            max_label = 0
            for label in range(1, num_labels):
                area = np.sum(labels == label)
                if area > max_area:
                    max_area = area
                    max_label = label

            # 最大ラベルだけを抽出
            largest_mask = (labels == max_label).astype(np.uint8)

            # 3チャンネルに拡張して色付け
            mask_3ch = np.stack([largest_mask]*3, axis=-1)
            color = np.array([0, 255, 0], dtype=np.uint8)
            overlay = np.where(mask_3ch, color, img_rgb)
            blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
            segmentation_image = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        else:
            print("マスクが見つかりませんでした。")
            segmentation_image = None

    # 結果の表示（上下に連結）
    if segmentation_image is not None:
        live_and_seg = np.vstack([frame, segmentation_image])
    else:
        live_and_seg = frame

    cv2.imshow("Live + Segmentation (Flipped)", live_and_seg)

    # ESCキーで終了
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()