import arcade
import pymunk
import random
import math
from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('runs/segment/train14/weights/best.pt')

results = model('IMG_2040.jpeg')
result = results[0]
if result.masks is None or result.masks.data.shape[0] == 0:
    print("No mask found.")
    
masks = result.masks.data.cpu().numpy()

        # 最初のマスクを使用
mask = (masks[0] * 255).astype(np.uint8)

        # 輪郭抽出
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No contour found.")
    

contour = max(contours, key=cv2.contourArea)
contour = contour.squeeze()

import matplotlib.pyplot as plt

# 空の背景（白）を作成
contour_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
contour_img[:] = (255, 255, 255)  # 背景を白にする

# 輪郭を赤で描画
cv2.drawContours(contour_img, [contour.reshape(-1, 1, 2)], -1, (255, 0, 0), thickness=2)

# matplotlibで表示
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title("Contour Image")
plt.axis("off")
plt.show()
