#実行方法： python3 game.py
import cv2
import time
import numpy as np
import threading
from ultralytics import YOLO
import pymunk
from pymunk import Vec2d

model = YOLO('model.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラが見つかりません (index 0)")
    exit()

latest_frame = None
latest_seg = None
latest_contour = None
contour_flag = False
message = ""
last_seg_time = time.time()


def segmentation_loop():
    global latest_seg, latest_contour, latest_frame
    global contour_flag, message, last_seg_time

    while True:
        time.sleep(5)
        last_seg_time = time.time()

        if latest_frame is None:
            continue

        frame = cv2.flip(latest_frame.copy(), 1)
        results = model(frame)
        result = results[0]

        if result.masks is not None:
            masks = result.masks.data.numpy()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            mask = masks[0]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(mask_bin)
            max_area = 0
            max_label = 0
            for lbl in range(1, num_labels):
                area = int(np.sum(labels == lbl))
                if area > max_area:
                    max_area = area
                    max_label = lbl
            largest_mask = (labels == max_label).astype(np.uint8)

            mask_3ch = np.stack([largest_mask] * 3, axis=-1)
            color = np.array([0, 255, 0], dtype=np.uint8)
            overlay = np.where(mask_3ch, color, img_rgb)
            blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
            latest_seg = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            if max_area < 4000:
                message = "Too Small"
                continue
            elif max_area > 200000:
                message = "Too Large"
                continue
            else:
                message = ""

            contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                raw = max(contours, key=cv2.contourArea)
                # ポリゴンを簡略化して点数を減らす
                epsilon = 0.01 * cv2.arcLength(raw, True)
                approx = cv2.approxPolyDP(raw, epsilon, True).squeeze()
                if approx.ndim == 2 and approx.shape[0] >= 3:
                    latest_contour = approx
                    contour_flag = True
        else:
            message = "Cannot Find"
            latest_seg = frame


threading.Thread(target=segmentation_loop, daemon=True).start()


class Game:
    W = 640
    H = 640

    def __init__(self):
        self.best_score = 0.0
        self.reset_game()

    def reset_game(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -900)
        self.gameover = False
        self.score_updated = False
        self.scroll_offset_y = 0.0
        self.highest_y = 0.0

        ground = pymunk.Segment(self.space.static_body, Vec2d(40, 10), Vec2d(600, 10), 5)
        ground.friction = 10
        self.space.add(ground)

    def update(self, dt):
        global contour_flag

        step_dt = 1 / 250.0
        steps = max(1, int(dt / step_dt))
        for _ in range(steps):
            self.space.step(step_dt)

        for shape in self.space.shapes:
            if (shape.body.body_type == pymunk.Body.DYNAMIC
                    and shape.body.position.y < 0):
                self.gameover = True
                if self.highest_y > self.best_score:
                    self.best_score = self.highest_y
                    self.score_updated = True
                break

        if contour_flag:
            contour_flag = False
            if latest_contour is not None:
                if self.gameover:
                    self.reset_game()
                self.drop_contour(latest_contour)

    def drop_contour(self, contour):
        h, w, _ = latest_frame.shape
        scale = self.W / w

        # テクスチャをカメラ画像から切り抜いて保存
        frame_flipped = cv2.flip(latest_frame.copy(), 1)
        contour_int = contour.astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(contour_int)
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        tex_crop = frame_flipped[y1:y2, x1:x2].copy()

        # 画像座標 → 物理座標 (y軸反転)
        points = [Vec2d(float(pt[0]), float(h - pt[1])) * scale for pt in contour]
        center = sum(points, Vec2d(0, 0)) / len(points)
        local_pts = [p - center for p in points]

        body_x = float(np.mean([pt[0] for pt in contour])) * scale

        mass = 5.0
        moment = pymunk.moment_for_poly(mass, local_pts)
        body = pymunk.Body(mass, moment)
        body.position = Vec2d(body_x, 450 + self.scroll_offset_y)
        body.creation_time = time.time()

        # テクスチャ情報をbodyに付与
        body.texture = tex_crop
        body.tex_x1 = float(x1)
        body.tex_y1 = float(y1)
        body.img_h = float(h)
        body.img_scale = scale
        body.center_phys = center

        shape = pymunk.Poly(body, local_pts)
        shape.friction = 10
        self.space.add(body, shape)

    def draw(self):
        W, H = self.W, self.H
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (255, 242, 230)  # 薄い水色 (BGR)

        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Segment):
                a = shape.body.local_to_world(shape.a)
                b = shape.body.local_to_world(shape.b)
                pt1 = (int(a.x), int(H - (a.y - self.scroll_offset_y)))
                pt2 = (int(b.x), int(H - (b.y - self.scroll_offset_y)))
                cv2.line(img, pt1, pt2, (60, 60, 60), 5)
            elif isinstance(shape, pymunk.Poly):
                body = shape.body
                verts = [body.local_to_world(v) for v in shape.get_vertices()]
                pts = np.array([
                    [int(v.x), int(H - (v.y - self.scroll_offset_y))]
                    for v in verts
                ], dtype=np.int32)

                drawn = False
                if hasattr(body, 'texture') and body.texture is not None:
                    th, tw = body.texture.shape[:2]
                    if th > 0 and tw > 0:
                        sc = body.img_scale
                        cx = body.center_phys.x
                        cy = body.center_phys.y
                        # crop座標 → フル画像座標
                        M0 = np.array([[1, 0, body.tex_x1],
                                       [0, 1, body.tex_y1],
                                       [0, 0, 1]], dtype=np.float64)
                        # フル画像座標 → ボディローカル物理座標
                        M1 = np.array([[sc,   0,  -cx],
                                       [0,  -sc,  body.img_h * sc - cy],
                                       [0,   0,   1]], dtype=np.float64)
                        # ローカル → ワールド (回転+平行移動)
                        cos_t = np.cos(body.angle)
                        sin_t = np.sin(body.angle)
                        bx, by = body.position.x, body.position.y
                        M2 = np.array([[cos_t, -sin_t, bx],
                                       [sin_t,  cos_t, by],
                                       [0,      0,     1]], dtype=np.float64)
                        # ワールド → スクリーン
                        M3 = np.array([[1,  0,  0],
                                       [0, -1,  H + self.scroll_offset_y],
                                       [0,  0,  1]], dtype=np.float64)
                        M_total = M3 @ M2 @ M1 @ M0
                        warped = cv2.warpPerspective(body.texture, M_total, (W, H))
                        poly_mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.fillPoly(poly_mask, [pts], 255)
                        img = np.where(poly_mask[:, :, np.newaxis], warped, img)
                        drawn = True

                if not drawn:
                    cv2.fillPoly(img, [pts], (255, 180, 120))
                cv2.polylines(img, [pts], True, (180, 100, 50), 2)

        dynamic_bodies = [
            s.body for s in self.space.shapes
            if s.body.body_type == pymunk.Body.DYNAMIC
        ]
        now = time.time()
        valid_bodies = [
            b for b in dynamic_bodies
            if hasattr(b, "creation_time") and (now - b.creation_time) > 1.0
        ]

        if not self.gameover:
            if valid_bodies:
                self.highest_y = max(b.position.y for b in valid_bodies)
                self.scroll_offset_y = max(0.0, self.highest_y - 200)
            score_text = f'{self.highest_y / 10:.1f} m   (best: {self.best_score / 10:.1f} m)'
            cv2.putText(img, score_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
        else:
            cv2.putText(img, "GAME OVER",
                        (W // 2 - 130, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 200), 3)
            msg2 = (f'New best!!  {self.best_score / 10:.1f} m'
                    if self.score_updated
                    else f'Score: {self.highest_y / 10:.1f} m')
            cv2.putText(img, msg2,
                        (W // 2 - 140, H // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
            cv2.putText(img, "Show hand to restart",
                        (W // 2 - 150, H // 2 + 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

        return img


if __name__ == "__main__":
    game = Game()
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        latest_frame = frame.copy()

        now = time.time()
        dt = now - last_time
        last_time = now
        game.update(dt)

        # --- 上段: ライブカメラ ---
        live = cv2.resize(cv2.flip(frame, 1), (640, 240))
        countdown = max(0, 5 - int(now - last_seg_time))
        if countdown > 0:
            cv2.putText(live, str(countdown), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        else:
            cv2.putText(live, "Scanning...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # --- 中段: セグメンテーション結果 ---
        if latest_seg is not None:
            seg = cv2.resize(latest_seg, (640, 240))
        else:
            seg = np.zeros((240, 640, 3), dtype=np.uint8)
        if message:
            cv2.putText(seg, message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- 下段: ゲーム ---
        game_img = game.draw()

        combined = np.vstack([live, seg, game_img])
        cv2.imshow("Hand Gesture Tower Battle", combined)

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == 27:           # ESC: 終了
            break
        elif key_pressed in (ord('r'), ord('R')):   # R: リスタート
            game.reset_game()

    cap.release()
    cv2.destroyAllWindows()
