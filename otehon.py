import cv2
import numpy as np
import pygame
import pymunk
from pymunk import Vec2d
import threading
import time
from ultralytics import YOLO

# モデルロード
model = YOLO('runs/segment/train14/weights/best.pt')

# カメラ起動
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

latest_frame = None
latest_seg = None
latest_contour = None
contour_flag = False
message = ""
last_seg_time = time.time()

# セグメンテーションスレッド
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
            masks = result.masks.data.cpu().numpy()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = img_rgb.shape

            mask = masks[0]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(mask_bin)
            max_area = 0
            max_label = 0
            for label in range(1, num_labels):
                area = np.sum(labels == label)
                if area > max_area:
                    max_area = area
                    max_label = label
            largest_mask = (labels == max_label).astype(np.uint8)

            mask_3ch = np.stack([largest_mask]*3, axis=-1)
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
                contour = max(contours, key=cv2.contourArea).squeeze()
                if contour.ndim == 2 and contour.shape[0] >= 3:
                    latest_contour = contour
                    contour_flag = True
        else:
            message = "Cannot Find"
            latest_seg = frame

threading.Thread(target=segmentation_loop, daemon=True).start()

# PygameとPymunk初期化
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Hand Gesture Tower Battle")
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = 0, -900

ground = pymunk.Segment(space.static_body, (40, 10), (600, 10), 5)
ground.friction = 10
space.add(ground)

scroll_offset_y = 0
gameover = False
bodies = []

# 輪郭から物体生成
def drop_contour(contour):
    global scroll_offset_y
    h, w, _ = latest_frame.shape
    scale = 640 / w

    points = [Vec2d(pt[0], h - pt[1]) * scale for pt in contour]
    center = sum(points, Vec2d(0, 0)) / len(points)
    points = [p - center for p in points]

    contour_x = [pt[0] for pt in contour]
    contour_cx = np.mean(contour_x)
    body_x = contour_cx * scale

    mass = 5.0
    moment = pymunk.moment_for_poly(mass, points)
    body = pymunk.Body(mass, moment)
    body.position = Vec2d(body_x, 450 + scroll_offset_y)
    body.creation_time = time.time()
    shape = pymunk.Poly(body, points)
    shape.friction = 10
    space.add(body, shape)
    bodies.append(body)

# メインループ
while True:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                space.remove(*space.bodies, *space.shapes)
                space.add(ground)
                gameover = False
                scroll_offset_y = 0
                bodies.clear()

    if gameover:
        continue

    for _ in range(5):
        space.step(1 / 250.0)

    if any(body.position.y < 0 for body in bodies):
        gameover = True

    if contour_flag:
        contour_flag = False
        if latest_contour is not None:
            drop_contour(latest_contour)

    # 描画
    screen.fill((230, 240, 255))
    for shape in space.shapes:
        if isinstance(shape, pymunk.Poly):
            points = [(p.x, 480 - (p.y - scroll_offset_y)) for p in shape.get_vertices()]
            pygame.draw.polygon(screen, (0, 0, 100), points)

    if not gameover:
        current_time = time.time()
        valid_bodies = [b for b in bodies if current_time - getattr(b, "creation_time", 0) > 1.0]
        if valid_bodies:
            highest_y = max(body.position.y for body in valid_bodies)
            scroll_offset_y = max(0, highest_y - 200)
            font = pygame.font.SysFont("Arial", 24)
            label = font.render(f"Height: {int(highest_y)//10} m", True, (0, 0, 0))
            screen.blit(label, (10, 10))
    else:
        font = pygame.font.SysFont("Arial", 32)
        label = font.render("GAME OVER", True, (255, 0, 0))
        screen.blit(label, (220, 220))

    pygame.display.flip()

    # カメラ処理
    ret, frame = cap.read()
    if not ret:
        continue

    latest_frame = frame.copy()
    live = cv2.resize(cv2.flip(frame, 1), (640, 240))

    countdown = max(0, 5 - int(time.time() - last_seg_time))
    if countdown > 0:
        cv2.putText(live, f'{countdown}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        cv2.putText(live, "taking a picture...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    seg = np.zeros((240, 640, 3), dtype=np.uint8)
    if latest_seg is not None:
        seg = cv2.resize(latest_seg, (640, 240))

    if message:
        cv2.putText(seg, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    game_surf = pygame.surfarray.array3d(screen)
    game_img = np.rot90(np.fliplr(game_surf)).copy()

    combined = np.vstack([live, seg, game_img])
    cv2.imshow("Hand Gesture Tower Battle", combined)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
