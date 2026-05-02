import cv2
import time
import numpy as np
import threading
from ultralytics import YOLO

import pyglet
from pyglet.window import key

import pymunk
import pymunk.pyglet_util
from pymunk import Vec2d

model = YOLO('runs/segment/train14/weights/best.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
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
        #seg
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
            continue

threading.Thread(target=segmentation_loop, daemon=True).start()

class Main(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=640, height=640, visible=False, vsync=False)
        self.set_caption("Hand Gesture Tower Battle")
        pyglet.clock.schedule_interval(self.update, 1 / 20)
        self.reset_game()

    def reset_game(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -900)
        self.draw_options = pymunk.pyglet_util.DrawOptions()
        self.gameover = False
        self.scroll_offset_y = 0

        ground = [
            pymunk.Segment(self.space.static_body, Vec2d(40, 10), Vec2d(600, 10), 5)
        ]
        for g in ground:
            g.friction = 10
        self.space.add(*ground)

    def update(self, dt):
        global contour_flag

        if self.gameover:
            return

        step_dt = 1 / 250.0
        for _ in range(int(dt / step_dt)):
            self.space.step(step_dt)

        for shape in self.space.shapes:
            if shape.body.position.y < 0:
                self.gameover = True
                break

        if contour_flag:
            contour_flag = False
            if latest_contour is not None:
                self.drop_contour(latest_contour)

    def drop_contour(self, contour):
        h, w, _ = latest_frame.shape
        scale = self.width / w

        points = [Vec2d(pt[0], h - pt[1]) * scale for pt in contour]
        center = sum(points, Vec2d(0, 0)) / len(points)
        points = [p - center for p in points]

        contour_x = [pt[0] for pt in contour]
        contour_cx = np.mean(contour_x)
        body_x = contour_cx * scale

        mass = 5.0
        moment = pymunk.moment_for_poly(mass, points)
        body = pymunk.Body(mass, moment)
        body.position = Vec2d(body_x, 450 + self.scroll_offset_y)
        body.creation_time = time.time()
        shape = pymunk.Poly(body, points)
        shape.friction = 10
        self.space.add(body, shape)

    def on_draw(self):
        pyglet.gl.glClearColor(0.9, 0.95, 1.0, 1.0)
        self.clear()
        transform = pymunk.Transform(tx=0, ty=-self.scroll_offset_y)
        self.draw_options.transform = transform
        self.space.debug_draw(self.draw_options)

        dynamic_bodies = [s.body for s in self.space.shapes if not isinstance(s, pymunk.Segment)]
        current_time = time.time()
        valid_bodies = [
            body for body in dynamic_bodies
            if hasattr(body, "creation_time") and (current_time - body.creation_time) > 1.0
        ]

        if valid_bodies:
            highest_y = max(body.position.y for body in valid_bodies)
            self.scroll_offset_y = max(0, highest_y - 200)
            label = pyglet.text.Label(
                f'Height: {int(highest_y)/10} m',
                font_name='Arial',
                font_size=25,
                x=10, y=self.height - 25,
                color=(0, 0, 0, 255)
            )
            label.draw()

        if self.gameover:
            game_over_label = pyglet.text.Label(
                "GAME OVER",
                font_name='Arial',
                font_size=25,
                x=self.width // 2,
                y=self.height // 2,
                anchor_x='center', anchor_y='center',
                color=(255, 0, 0, 255)
            )
            game_over_label.draw()

        buffer = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        game_bytes = buffer.get_data('RGB', buffer.width * 3)
        game_img = np.frombuffer(game_bytes, dtype=np.uint8).reshape(buffer.height, buffer.width, 3)
        game_img = cv2.cvtColor(game_img, cv2.COLOR_RGB2BGR)
        game_img = np.flipud(game_img)

        global latest_frame, latest_seg, message, last_seg_time
        ret, frame = cap.read()
        if not ret:
            return

        latest_frame = frame.copy()

        live = cv2.resize(cv2.flip(frame, 1), (640, 240))

        countdown = max(0, 5 - int(time.time() - last_seg_time))
        if countdown > 0:
            cv2.putText(live, f'{countdown}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(live, "taking a picture...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if latest_seg is not None:
            seg = cv2.resize(latest_seg, (640, 240))
        else:
            seg = np.zeros((240, 640, 3), dtype=np.uint8)

        if message:
            cv2.putText(seg, message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        combined = np.vstack([live, seg, game_img])
        cv2.imshow("Hand Gesture Tower Battle", combined)

        if cv2.waitKey(1) == 27:
            pyglet.app.exit()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.R:
            self.reset_game()

    def on_close(self):
        cap.release()
        cv2.destroyAllWindows()
        super().on_close()

if __name__ == "__main__":
    Main()
    pyglet.app.run()
