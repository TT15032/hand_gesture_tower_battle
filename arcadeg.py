import arcade
import pymunk
import random
import math
from ultralytics import YOLO
import numpy as np

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN_TITLE = "Hand Geesture Tower Battle"
model = YOLO('runs/segment/train14/weights/best.pt')

class MyGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        arcade.set_background_color(arcade.color.SKY_BLUE)

        self.block_list = arcade.SpriteList()
        self.terrain_list = arcade.SpriteList()

        self.space = pymunk.Space()
        self.space.gravity = (0, -900)

        self.player1_turn = True
        self.gameover = False

        self.left_pressed = False
        self.right_pressed = False

        self.setup()

    def setup(self):
        self.block_list = arcade.SpriteList()
        self.terrain_list = arcade.SpriteList()
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)
        self.gameover = False
        self.player1_turn = True

        # 地面
        ground = arcade.SpriteSolidColor(SCREEN_WIDTH/2, 50, arcade.color.DARK_BROWN)
        ground.center_x = SCREEN_WIDTH // 2
        ground.center_y = 25
        self.terrain_list.append(ground)

        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = ground.center_x, ground.center_y
        shape = pymunk.Poly.create_box(body, (SCREEN_WIDTH/2, 50))
        shape.friction = 100
        self.space.add(body, shape)
        ground.pymunk_shape = shape

    def on_draw(self):
        self.clear()  # これが描画前に必要
        self.block_list.draw()
        self.terrain_list.draw()

        if self.gameover:
            arcade.draw_text("Game Over", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2,
                             arcade.color.RED, 36)

    def on_update(self, delta_time: float):
        if self.gameover:
            return

        self.space.step(delta_time)

        # ブロックの物理更新
        for block in self.block_list:
            block.center_x = block.pymunk_shape.body.position.x
            block.center_y = block.pymunk_shape.body.position.y
            block.angle = math.degrees(block.pymunk_shape.body.angle)

        # ブロックが落ちたら削除してゲームオーバー
        for block in self.block_list:
            if block.center_y < 0:
                self.gameover = True
                break

    def on_key_press(self, key, modifiers):
        if key == arcade.key.SPACE and not self.gameover:
            self.drop_block()
        elif key == arcade.key.R:
            self.setup()
        elif key == arcade.key.LEFT:
            self.left_pressed = True
        elif key == arcade.key.RIGHT:
            self.right_pressed = True

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT:
            self.left_pressed = False
        elif key == arcade.key.RIGHT:
            self.right_pressed = False

    def drop_block(self):
        # ランダムな色のブロックを生成
        color = random.choice([
            arcade.color.RED, arcade.color.BLUE, arcade.color.GREEN, arcade.color.YELLOW
        ])
        block = arcade.SpriteSolidColor(60, 60, color)
        x = SCREEN_WIDTH // 2
        y = SCREEN_HEIGHT - 50
        block.center_x = x
        block.center_y = y
        self.block_list.append(block)

        mass = 1
        moment = pymunk.moment_for_box(mass, (60, 60))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        shape = pymunk.Poly.create_box(body, (60, 60))
        shape.friction = 100
        self.space.add(body, shape)
        block.pymunk_shape = shape


def main():
    game = MyGame()
    arcade.run()


if __name__ == "__main__":
    main()

