import pyglet
import pymunk
from pymunk.pyglet_util import DrawOptions

window = pyglet.window.Window()
space = pymunk.Space()
space.gravity = (0, -900)
options = DrawOptions()

drawn_points = []
lines = []

def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    drawn_points.append((x, y))

def on_mouse_release(x, y, button, modifiers):
    for i in range(len(drawn_points) - 1):
        p1 = drawn_points[i]
        p2 = drawn_points[i + 1]
        segment = pymunk.Segment(space.static_body, p1, p2, 5)
        segment.friction = 0.9
        space.add(segment)
        lines.append(segment)
    drawn_points.clear()
    spawn_ball()

def spawn_ball():
    body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 10))
    body.position = (100, 400)
    shape = pymunk.Circle(body, 10)
    shape.friction = 0.9
    space.add(body, shape)

def update(dt):
    space.step(dt)

def on_draw():
    window.clear()
    space.debug_draw(options)

pyglet.clock.schedule_interval(update, 1/60)
pyglet.app.run()
