import asyncio
import pygame as pg
import sys, math, random

async def main():
    screen_size = [320,180]

    if sys.platform == "emscripten":
        platform.window.canvas.style.imageRendering = "pixelated"
        screen = pg.display.set_mode(screen_size)
    else:
        screen = pg.display.set_mode(screen_size, pg.SCALED)

    clock = pg.time.Clock()
    clock.tick(); pg.time.wait(16)
    road_texture = pg.image.load("assets/road.png").convert()
    mountains_texture = pg.image.load("assets/mountains.png").convert()
    car_sprite = pg.image.load("assets/car.png").convert()
    car_sprite.set_colorkey((255,0,255))

    car = Player()
    running = 1

    while running:

        delta = clock.tick()/1000 + 0.00001
        car.controls(delta)

        for event in pg.event.get():
            if event.type == pg.QUIT: running = 0

        screen.fill((100,150,250))
        screen.blit(mountains_texture, (-65 - car.angle*82, 0))
        pg.draw.rect(screen, (55,125,55), (0, 60, 320, 120))
        vertical, draw_distance= 180, 1
        car.z = 100+40*math.sin(car.x/13)-60*math.sin(car.x/7)

        while draw_distance < 120:
            last_vertical = vertical
            while vertical >= last_vertical and draw_distance < 120:
                draw_distance += draw_distance / 150
                x = car.x + draw_distance
                scale = 1 /draw_distance
                z = 100 + 40 * math.sin(x / 13) - 60 * math.sin(x / 7) - car.z
                vertical = int(60+120*scale + z*scale)
                if draw_distance < 120:
                    y = 200 * math.sin(x / 1170) + 170 * math.sin(x / 8) - car.y
                    horizontal = 160 - (160 - y) * scale + car.angle * (vertical-150)
                    road_slice = road_texture.subsurface((0, (x) % 360, 320, 1))
                    slice_width = max(int(320 * scale), 1)
                    scaled_slice = pg.transform.scale(road_slice, (500*scale, 1))
                    color = (int(50 - draw_distance / 3), int(130 - draw_distance), int(50 + 30 * math.sin(x)))
                    pg.draw.rect(screen, color, (0, vertical, 320, 1))
                    screen.blit(scaled_slice, (int(horizontal - slice_width / 2), vertical))

        screen.blit(car_sprite, (100, 120))
        pg.display.update()
        await asyncio.sleep(0)

class Player():
    def __init__ (self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.angle = 0
        self.velocity = 0
        self.acceleration = 0

    def controls(self, delta):
        pressed_keys = pg.key.get_pressed()
        self.acceleration += -0.5*self.acceleration*delta
        self.velocity += 0.5*self.velocity*delta

        if pressed_keys[pg.K_w] or pressed_keys[pg.K_UP]:
            if self.velocity > -1:
                self.acceleration += 4*delta
            else:
                self.acceleration = 0
                self.velocity += -self.acceleration*delta
        elif pressed_keys[pg.K_s] or pressed_keys[pg.K_DOWN]:
            if self.velocity > -1:
                self.acceleration -= delta
            else:
                self.acceleration = 0
                self.velocity += self.velocity*delta
        if pressed_keys[pg.K_a] or pressed_keys[pg.K_LEFT]:
            self.angle -= delta*self.velocity/10
        elif pressed_keys[pg.K_d] or pressed_keys[pg.K_RIGHT]:
            self.angle += delta*self.velocity/10
        self.velocity = max(-10,min(10,self.velocity))
        self.velocity += self.acceleration*delta
        self.x += self.velocity*delta*math.cos(self.angle)
        self.y += self.velocity*math.sin(self.angle)*delta*100

if __name__ == "__main__":
    pg.init()
    asyncio.run(main())
    pg.quit()