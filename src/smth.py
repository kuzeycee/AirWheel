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

    car_x = 0
    running = 1

    while running:
        delta = clock.tick()/1000 + 0.00001
        car_x += delta*100

        for event in pg.event.get():
            if event.type == pg.QUIT: running = 0
        screen.fill((100,150,250))

        for i in range(120):
            x = car_x + i
            scale = (120-i)/120
            road_slice = road_texture.subsurface((0,(x)%360,320,1))
            slice_width = max(int(320*scale), 1)
            scaled_slice = pg.transform.scale(road_slice, (slice_width, 1))
            color = (int(50-i/3), int(130-i), int(50+30*math.sin(x)))
            pg.draw.rect(screen, color, (0, 180-i,320,1))
            screen.blit(scaled_slice, ((320-slice_width)//2, 180-i))

        pg.display.update()
        await asyncio.sleep(0)

if __name__ == "__main__":
    pg.init()
    asyncio.run(main())
    pg.quit()