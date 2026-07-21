import asyncio
import pygame as pg
import sys, math, random, time
import cv2
import mediapipe as mp
import Main as airwheel

HAND_ANGLE_SMOOTHING = 0.3
HAND_CALIBRATION_SECONDS = 1.5
HAND_ANGLE_TO_RADIANS = 60

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
    tree_texture = pg.image.load("assets/tree.png").convert_alpha()
    grass_texture = pg.image.load("assets/grass.png").convert_alpha()
    TREE_SPACING = 18
    GRASS_SPACING = 6
    # the road texture is blitted 500*scale wide but centered using 320*scale,
    # so it actually reaches 160*scale left of `horizontal` and 340*scale right of it
    LEFT_ROAD_EDGE = 160
    RIGHT_ROAD_EDGE = 340

    def roadside_sprite_at(x):
        xi = int(x)
        if xi % TREE_SPACING == 0:
            side = 1 if (xi // TREE_SPACING) % 2 == 0 else -1
            road_edge = RIGHT_ROAD_EDGE if side == 1 else LEFT_ROAD_EDGE
            return tree_texture, side, road_edge + 40, 14
        if xi % GRASS_SPACING == 0:
            side = 1 if (xi // GRASS_SPACING) % 2 == 0 else -1
            road_edge = RIGHT_ROAD_EDGE if side == 1 else LEFT_ROAD_EDGE
            return grass_texture, side, road_edge + 15, 10
        return None

    pg.font.init()
    hud_font = pg.font.SysFont(None, 16)

    use_hand_control = None
    while use_hand_control is None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_1:
                    use_hand_control = False
                elif event.key == pg.K_2:
                    use_hand_control = True
        screen.fill((20,20,30))
        screen.blit(hud_font.render("How do you want to drive?", True, (255,255,255)), (60,60))
        screen.blit(hud_font.render("Press 1: Keyboard", True, (255,255,0)), (70,85))
        screen.blit(hud_font.render("Press 2: Hand steering", True, (255,255,0)), (70,100))
        pg.display.update()
        await asyncio.sleep(0)

    hand_cap = None
    hand_landmarker = None
    hand_start_time = time.time()
    hand_smoothed_angle = 0.0
    hand_baseline_angle = 0.0
    hand_calibrated = False
    hand_calibration_start = None
    hand_calibration_samples = []

    if use_hand_control:
        airwheel.ensure_model()
        hand_cap = cv2.VideoCapture(airwheel.CAMERA_INDEX)
        hand_landmarker = airwheel.create_landmarker()

    car = Player()
    running = 1

    while running:

        delta = clock.tick()/1000 + 0.00001
        car.controls(delta)

        hand_ok = False
        if use_hand_control:
            hand_ok, hand_frame = hand_cap.read()
        if hand_ok:
            hand_frame = cv2.flip(hand_frame, 1)
            hf_h, hf_w = hand_frame.shape[:2]
            rgb = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - hand_start_time) * 1000)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            if hand_result.hand_landmarks and len(hand_result.hand_landmarks) == 2:
                wrist_points = sorted(
                    (int(landmarks[0].x * hf_w), int(landmarks[0].y * hf_h))
                    for landmarks in hand_result.hand_landmarks
                )
                raw_angle = airwheel.steering_angle(wrist_points[0], wrist_points[1])
                hand_smoothed_angle += HAND_ANGLE_SMOOTHING * (raw_angle - hand_smoothed_angle)

                if not hand_calibrated:
                    if hand_calibration_start is None:
                        hand_calibration_start = time.time()
                    hand_calibration_samples.append(hand_smoothed_angle)
                    if time.time() - hand_calibration_start >= HAND_CALIBRATION_SECONDS:
                        hand_baseline_angle = sum(hand_calibration_samples) / len(hand_calibration_samples)
                        hand_calibrated = True
                else:
                    hand_relative_angle = hand_smoothed_angle - hand_baseline_angle
                    car.angle = max(-0.9, min(0.9, hand_relative_angle / HAND_ANGLE_TO_RADIANS))

        for event in pg.event.get():
            if event.type == pg.QUIT: running = 0

        screen.fill((100,150,250))
        screen.blit(mountains_texture, (-65 - car.angle*82, 0))
        pg.draw.rect(screen, (55,125,55), (0, 60, 320, 120))
        vertical, draw_distance= 180, 1
        car.z = 100+40*math.sin(car.x/13)-60*math.sin(car.x/7)
        roadside_sprites = []

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

                    sprite_info = roadside_sprite_at(x)
                    if sprite_info:
                        sprite_texture, side, world_offset, max_height = sprite_info
                        sprite_scale = min(scale * 14, max_height / sprite_texture.get_height())
                        if sprite_scale > 0.15:
                            sprite_w = max(int(sprite_texture.get_width() * sprite_scale), 1)
                            sprite_h = max(int(sprite_texture.get_height() * sprite_scale), 1)
                            scaled_sprite = pg.transform.scale(sprite_texture, (sprite_w, sprite_h))
                            sprite_x = horizontal + side * world_offset * scale
                            roadside_sprites.append((scaled_sprite, int(sprite_x - sprite_w / 2), int(vertical - sprite_h)))

        for scaled_sprite, sprite_left, sprite_top in reversed(roadside_sprites):
            screen.blit(scaled_sprite, (sprite_left, sprite_top))

        screen.blit(car_sprite, (100, 120))

        if use_hand_control and not hand_calibrated:
            hud_text = "Raise both hands to calibrate steering" if hand_calibration_start is None else "Calibrating... hold straight"
            screen.blit(hud_font.render(hud_text, True, (255,255,0)), (10, 10))

        pg.display.update()
        await asyncio.sleep(0)

    if use_hand_control:
        hand_landmarker.close()
        hand_cap.release()

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
        self.velocity += -0.5*self.velocity*delta

        if pressed_keys[pg.K_w] or pressed_keys[pg.K_UP]:
            if self.velocity > -1:
                self.acceleration += 20*delta
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
        self.velocity = max(-30,min(40,self.velocity))
        self.velocity += self.acceleration*delta
        self.x += self.velocity*delta*math.cos(self.angle)
        self.y += self.velocity*math.sin(self.angle)*delta*100

if __name__ == "__main__":
    pg.init()
    asyncio.run(main())
    pg.quit()