# AirWheel

Steer a browser racing game with nothing but your hands in the air, no wheel, no gamepad, just a webcam.

I saw a reel of someone doing this on Instagram and figured it wasn't that hard to build myself, so here it is.

## How it works

It's pretty simple under the hood. MediaPipe watches your webcam feed and finds your two wrists. The angle of the line between them is basically your steering angle, tilt your hands like you're holding a wheel and turning it, and the script translates that into arrow key presses. Hold both hands up and it also holds the "up" key down for you, so you don't need to worry about the gas.

There's a short calibration step when you launch it. Just hold your hands wherever feels natural and comfortable and the app locks that in as "straight" for you, since nobody's hands sit at a perfect 0 degrees.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run it:

```
python3 src/Main.py
```

First run will download the hand tracking model automatically (a few MB), so give it a second.

## Using it

1. Open a racing game in your browser, something that only needs arrow keys works best. [Racing Limits](https://www.crazygames.com/game/racing-limits) on CrazyGames is a good one to start with.
2. Click on the browser tab so it has keyboard focus.
3. Run the script, allow camera access if macOS asks.
4. Hold both hands up in front of the camera and stay still for a second while it calibrates.
5. Tilt your hands to steer. Straighten them back out to go straight.
6. Press `q` in the camera window to quit, or `c` any time to recalibrate (only works if the camera window itself has focus).

## Tuning

If steering feels too twitchy or too sluggish, the knobs are at the top of `src/Main.py`:

- `ENTER_TURN_DEG` / `EXIT_TURN_DEG` — how far you need to tilt to trigger a turn, and how far back to release it. There's a gap between the two on purpose so it doesn't flicker between left/right when you're near the edge.
- `ANGLE_SMOOTHING` — how much weight new webcam readings get vs the previous smoothed value. Lower it if the angle feels jittery.
- `CALIBRATION_SECONDS` — how long the calibration hold takes.

## Credit

Inspired by a project I saw from [@jeyy.sh](https://www.instagram.com/jeyy.sh/) on Instagram.
