"""Finite State Machine (FSM) template for Unibotics / RoboticsAcademy.

It is shown inside notebooks/s2_architecture.py.

How to use it:
  1. Replace read_sensors() with the sensors of your exercise.
  2. Write one function per state (the boxes of your diagram). Each one
     ACTS (sends commands to the robot) and RETURNS the name of the next
     state, or its own name to stay where it is.
  3. Register every state in STATES and choose the initial state.
  4. Keep what the robot must remember in `memory` (your world model).

The motion commands below (HAL.setV / HAL.setW) are the ones of wheeled
robots. Other robots (drones, arms) use different calls: check the API
section of your exercise in Unibotics.
"""
import time

import Frequency
import HAL

# ---------------- Parameters: tune these ----------------
LOOP_HZ = 20          # iterations per second of the main loop
FORWARD_V = 0.3       # linear speed when moving forward
TURN_W = 0.5          # angular speed when turning
TURN_TIME = 2.0       # seconds spent turning
# ---------------------------------------------------------

# ---------------- World model: what the robot remembers ----------------
memory = {
    "state_start": time.time(),  # when the current state was entered
    # Add what your architecture needs, for example:
    # "visited": set(), "path": [], "targets_found": [], "laps": 0,
}


def time_in_state():
    """Seconds since the current state was entered."""
    return time.time() - memory["state_start"]


# ---------------- Sensing ----------------
def read_sensors():
    """Read every sensor ONCE per iteration and return them in a dict."""
    return {
        # Replace with the sensors of your exercise (see its API section), e.g.
        # "laser": HAL.getLaserData(),
        # "pose": HAL.getPose3d(),
        # "image": HAL.getImage(),
    }


def obstacle_ahead(sensors):
    """Example of an event computed from the sensors. Replace with your own."""
    return False


# ---------------- States: one function per state ----------------
def forward(sensors):
    HAL.setV(FORWARD_V)
    HAL.setW(0.0)
    if obstacle_ahead(sensors):
        return "TURN"
    return "FORWARD"


def turn(sensors):
    HAL.setV(0.0)
    HAL.setW(TURN_W)
    if time_in_state() > TURN_TIME:
        return "FORWARD"
    return "TURN"


def stop(sensors):
    HAL.setV(0.0)
    HAL.setW(0.0)
    return "STOP"


STATES = {
    "FORWARD": forward,
    "TURN": turn,
    "STOP": stop,
}
state = "FORWARD"  # initial state

# ---------------- Main loop: you should not need to change it ----------------
print("Starting in", state)
while True:
    Frequency.tick(LOOP_HZ)
    sensors = read_sensors()
    next_state = STATES[state](sensors)
    if next_state != state:
        print(f"[{time.strftime('%H:%M:%S')}] {state} -> {next_state}")
        memory["state_start"] = time.time()
        state = next_state
