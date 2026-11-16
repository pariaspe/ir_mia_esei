# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "opencv-python",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/s1_follow_line.slides.json",
    css_file="theme/slides-theme.css",
)

with app.setup:
    import marimo as mo

    ASSETS = mo.notebook_dir() / "assets"
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    # ------------------------------------------------------------------
    # Toy simulator parameters (a 2D stand-in for the Unibotics F1 car).
    # Not meant to be realistic, only to show the same trade-offs:
    #   - the camera sees the line `lookahead` metres ahead,
    #   - perception + communication add latency,
    #   - the steering does not react instantly (first-order lag),
    #   - tyres have limited grip: at high speed the car cannot turn sharply.
    # ------------------------------------------------------------------
    SIM_DT = 0.02  # 50 Hz control loop, like Frequency.tick(50)
    SIM_LATENCY_STEPS = 5  # 100 ms between image capture and command
    SIM_STEER_TAU = 0.25  # [s] steering response time constant
    SIM_W_MAX = 2.5  # [rad/s] max angular speed command
    SIM_GRIP = 6.0  # [m/s^2] max lateral acceleration before skidding
    SIM_FOV = np.deg2rad(80)  # camera horizontal field of view
    SIM_LOST_DIST = 1.5  # [m] distance from the line considered "lost"
    SIM_T_MAX = 120.0  # [s] give up after this time
    SIM_DS = 0.1


@app.function
def make_track(ds=SIM_DS):
    """Closed circuit with straights and curves of different radius."""
    phi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    r = 25 * (1 + 0.25 * np.cos(2 * phi) + 0.12 * np.sin(3 * phi))
    pts = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    s_new = np.arange(0, s[-1], ds)
    return np.stack(
        [np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])], axis=1
    )


@app.function
def simulate_follow_line(v_max, kp, kd=0.0, lookahead=3.0, kv=0.0, v_min=1.0):
    """Run the toy car around the track with the same control law students
    write in Unibotics:  w = -(Kp*e + Kd*de/dt),  v = max(v_min, v_max - Kv*|e|)
    where e in [-1, 1] is the normalised horizontal error of the line in the image.
    """
    track = make_track()
    n = len(track)
    x, y = track[0]
    th = np.arctan2(track[1, 1] - track[0, 1], track[1, 0] - track[0, 0])
    idx, progress = 0, 0
    look = int(lookahead / SIM_DS)
    pipeline = [0.0] * SIM_LATENCY_STEPS
    prev_e, w, t = 0.0, 0.0, 0.0
    log = {"x": [], "y": [], "t": [], "e": [], "w": [], "v": []}
    status = "timeout"
    for _ in range(int(SIM_T_MAX / SIM_DT)):
        # Where are we on the track?
        cand = (idx + np.arange(-20, 80)) % n
        d = np.linalg.norm(track[cand] - [x, y], axis=1)
        j = int(np.argmin(d))
        step = (cand[j] - idx) % n
        progress += step - n if step > n / 2 else step
        idx = cand[j]
        if d[j] > SIM_LOST_DIST:
            status = "lost"
            break
        if progress >= n:
            status = "lap"
            break
        # "Camera": project the line point ahead into a normalised image error
        tx, ty = track[(idx + look) % n] - [x, y]
        fx = np.cos(th) * tx + np.sin(th) * ty
        fy = -np.sin(th) * tx + np.cos(th) * ty
        ang = np.clip(np.arctan2(fy, fx), -1.5, 1.5)
        e = float(np.clip(-np.tan(ang) / np.tan(SIM_FOV / 2), -1, 1))
        pipeline.append(e)
        e_seen = pipeline.pop(0)
        # Controller (what the students write)
        de = (e_seen - prev_e) / SIM_DT
        prev_e = e_seen
        w_cmd = float(np.clip(-(kp * e_seen + kd * de), -SIM_W_MAX, SIM_W_MAX))
        v = max(v_min, v_max - kv * abs(e_seen))
        # Car dynamics: steering lag + grip limit
        w += (w_cmd - w) * SIM_DT / SIM_STEER_TAU
        w_real = float(np.clip(w, -SIM_GRIP / v, SIM_GRIP / v))
        x += v * np.cos(th) * SIM_DT
        y += v * np.sin(th) * SIM_DT
        th += w_real * SIM_DT
        t += SIM_DT
        for k, val in zip(log, (x, y, t, e_seen, w_cmd, v)):
            log[k].append(val)
    log = {k: np.array(val) for k, val in log.items()}
    return status, t, progress / n, log


@app.function
def plot_simulation(log, status):
    track = make_track()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    ax.plot(track[:, 0], track[:, 1], color="C3", lw=3, alpha=0.35, label="Red line")
    if len(log["x"]):
        sc = ax.scatter(log["x"], log["y"], c=log["v"], cmap="viridis", s=3, label="Car")
        fig.colorbar(sc, ax=ax, label="Speed [m/s]", shrink=0.8)
        if status == "lost":
            ax.plot(log["x"][-1], log["y"][-1], "kx", ms=12, mew=3, label="Line lost")
    ax.plot(track[0, 0], track[0, 1], "ks", ms=7, label="Start")
    ax.set_aspect("equal")
    ax.set_title("Top view")
    ax.legend(loc="lower left", fontsize=8)
    ax.axis("off")

    ax = axes[1]
    if len(log["t"]):
        ax.plot(log["t"], log["e"], color="C0", label="error e (normalised)")
        ax.plot(log["t"], log["w"] / SIM_W_MAX, color="C1", alpha=0.7, label="W command / W max")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time [s]")
    ax.set_title("What the controller sees and does")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


@app.function
def find_line(img_bgr, lower_hsv, upper_hsv, roi_top, roi_bottom):
    """Return (mask, cx, cy) where (cx, cy) is the centroid of the red pixels
    inside the horizontal band [roi_top, roi_bottom) of the image.
    cx, cy are None when no red pixel is found."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    roi = mask[roi_top:roi_bottom, :]
    m = cv2.moments(roi)
    if m["m00"] == 0:
        return mask, None, None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"] + roi_top
    return mask, cx, cy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 1 · Follow Line
    ### A first reactive robot, step by step

    **Goal for today:** leave the room with a car that completes a lap of the
    circuit on its own. It doesn't have to be fast.

    **Goal for next week:** make it fast.
    """)
    return


@app.cell
def _():
    mo.outline(label="Table of Contents")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plan for today

    | Time | Block | You end up with |
    |---|---|---|
    | 0:00 - 0:15 | Tour of Unibotics | The Follow Line exercise open and the simulator running |
    | 0:15 - 0:20 | **Step 1**: move the robot | A car that drives forward... and crashes |
    | 0:20 - 0:45 | **Step 2**: see the line | A mask of the red line and its centre in the image |
    | 0:45 - 1:10 | **Step 3**: close the loop (P controller) | **A first complete lap** |
    | 1:10 - 1:20 | **Step 4**: speed vs. stability | An intuition of what limits your lap time |
    | 1:20 - 1:30 | Homework and wrap-up | A plan to make your car faster |

    /// admonition | Different backgrounds, same goal
    You **don't need** to know control theory or be an expert programmer.
    Every step comes with the full solution (hidden until you click on it).
    Try first, then peek. If you code comfortably, look for the
    🚀 boxes: they contain extra challenges.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # The platform

    **RoboticsAcademy** (JdeRobot) is a collection of robotics exercises
    running on a real robotics simulator (Gazebo + ROS 2) inside a Docker
    container. **Unibotics** is the web front-end: you write Python in the
    browser, press ▶ and your code drives the simulated robot.

    1. Start the Docker container (you installed it before the session).
    2. Log in at Unibotics and open **Follow Line**.
    3. Pick the **Simple circuit** and press ▶ with the default code.

    What you see on screen:

    * **Code editor**: your program. It runs as a normal Python script.
    * **Simulator view**: the F1 car on the circuit (the "real world").
    * **Debug image** (WebGUI): whatever image *you* decide to show.
    * **Console**: output of your `print()` calls and errors.
    """)
    return


@app.cell
def _():
    image1 = mo.vstack(
        [
            mo.image(src=str(ASSETS / "robot_model.png"),
                     style={"display": "block", "margin": "0 auto"},
                     height=450),
            mo.md("Actuators: linear speed **V** and angular speed **W**").center(),
        ],
        align="center",
    )
    image2 = mo.vstack(
        [
            mo.image(src=str(ASSETS / "follow_line_camera.png"),
                     style={"display": "block", "margin": "0 auto"},
                     height=450),
            mo.md("Sensor: front RGB camera").center(),
        ],
        align="center",
    )
    mo.hstack([image1, image2], justify="center", align="center", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The API: everything you need fits in a few lines

    | Module | Call | What it does |
    |---|---|---|
    | `HAL` | `HAL.getImage()` | Returns the camera image (a NumPy array, **BGR** order) |
    | `HAL` | `HAL.setV(v)` | Sets the linear (forward) speed |
    | `HAL` | `HAL.setW(w)` | Sets the angular speed. **Positive turns left**, negative turns right |
    | `WebGUI` | `WebGUI.showImage(img)` | Shows any image in the debug window |
    | `Frequency` | `Frequency.tick(hz)` | Keeps the loop running at a fixed rate (default 50 Hz) |

    Every robot program has the same shape: **sense → think → act**, repeated forever.
    """)
    return


@app.cell
def _():
    mo.image(src=str(ASSETS / "control_arch.png"), width=900).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```python
    import HAL
    import WebGUI
    import Frequency

    while True:
        Frequency.tick(50)        # run at 50 Hz
        img = HAL.getImage()      # SENSE
        # ... THINK: decide v and w from the image ...
        HAL.setV(v)               # ACT
        HAL.setW(w)
    ```

    This is a **reactive** controller: at every instant the action depends
    **only on what the robot sees now**. No map, no plan, no memory of the past.
    Remember this when we discuss its weak points at the end.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 1 · Move the robot

    **Task:** make the car drive forward slowly, then make it drive in a circle.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "💡 Solution": mo.md(r"""
    ```python
    import HAL
    import Frequency

    while True:
        Frequency.tick(50)
        HAL.setV(1.0)    # forward
        HAL.setW(0.3)    # set to 0.0 to go straight; > 0 turns left
    ```
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// attention | Discuss
    Going straight, the car leaves the track at the first curve. Could you
    write a list of `setW` values over time that completes the lap? That is
    **open-loop** control. Why would it never work reliably?
    ///

    We need to **close the loop**: look at the line, and correct.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 2 · See the line

    The idea (visual servoing): **keep the red line in the centre of the
    image**. The reference of our controller is not a position in the world
    but a pixel position.
    """)
    return


@app.cell
def _():
    mo.image(src=str(ASSETS / "final_loop.png")).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Perception in four small operations:

    1. **Convert to HSV.** Hue isolates "redness" much better than raw BGR values.
    2. **Threshold the colour** with `cv2.inRange` to get a black-and-white *mask*.
    3. **Keep a horizontal band** (Region Of Interest, ROI) instead of the whole image.
    4. **Compute the centroid** of the white pixels with `cv2.moments`.

    Below, the same pipeline runs on a real camera frame from the simulator.
    Play with the sliders.
    """)
    return


@app.cell
def _():
    img_bgr = cv2.imread(str(ASSETS / "follow_line_camera.png"))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_bgr.shape[:2]
    return img_bgr, img_h, img_rgb, img_w


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.1 Colour filter: `cvtColor` + `inRange`

    In OpenCV, Hue goes from 0 to 179. Red sits at both ends (near 0 *and* near 179).
    """)
    return


@app.cell
def _():
    # Displayed in the cell below. Clicking it re-runs that cell, which
    # recreates the sliders with their initial values.
    reset_hsv = mo.ui.button(label="↺ Reset values")
    return (reset_hsv,)


@app.cell
def _(reset_hsv):
    reset_hsv  # dependency: re-create the sliders when the button is clicked
    hue_range = mo.ui.range_slider(start=0, stop=179, value=[0, 10], label="Hue")
    sat_range = mo.ui.range_slider(start=0, stop=255, value=[120, 255], label="Saturation")
    val_range = mo.ui.range_slider(start=0, stop=255, value=[70, 255], label="Value")
    mo.hstack(
        [
            mo.vstack([hue_range, sat_range, val_range], align="center"),
            reset_hsv
        ],
        justify="space-around",
        align="center",
    )
    return hue_range, sat_range, val_range


@app.cell
def _(hue_range, img_bgr, img_rgb, sat_range, val_range):
    lower_red = np.array([hue_range.value[0], sat_range.value[0], val_range.value[0]])
    upper_red = np.array([hue_range.value[1], sat_range.value[1], val_range.value[1]])
    _hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(_hsv, lower_red, upper_red)
    mo.hstack(
        [
            mo.image(img_rgb, caption="Camera image", width=400),
            mo.image(red_mask, caption=f"Mask  (lower={lower_red}, upper={upper_red})", width=400),
        ],
        justify="center",
    )
    return lower_red, upper_red


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.2 Region of interest and centroid

    Which rows of the image should we look at? **Near rows** (bottom) tell you
    where the line is *now*; **far rows** (towards the horizon) tell you where
    it is *going*. Keep this in mind: it becomes important when you go fast.
    """)
    return


@app.cell
def _():
    # Defined here but displayed next to the image in the cell below
    # (a cell cannot read the value of a slider it creates itself).
    roi_center = mo.ui.slider(0.50, 0.98, step=0.01, value=0.80,
                              label="ROI position", show_value=True)
    roi_height = mo.ui.slider(0.02, 0.30, step=0.01, value=0.10,
                              label="ROI height", show_value=True)
    return roi_center, roi_height


@app.cell
def _(img_bgr, img_h, img_w, lower_red, roi_center, roi_height, upper_red):
    roi_top = int(img_h * max(0.0, roi_center.value - roi_height.value / 2))
    roi_bottom = int(img_h * min(1.0, roi_center.value + roi_height.value / 2))
    _, line_cx, line_cy = find_line(img_bgr, lower_red, upper_red, roi_top, roi_bottom)

    # Debug drawing, like you will do with WebGUI.showImage()
    _dbg = img_bgr.copy()
    cv2.rectangle(_dbg, (0, roi_top), (img_w - 1, roi_bottom), (255, 200, 0), 3)
    cv2.line(_dbg, (img_w // 2, 0), (img_w // 2, img_h), (255, 0, 0), 3)
    if line_cx is not None:
        _c = (int(line_cx), int(line_cy))
        cv2.line(_dbg, (img_w // 2, _c[1]), _c, (0, 255, 255), 4)
        cv2.circle(_dbg, _c, 12, (0, 255, 0), -1)
    mo.hstack(
        [
            mo.image(cv2.cvtColor(_dbg, cv2.COLOR_BGR2RGB), width=520),
            mo.vstack(
                [
                    roi_center,
                    roi_height,
                    mo.md("Values are fractions of the image height (0 = top, 1 = bottom)."),
                ],
                gap=1,
            ),
        ],
        justify="center",
        align="center",
        gap=2,
    )
    return line_cx, roi_bottom, roi_top


@app.cell(hide_code=True)
def _(line_cx, roi_bottom, roi_top):
    mo.md(
        f"""
    Cyan box: ROI (rows {roi_top}-{roi_bottom}). Blue: image centre. Green dot: centroid
    of the line, **cx = {line_cx:.1f} px**.
    """
        if line_cx is not None
        else "⚠️ **No red pixels in the ROI**: the line is lost! Your code must handle this case."
    ).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.3 Your turn in Unibotics

    **Task:** read the image, compute the mask and the centroid, draw it, and
    show it with `WebGUI.showImage`. Keep the car **stopped** (`v = 0`) for now.
    Use `print(cx)` to check the values.

    /// warning | Common pitfalls
    * `HAL.getImage()` returns **BGR**, not RGB → use `cv2.COLOR_BGR2HSV`.
    * If the mask is empty, `m["m00"]` is 0 and you divide by zero.
    * Image coordinates: `x` grows to the right, `y` grows **downwards**.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "💡 Solution": mo.md(r"""
    ```python
    import HAL
    import WebGUI
    import Frequency
    import cv2
    import numpy as np

    LOWER_RED = np.array([0, 120, 70])
    UPPER_RED = np.array([10, 255, 255])

    while True:
        Frequency.tick(50)
        img = HAL.getImage()
        h, w = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)

        top, bottom = int(0.75 * h), int(0.85 * h)
        m = cv2.moments(mask[top:bottom, :])
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = (top + bottom) // 2
            cv2.circle(img, (int(cx), cy), 6, (0, 255, 0), -1)
            print(cx)

        cv2.rectangle(img, (0, top), (w - 1, bottom), (255, 200, 0), 2)
        WebGUI.showImage(img)
    ```
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 3 · Close the loop

    **Error**: how far the line is from the centre of the image, normalised
    to $[-1, 1]$ so that it does not depend on the image resolution:

    $$
    e = \frac{c_x - W_{img}/2}{W_{img}/2}
    \qquad
    \begin{cases}
    e > 0 & \text{line to the right} \\
    e < 0 & \text{line to the left}
    \end{cases}
    $$

    **Proportional (P) control**: turn proportionally to the error, *towards* the line.
    Since positive `W` turns **left**, we need a minus sign:

    $$
    W = -K_p \, e \qquad\qquad V = \text{constant}
    $$
    """)
    return


@app.cell
def _():
    kp_img_slider = mo.ui.slider(0, 5, step=0.1, value=1.0, label="Kp")
    kp_img_slider.center()
    return (kp_img_slider,)


@app.cell(hide_code=True)
def _(img_w, kp_img_slider, line_cx):
    if line_cx is None:
        _out = mo.md("⚠️ No line in the ROI: go back to step 2.2 and move the ROI.")
    else:
        _e = (line_cx - img_w / 2) / (img_w / 2)
        _w_cmd = -kp_img_slider.value * _e
        _dir = "left ⬅️" if _w_cmd > 0 else ("right ➡️" if _w_cmd < 0 else "straight ⬆️")
        _out = mo.hstack(
            [
                mo.stat(f"{line_cx:.0f} px", label="cx"),
                mo.stat(f"{_e:+.3f}", label="error e"),
                mo.stat(f"{_w_cmd:+.3f} rad/s", label="W = -Kp·e"),
                mo.stat(_dir, label="The car turns"),
            ],
            justify="center",
            gap=2,
        )
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For the frame above (from steps 2.1-2.2), these are the error and the command
    the car would send. Move the ROI in 2.2 and see how the error changes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.1 Your turn: first lap

    **Task:** combine step 2 and the control law. Start **slow** (`V = 1`) with a
    small `Kp`, and increase until you complete a lap.

    /// attention | If the car turns away from the line...
    ...your sign is wrong. This is the most common bug in robotics: always check
    sign conventions!
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    # Single source of truth: the same file students paste into Unibotics.
    _code = (mo.notebook_dir().parent / "unibotics" / "follow_line_basic.py").read_text()
    mo.accordion(
        {
            "💡 Solution: a complete, simple Follow Line": mo.vstack(
                [
                    mo.md(f"```python\n{_code}```"),
                    mo.md(r"""
    `last_error` is the only "memory" in this controller. It is a tiny trick
    to recover when the line goes out of view in a sharp curve.
    """),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | ✅ Checkpoint
    Before moving on, **everyone** should have a car that completes a lap of
    the Simple circuit, however slowly. Help your neighbour!
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 4 · Speed vs. stability

    Tuning in Gazebo is slow: each trial takes a full lap. Below is a **toy 2D
    simulator** running exactly the same control law. It is much simpler than
    Gazebo, but it has the same ingredients that make real robots hard:

    * **Latency**: the image you process is already ~100 ms old.
    * **Inertia**: the car does not turn the instant you ask it to.
    * **Grip**: at high speed the tyres cannot follow a tight curve.

    Numbers here **won't transfer** directly to Unibotics. The *intuition* will.
    """)
    return


@app.cell
def _():
    sim_v = mo.ui.slider(1, 14, step=0.5, value=3, label="V (max speed) [m/s]", show_value=True)
    sim_kp = mo.ui.slider(0, 8, step=0.1, value=1.0, label="Kp", show_value=True)
    sim_look = mo.ui.slider(0.5, 8, step=0.5, value=3,
                            label="Lookahead [m]  (≈ ROI position)", show_value=True)
    sim_kd = mo.ui.slider(0, 0.5, step=0.01, value=0.0, label="🚀 Kd", show_value=True)
    sim_kv = mo.ui.slider(0, 40, step=1, value=0,
                          label="🚀 Kv (slow down when error is large)", show_value=True)
    mo.hstack(
        [
            mo.vstack([mo.md("**Step 4 controls**"), sim_v, sim_kp, sim_look]),
            mo.vstack([mo.md("**Homework preview**"), sim_kd, sim_kv]),
        ],
        justify="space-around",
        gap=0
    )
    return sim_kd, sim_kp, sim_kv, sim_look, sim_v


@app.cell
def _(sim_kd, sim_kp, sim_kv, sim_look, sim_v):
    sim_status, sim_time, sim_progress, sim_log = simulate_follow_line(
        v_max=sim_v.value,
        kp=sim_kp.value,
        kd=sim_kd.value,
        lookahead=sim_look.value,
        kv=sim_kv.value,
    )
    if sim_status == "lap":
        _banner = mo.callout(mo.md(f"🏁 **Lap completed in {sim_time:.1f} s**"), kind="success")
    elif sim_status == "lost":
        _banner = mo.callout(
            mo.md(f"💥 **Line lost** after {sim_time:.1f} s ({sim_progress:.0%} of the lap)"),
            kind="danger",
        )
    else:
        _banner = mo.callout(
            mo.md(f"🐢 Too slow: only {sim_progress:.0%} of the lap in {SIM_T_MAX:.0f} s"),
            kind="warn",
        )
    mo.vstack([_banner, plot_simulation(sim_log, sim_status)], gap=0)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiments (5 minutes each, discuss in pairs)

    1. **Kp too low.** With `V = 3`, set `Kp = 0.3`. What happens in curves? Why?
    2. **Kp too high.** With `V = 4`, raise `Kp` from 2 to 8. The car may still
       finish, but look at the orange `W` curve. What shape appears, and what
       would it mean for a real car? (Compare with the P demo from the last session.)
    3. **Speed limit.** Keep only `V` and `Kp`. What is the **fastest lap**
       you can get? What stops you from going faster?
    4. **Looking ahead.** At the speed from experiment 3, try a very short
       lookahead (0.5-1 m), then a very long one (8 m). Which one fails, and why?
       In the top view, does the long one follow the line in curves, or cut the corners?
    5. 🚀 **Homework preview.** Turn on `Kv`: the car slows down when the error
       is large (in curves) and speeds up on straights. How much does your lap
       time improve?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Homework: make it faster 🏎️

    **Due before Session 2 (03/12)** · individual

    Starting from today's code, **minimise your lap time** on the Simple
    circuit, without leaving the line. Then check that it still works on a
    second circuit of your choice (no re-tuning allowed).

    Ideas, roughly from easiest to hardest. You don't have to do all of them:

    | Level | Idea | Hint |
    |---|---|---|
    | ⭐ | Tune `V` and `Kp` systematically | Change one thing at a time and write down the results in a table |
    | ⭐ | Choose a better ROI | Near vs. far rows (step 4, experiment 4) |
    | ⭐⭐ | Add a **D term** | `d_error = error - prev_error`; it damps oscillations (see the PID demo) |
    | ⭐⭐ | **Speed scheduling** | `v = V_MAX - KV * abs(error)`: slow in curves, fast on straights |
    | ⭐⭐⭐ | **Two ROIs** | Steer with the near one; use the far one to *anticipate* curves and brake earlier |
    | ⭐⭐⭐ | **Robust perception** | Handle both ends of the red hue, filter noise, recover when the line is lost |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Deliverable

    1. **Your code** (a single `.py` file).
    2. **Half a page** (PDF) with:
        * your **control law**, written as an equation, and what each parameter does;
        * your **best lap time**, and the circuit you used to check robustness;
        * its **weak points**: when and why does it fail?
    3. **Pairs and exercise choice** for the second assignment: team up with
       someone with a **different background** and send your **top 3** exercises.

    /// admonition | Question to think about (for the ½ page and for Session 2)
    Your controller is purely **reactive**: it only uses the current image.
    What could the car do better if it **remembered** the circuit after the
    first lap? What would that require?
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
