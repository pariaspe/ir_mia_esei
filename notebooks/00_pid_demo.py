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
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    # notebook_location() is the notebook folder locally and its URL in the browser (WASM)
    ASSETS = mo.notebook_location() / "public"
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    def read_bytes(path) -> bytes:
        """Read a file from disk, or download it when running in the browser (WASM)."""
        import sys
        if "pyodide" not in sys.modules:
            from pathlib import Path
            return Path(path).read_bytes()
        from js import Uint8Array, XMLHttpRequest
        req = XMLHttpRequest.new()
        req.open("GET", str(path), False)  # synchronous: fine inside the Pyodide web worker
        req.responseType = "arraybuffer"
        req.send(None)
        if req.status != 200:
            raise FileNotFoundError(f"{path} (HTTP {req.status})")
        return Uint8Array.new(req.response).to_bytes()

    def load_image(path):
        """Like cv2.imread (BGR image), but also works in the browser."""
        return cv2.imdecode(np.frombuffer(read_bytes(path), np.uint8), cv2.IMREAD_COLOR)

    # Plant: simple mass-spring-damper, m*x'' + b*x' + k*x = u (fixed, not user-tunable)
    PLANT_M = 1.0
    PLANT_B = 0.5
    PLANT_K = 1.0
    REFERENCE = 1.0
    DT = 0.02
    DURATION = 15.0
    N_STEPS = int(DURATION / DT)


@app.function
def simulate(kp, ki=0.0, kd=0.0):
    t = np.arange(N_STEPS) * DT
    x = np.zeros(N_STEPS)
    u_hist = np.zeros(N_STEPS)
    pos = 0.0
    vel = 0.0
    integral_error = 0.0
    for i in range(N_STEPS):
        error = REFERENCE - pos
        integral_error += error * DT
        # Derivative acts on velocity (measurement), not d(error)/dt, to
        # avoid a derivative "kick" at t=0 when the step reference appears.
        u = kp * error + ki * integral_error - kd * vel
        accel = (u - PLANT_B * vel - PLANT_K * pos) / PLANT_M
        vel += accel * DT
        pos += vel * DT
        x[i] = pos
        u_hist[i] = u
    return t, x, u_hist


@app.function
def plot_response(t, y, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(REFERENCE, color="gray", linestyle="--", label="Reference (step)")
    ax.plot(t, y, color="C0", label="Response")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


@app.function
def plot_effort_comparison(t, efforts, titles):
    fig, axes = plt.subplots(1, len(efforts), figsize=(4 * len(efforts), 4))
    for ax, u, title in zip(axes, efforts, titles):
        ax.plot(t, u, color="C1")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Control effort u")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # A practical approach to Autonomous Systems
    """)
    return


@app.cell
def _():
    mo.outline(label="Table of Contents")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Our Robot Model
    """)
    return


@app.cell
def _():

    image1 = mo.vstack([
        mo.image(src=str(ASSETS / "robot_model.png"), style={"display": "block", "margin": "0 auto"}),
        mo.md("Robot Actuators: Holonomic car").center()
    ], align="center")

    image2 = mo.vstack([
        mo.image(src=str(ASSETS / "follow_line_camera.png"), style={"display": "block", "margin": "0 auto"}),
        mo.md("Robot Sensors: RGB camera").center()
    ], align="center")

    mo.hstack([image1, image2], justify="center", align="center", gap=2)
    return


@app.cell
def _():
    mo.md(r"""
    ## Holonomic Robot Dynamic Model
    """)
    return


@app.cell
def _():
    actuation = mo.md(r"""
    $$
    \mathbf{u} = \begin{bmatrix} v_x \\ w \end{bmatrix}
    $$
    """)
    actuation_txt = mo.md(r"""
    **Actuation:**
    """)
    astack = mo.vstack([actuation_txt, actuation], justify='center', align="center", gap=0)

    state = mo.md(r"""
    $$
    \mathbf{x} = \begin{bmatrix} x \\ y \\ \theta \end{bmatrix}
    $$
    """)
    state_txt = mo.md(r"""
    **State:**
    """)
    sstack = mo.vstack([state_txt, state], justify='center', align="center", gap=0)

    dynamic = mo.md(r"""
    $$
    \mathbf{\dot{x}} = \begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix}
    $$
    """)
    dynamic_txt = mo.md(r"""
    **Dynamic:**
    """)
    dstack = mo.vstack([dynamic_txt, dynamic], justify='center', align="center", gap=0)

    mo.hstack([astack, sstack, dstack], justify="center", align="center", gap=5)
    return actuation, state


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## State Estimation & Control
    """)
    return


@app.cell
def _(actuation, state):

    sensors_api = mo.md(r"""
    `HAL.getPose()` 
    `HAL.getImage()`
    """)
    sensors_api_stack = mo.vstack([sensors_api, state], justify='center', align="center", gap=0)

    control_img = mo.image(src=str(ASSETS / "control_arch.png"))

    actuators_api = mo.md(r"""
    `HAL.setV(velocity)`
    `HAL.setW(velocity)`
    """)
    actuators_api_stack = mo.vstack([actuators_api, actuation],
                                    justify='center', align="center", gap=0)

    mo.hstack([sensors_api_stack, control_img, actuators_api_stack],
              justify="center", align="center", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Control loop
    """)
    return


@app.cell
def _():
    p_formula = mo.md(r"""
    $$
    u(t) = K_p e(t)
    $$
    """)

    p_before_img = mo.image(src=str(ASSETS / "p_control_before.jpeg"), width=700)

    mo.hstack([p_before_img, p_formula], justify="center", align="center", gap=2)
    return (p_formula,)


@app.cell
def _(p_formula):
    p_after_img = mo.image(src=str(ASSETS / "p_control_after.jpeg"), width=700)

    mo.hstack([p_after_img, p_formula], justify="center", align="center", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PID Controller, Step by Step
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## P — Proportional control

    $u(t) = K_p e(t)$

    The control effort is simply proportional to how far the system is
    from the reference. Higher `Kp` reacts more aggressively (and tends to
    overshoot/oscillate more), but notice that the response **settles
    below the reference** — proportional control alone can't fully close
    the gap.
    """)
    return


@app.cell
def _():
    p_kp_slider = mo.ui.slider(0, 10, step=0.1, value=2, label="Kp")
    p_kp_slider
    return (p_kp_slider,)


@app.cell
def _(p_kp_slider):
    _t, _x, _ = simulate(kp=p_kp_slider.value)
    plot_response(_t, _x, f"P control (Kp={p_kp_slider.value})")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PD — Proportional + Derivative

    $u(t) = K_p e(t) + K_d \frac{d e(t)}{dt}$

    Adding a derivative term damps the response — it resists fast change,
    reducing overshoot and oscillation for a given `Kp`. Try raising `Kd`
    and watch the ringing settle down faster. But notice the **steady-state
    offset stays exactly the same as the P-only case** — derivative action
    doesn't fix the final error, only the transient behavior.
    """)
    return


@app.cell
def _():
    pd_kp_slider = mo.ui.slider(0, 10, step=0.1, value=2, label="Kp")
    pd_kd_slider = mo.ui.slider(0, 5, step=0.1, value=1, label="Kd")
    mo.hstack([pd_kp_slider, pd_kd_slider])
    return pd_kd_slider, pd_kp_slider


@app.cell
def _(pd_kd_slider, pd_kp_slider):
    _t, _x, _ = simulate(kp=pd_kp_slider.value, kd=pd_kd_slider.value)
    plot_response(
        _t, _x, f"PD control (Kp={pd_kp_slider.value}, Kd={pd_kd_slider.value})"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PID — Proportional + Integral + Derivative

    $u(t) = K_p e(t) + K_i \int e(t) \,dt + K_d \frac{d e(t)}{dt}$

    The integral term accumulates error over time and keeps pushing until
    it's gone — this is what finally eliminates the steady-state offset
    left behind by P and PD. Try raising `Ki` from zero and watch the
    response converge all the way to the reference. (Push it too high,
    though, and you'll see it start to overshoot and oscillate — integral
    action isn't free.)
    """)
    return


@app.cell
def _():
    pid_kp_slider = mo.ui.slider(0, 10, step=0.1, value=2, label="Kp")
    pid_ki_slider = mo.ui.slider(0, 5, step=0.1, value=1, label="Ki")
    pid_kd_slider = mo.ui.slider(0, 5, step=0.1, value=1, label="Kd")
    mo.hstack([pid_kp_slider, pid_ki_slider, pid_kd_slider])
    return pid_kd_slider, pid_ki_slider, pid_kp_slider


@app.cell
def _(pid_kd_slider, pid_ki_slider, pid_kp_slider):
    _t, _x, _ = simulate(
        kp=pid_kp_slider.value, ki=pid_ki_slider.value, kd=pid_kd_slider.value
    )
    plot_response(
        _t,
        _x,
        f"PID control (Kp={pid_kp_slider.value}, "
        f"Ki={pid_ki_slider.value}, Kd={pid_kd_slider.value})",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Control effort

    `u(t)` is the raw command sent to the plant (e.g. a force or torque) —
    not the system's output, but the actuator's workload. It matters
    because real actuators saturate: a controller that looks great on
    paper can be useless if it demands more force/torque than the
    hardware can deliver. More aggressive gains generally mean larger
    effort, and the `D` term in particular can produce sharp spikes since
    it reacts to how fast things are changing.

    The three plots below reuse the gains currently set in the P, PD, and
    PID sections above, so you can compare effort directly against the
    responses you were just looking at.
    """)
    return


@app.cell
def _(
    p_kp_slider,
    pd_kd_slider,
    pd_kp_slider,
    pid_kd_slider,
    pid_ki_slider,
    pid_kp_slider,
):
    _t, _, _u_p = simulate(kp=p_kp_slider.value)
    _, _, _u_pd = simulate(kp=pd_kp_slider.value, kd=pd_kd_slider.value)
    _, _, _u_pid = simulate(
        kp=pid_kp_slider.value, ki=pid_ki_slider.value, kd=pid_kd_slider.value
    )
    plot_effort_comparison(
        _t,
        [_u_p, _u_pd, _u_pid],
        [
            f"P (Kp={p_kp_slider.value})",
            f"PD (Kp={pd_kp_slider.value}, Kd={pd_kd_slider.value})",
            f"PID (Kp={pid_kp_slider.value}, "
            f"Ki={pid_ki_slider.value}, Kd={pid_kd_slider.value})",
        ],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Image-Based Visual Servoing

    But where should I go to? We don't know the racing track... What is the reference for our controller?

    * Let's use our sensors: keep the red line in the middle of the image at any time.
    * Close a control loop using pixels instead of physical coordinates.

    [ima](assets/visual_servoing.png)
    """)
    return


@app.cell
def _():
    mo.image(src=str(ASSETS / "final_loop.png"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Perception
    Computer Vision techniques to detect the red line over the image. Color filer.
    """)
    return


@app.cell
def _():
    img_bgr = load_image(ASSETS / "follow_line_camera.png")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return (img_rgb,)


@app.cell
def _(img_rgb):
    mo.image(img_rgb, caption="Original", width=500)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `cvtColor` — color space conversion

    Pick a target color space. Grayscale collapses to a single channel;
    HSV/LAB are shown as their three separate channels.
    """)
    return


@app.cell
def _():
    colorspace_dropdown = mo.ui.dropdown(
        options=["Grayscale", "HSV", "LAB"],
        value="Grayscale",
        label="Convert to",
    )
    colorspace_dropdown
    return (colorspace_dropdown,)


@app.cell
def _(colorspace_dropdown, img_rgb):
    _flags = {
        "Grayscale": cv2.COLOR_RGB2GRAY,
        "HSV": cv2.COLOR_RGB2HSV,
        "LAB": cv2.COLOR_RGB2LAB,
    }
    _converted = cv2.cvtColor(img_rgb, _flags[colorspace_dropdown.value])

    if _converted.ndim == 2:
        _preview = mo.image(_converted, caption=colorspace_dropdown.value, width=500)
    else:
        _channel_names = {
            "HSV": ("Hue", "Saturation", "Value"),
            "LAB": ("L", "A", "B"),
        }[colorspace_dropdown.value]
        _preview = mo.hstack(
            [
                mo.image(
                    _converted[:, :, i], vmin=0, vmax=255, caption=name, width=250, style={"display": "block", "margin": "0 auto"}
                )
                for i, name in enumerate(_channel_names)
            ]
        )
    _preview
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `GaussianBlur` — smoothing

    Larger kernels blur the image more aggressively. The kernel size must
    be odd, so the slider is clamped up if it lands on an even value.
    """)
    return


@app.cell
def _():
    blur_slider = mo.ui.slider(1, 31, step=2, value=5, label="Kernel size")
    blur_slider
    return (blur_slider,)


@app.cell
def _(blur_slider, img_rgb):
    _k = blur_slider.value
    _k = _k if _k % 2 == 1 else _k + 1
    _blurred = cv2.GaussianBlur(img_rgb, (_k, _k), 0)
    mo.hstack(
        [
            mo.image(img_rgb, caption="Original", width=400),
            mo.image(_blurred, caption=f"GaussianBlur (k={_k})", width=400),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `threshold` — binarization

    Applied to a grayscale version of the image.
    """)
    return


@app.cell
def _(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return (img_gray,)


@app.cell
def _():
    threshold_slider = mo.ui.slider(0, 255, value=127, label="Threshold value")
    threshold_type = mo.ui.dropdown(
        options=["Binary", "Binary Inverted", "Otsu"],
        value="Binary",
        label="Threshold type",
    )
    mo.hstack([threshold_slider, threshold_type])
    return threshold_slider, threshold_type


@app.cell
def _(img_gray, threshold_slider, threshold_type):
    _flags = {
        "Binary": cv2.THRESH_BINARY,
        "Binary Inverted": cv2.THRESH_BINARY_INV,
        "Otsu": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    }
    _, _thresholded = cv2.threshold(
        img_gray, threshold_slider.value, 255, _flags[threshold_type.value]
    )
    mo.hstack(
        [
            mo.image(img_gray, caption="Grayscale", width=400),
            mo.image(_thresholded, caption="Thresholded", width=400),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `inRange` — color segmentation

    Converts to HSV and keeps only pixels whose H/S/V fall inside the
    chosen ranges. Defaults are tuned to pick out the red region in
    `assets/follow_line_camera.png`.
    """)
    return


@app.cell
def _(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return (img_hsv,)


@app.cell
def _():
    hue_range = mo.ui.range_slider(start=0, stop=179, value=[0, 10], label="Hue")
    sat_range = mo.ui.range_slider(
        start=0, stop=255, value=[120, 255], label="Saturation"
    )
    val_range = mo.ui.range_slider(start=0, stop=255, value=[70, 255], label="Value")
    mo.vstack([hue_range, sat_range, val_range])
    return hue_range, sat_range, val_range


@app.cell
def _(hue_range, img_hsv, img_rgb, sat_range, val_range):
    _lower = np.array([hue_range.value[0], sat_range.value[0], val_range.value[0]])
    _upper = np.array([hue_range.value[1], sat_range.value[1], val_range.value[1]])
    _mask = cv2.inRange(img_hsv, _lower, _upper)
    _segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=_mask)
    mo.hstack(
        [
            mo.image(img_rgb, caption="Original", width=300),
            mo.image(_mask, caption="Mask", width=300),
            mo.image(_segmented, caption="Segmented", width=300),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
