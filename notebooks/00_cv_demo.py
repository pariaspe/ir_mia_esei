# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "opencv-python",
#     "numpy",
#     "pillow",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(
    width="medium",
    layout_file="layouts/00_cv_demo.slides.json",
    css_file="theme/slides-theme.css"
)

with app.setup:
    import marimo as mo

    # notebook_location() is the notebook folder locally and its URL in the browser (WASM)
    ASSETS = mo.notebook_location() / "public"
    import cv2
    import numpy as np

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # OpenCV Vision Examples

    A quick interactive tour of a few core OpenCV building blocks —
    `cvtColor`, `GaussianBlur`, `threshold`, and `inRange` — applied to
    `assets/follow_line_camera.png`. Move the sliders to see how each function reacts.
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
