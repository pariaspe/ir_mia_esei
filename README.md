# Intelligent Robotics I: practice material (UVigo)

Practice material for *Intelligent Robotics I* (inter-university master's course UDC / USC / UVigo, 2026/27).
The notebooks are published as a website on GitHub Pages; they run in the browser, with no install.
The practice sessions use [Unibotics](https://unibotics.org), which runs [JdeRobot RoboticsAcademy](https://jderobot.github.io/RoboticsAcademy/).

## Contents
| Folder | What it holds |
|---|---|
| `notebooks/` | Interactive [marimo](https://marimo.io) notebooks used in class. Images are in `notebooks/public/` and slide layouts in `notebooks/layouts/` |
| `unibotics/` | Plain Python files to paste into the Unibotics editor (solutions and templates) |
| `site/` | The website: home page template and build script |
| `docs/` | The course plan, milestones, evaluation and exercise pool |

### Notebooks
| Notebook | Content |
|---|---|
| `00_cv_demo.py` | OpenCV basics: colour spaces, blur, threshold, colour segmentation |
| `00_pid_demo.py` | Robot model, control loop, interactive P / PD / PID, visual servoing |
| `s1_follow_line.py` | Session 1: guided Follow Line, from the camera image to a P controller, plus a small 2D simulator for exploring speed vs. stability |

## Running the notebooks
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/marimo edit notebooks/s1_follow_line.py
```

With [uv](https://docs.astral.sh/uv/) installed, no venv is needed. Each notebook declares its own dependencies:
```bash
uvx marimo edit --sandbox notebooks/s1_follow_line.py
```

Use `marimo run` instead of `marimo edit` to show a notebook as an app, or switch to slide view from the notebook menu.

## Website
`.github/workflows/pages.yml` builds and publishes the site on every push to `main`.
To build it locally:
```bash
.venv/bin/pip install -r requirements.txt && .venv/bin/python -m playwright install chromium
.venv/bin/python site/build.py
python3 -m http.server --directory _site    # open http://localhost:8000
```
To publish a new notebook, add it to `NOTEBOOKS` in `site/build.py`.
Notebooks with a slides layout are published twice, as slides and as a normal (vertical) notebook. The home page links both, and a button in the top-left corner of each page switches between them.
