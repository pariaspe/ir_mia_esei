"""Build the course website into _site/ (served by GitHub Pages).

Usage (from the repo root):
    .venv/bin/python site/build.py
    python -m http.server --directory _site     # preview at http://localhost:8000

Each notebook in NOTEBOOKS is exported as a WebAssembly app (it runs in the
visitor's browser, no server needed). Notebooks not listed are not published.
Notebooks with a slides layout are also exported in the vertical layout, with
a button in each page to switch views and links to both on the home page.

--offline bundles the Python runtime and packages into the site. Without it,
the browser downloads them from cdn.jsdelivr.net, which fails for OpenCV
(compressed responses of large files return 503). It needs Playwright:
    pip install -r requirements.txt && python -m playwright install chromium
"""

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"

# (file in notebooks/, title, description)
NOTEBOOKS = [
    ("s1_follow_line.py", "Session 1: Follow Line",
     "Guided Follow Line: from the camera image to a P controller, plus a small 2D simulator for exploring speed vs. stability."),
    # ("00_cv_demo.py", "OpenCV basics",
    #  "Colour spaces, blur, threshold and colour segmentation."),
    # ("00_pid_demo.py", "PID control",
    #  "Robot model, control loop, interactive P / PD / PID and visual servoing."),
]

# Notebooks with a slides layout are exported twice: as slides (<name>.html)
# and without the layout (<name>.vertical.html). A small button injected into
# each page switches between the two.
SWITCH = """
<a href="{href}" style="position:fixed;top:10px;left:12px;z-index:1000;
   padding:4px 10px;border-radius:6px;font:13px system-ui,sans-serif;text-decoration:none;
   color:#fff;background:#073349;opacity:.75">⇄ {label}</a>
"""


def export(file: str, out: Path) -> None:
    # Run from notebooks/ so css_file and layout_file paths resolve.
    subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html-wasm", file, "-o", str(out),
         "--mode", "run", "--no-sandbox", "--offline", "-f"],
        cwd=ROOT / "notebooks", check=True,
    )


def add_switch(page: Path, href: str, label: str) -> None:
    page.write_text(page.read_text().replace(
        "</body>", SWITCH.format(href=href, label=label) + "</body>", 1))


def has_slides(file: str) -> bool:
    """True if the notebook declares a slides layout_file in marimo.App(...)."""
    src = (ROOT / "notebooks" / file).read_text()
    return re.search(r'layout_file\s*=\s*"[^"]*slides\.json"', src) is not None


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "notebooks").mkdir(parents=True)

    items = []
    for file, title, description in NOTEBOOKS:
        name = Path(file).stem
        page = OUT / "notebooks" / f"{name}.html"
        export(file, page)
        url = f"notebooks/{name}.html"
        views = ""
        if has_slides(file):
            # Vertical copy: same notebook without layout_file, exported from a
            # temporary file next to it (so theme/ and public/ still resolve).
            src = ROOT / "notebooks" / file
            tmp = src.with_name(f"{name}.vertical.py")
            tmp.write_text(re.sub(r'\s*layout_file\s*=\s*"[^"]*",?', "", src.read_text(), count=1))
            vpage = OUT / "notebooks" / f"{name}.vertical.html"
            try:
                export(tmp.name, vpage)
            finally:
                tmp.unlink()
            title = re.search(r"<title>[^<]*", page.read_text())[0]  # same tab title as the slides
            vpage.write_text(re.sub(r"<title>[^<]*", title, vpage.read_text(), count=1))
            add_switch(page, f"{name}.vertical.html", "Notebook view")
            add_switch(vpage, f"{name}.html", "Slides view")
            views = (f'<span class="views"><a href="{url}">Slides</a> · '
                     f'<a href="notebooks/{name}.vertical.html">Notebook</a></span>')
        items.append(
            f'      <li><a href="{url}">{html.escape(title)}</a>{views}'
            f"<p>{html.escape(description)}</p></li>"
        )

    # Code files that the notebooks display (read through notebook_location().parent).
    shutil.copytree(ROOT / "unibotics", OUT / "unibotics",
                    ignore=shutil.ignore_patterns("__pycache__"))

    page = (ROOT / "site" / "index.html").read_text()
    page = page.replace("<!-- NOTEBOOKS -->", "\n".join(items))
    (OUT / "index.html").write_text(page)
    (OUT / ".nojekyll").touch()  # serve files as they are
    print(f"Site built in {OUT}")


if __name__ == "__main__":
    main()
