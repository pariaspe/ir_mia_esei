# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/s4_demos.slides.json",
    css_file="theme/slides-theme.css",
)

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 4 · Demos and final advice
    ### Last working session before the final delivery

    * **Today:** I will watch every pair's robot **running live** (~15 min per
      pair) and give you some final advice. The rest of the time is yours to
      polish the robot, the report and the video.

    * **Final delivery: tomorrow, 18/12.**

    /// admonition | No new features today
    Today is for fixing, measuring and documenting. If it isn't working by now,
    leave it out and explain it under *Limitations*.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plan for today

    | Time | Block |
    |---|---|
    | 0:00 - 0:10 | Welcome and order of the demos |
    | 0:10 - 2:50 | **Live demos** (~15 min per pair) · the rest keep working |
    | 2:50 - 3:00 | Questions about the delivery |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Your live demo (~15 min)

    **Before your turn**

    * The exercise **open and loaded** in Unibotics, your final code pasted in.
    * The FSM diagram at hand (on screen or on paper).
    * A **recorded run** ready, in case the simulator misbehaves.

    **During the demo**

    1. **~5 min · Run:** start the robot and let it run. Tell me what it is doing
       while it does it; the console with the state transitions helps.
    2. **~5 min · Questions:** to **both** of you, about any part of the system.
    3. **~5 min · Advice:** what to fix, measure or explain before tomorrow.
    """)
    return


@app.cell
def _():
    # Teacher's controls for the schedule (displayed on the next slide).
    n_pairs = mo.ui.number(start=1, stop=20, value=8, label="Pairs")
    slot = mo.ui.number(start=5, stop=25, value=15, label="Minutes per demo")
    return n_pairs, slot


@app.cell(hide_code=True)
def _(n_pairs, slot):
    _start, _end = 10, 170  # minutes since the session started (0:10 and 2:50)

    def _hm(m):
        return f"{m // 60}:{m % 60:02d}"

    _rows = "\n".join(
        f"| Pair {i + 1} | {_hm(_start + i * slot.value)} - {_hm(_start + (i + 1) * slot.value)} | |"
        for i in range(n_pairs.value)
    )
    _finish = _start + n_pairs.value * slot.value
    _note = (
        mo.callout(mo.md(f"Ends at **{_hm(_finish)}**, before 2:50."), kind="success")
        if _finish <= _end
        else mo.callout(
            mo.md(f"Ends at **{_hm(_finish)}**, after 2:50: shorten the demos."),
            kind="warn",
        )
    )
    _table = "## Demo schedule\n\n| Pair | Time | Exercise |\n|---|---|---|\n" + _rows
    mo.hstack(
        [mo.md(_table), mo.vstack([n_pairs, slot, _note], gap=1)],
        widths=[3, 2], align="start", gap=2,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # While you wait (and after your demo)

    In this order:

    1. **Freeze the code.** Save the version that works as `final.py`. Any
       later change goes to a copy, and only replaces `final.py` if it is
       clearly better.
    2. **Record the video** with the frozen code, if you haven't yet.
    3. **Run the experiments** you are missing: 5+ runs per condition, write
       down the failures too.
    4. **Write the report:** experiments, limitations and what you would change.
    5. After your demo: apply the advice you got, **if it is quick and safe**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Final delivery checklist (18/12)

    | | Item | Check |
    |---|---|---|
    | 💻 | **Code** | The `.py` you ran in the video, it runs as it is in Unibotics, parameters explained in comments |
    | 📄 | **Report** (PDF) | Architecture as implemented (and changes since the proposal), design choices, experiments (runs, success rate, mean ± spread), limitations, who did what |
    | 🎬 | **Video** (2-3 min) | Mission, architecture, the robot running start to end. The link works in a private window |
    | 👥 | **Both names** | On the report, the code and the video |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | Honest beats perfect
    A robot that completes part of the mission, with experiments that show
    *where* and *why* it fails, is a good delivery. Don't hide the failures.
    ///

    **Optional revision: 15/01.** Same deliverables, with the improvements.
    """)
    return


if __name__ == "__main__":
    app.run()
