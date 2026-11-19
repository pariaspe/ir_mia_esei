# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/s3_implementation.slides.json",
    css_file="theme/slides-theme.css",
)

with app.setup:
    import marimo as mo

    # notebook_location() is the notebook folder locally and its URL in the browser (WASM)
    ASSETS = mo.notebook_location() / "public"
    import matplotlib.pyplot as plt
    import numpy as np


@app.function
def run_trial(speed, noise, rng, length=12.0, half=1.0, dt=0.05, period=10, t_max=60.0):
    """Toy robot driving down a corridor (2 m wide, 12 m long) to the exit.

    Every `period` ticks a P controller steers it back to the centre line.
    Heading noise and wheel slip make every run different. Returns the
    outcome ("goal", "crash" or "timeout"), the time and the trajectory.
    """
    x = y = th = w = 0.0
    xs, ys = [x], [y]
    for i in range(int(t_max / dt)):
        if i % period == 0:  # the controller runs slower than the physics
            w = -1.5 * y - 2.0 * np.sin(th)
        th += w * dt + noise * np.sqrt(dt) * rng.normal()
        v = speed * max(0.0, 1 + 0.6 * rng.normal())  # wheel slip
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        xs.append(x)
        ys.append(y)
        if abs(y) > half:
            return "crash", (i + 1) * dt, xs, ys
        if x >= length:
            return "goal", (i + 1) * dt, xs, ys
    return "timeout", t_max, xs, ys


@app.function
def plot_trials(trials):
    """Trajectories (green: reached the exit, red: crashed) and histogram of times."""
    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    ax.axhspan(1.0, 1.4, color="#44525c")
    ax.axhspan(-1.4, -1.0, color="#44525c")
    for outcome, _, xs, ys in trials:
        ok = outcome == "goal"
        ax.plot(xs, ys, color="#2ca02c" if ok else "#d62728", lw=1.2, alpha=0.6)
        if not ok:
            ax.plot(xs[-1], ys[-1], "x", color="#d62728", ms=7, mew=2)
    ax.axvline(12, color="#2ca02c", ls="--", lw=1)
    ax.set_xlim(-0.3, 12.5)
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("x (m)")
    ax.set_yticks([])
    ax.set_title("Corridor (exit at x = 12 m)", fontsize=10)

    times = [t for outcome, t, *_ in trials if outcome == "goal"]
    if times:
        axh.hist(times, bins=min(15, max(3, len(times) // 3)), color="#2ca02c", alpha=0.8)
        axh.axvline(np.mean(times), color="k", lw=1.5, label="mean")
        axh.axvline(min(times), color="#ff7f0e", lw=1.5, ls="--", label="best")
        axh.legend(fontsize=8)
    axh.set_xlabel("time to the exit (s)")
    axh.set_title("Successful runs", fontsize=10)
    fig.tight_layout()
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 3 · From skeleton to robot
    ### Implementation, experiments and a chat with every pair

    * **Goal for today:** turn your skeleton into a robot that completes a
      first version of the mission, plan your experiments, and agree with me
      on what you will deliver.

    * **Final delivery: 18/12**, the day after Session 4. Today is the last
      full working session before it: use it well.

    /// admonition | 1:1 meetings today
    I will meet **every pair for 10-15 minutes** while the rest keep working.
    Check the schedule and what to bring in Step 4.
    ///
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
    | 0:00 - 0:15 | Feedback on the proposals | The common issues, so you can fix yours |
    | 0:15 - 0:35 | **Step 1**: from skeleton to robot | A way to build and debug your robot state by state |
    | 0:35 - 0:55 | **Step 2**: experiments | What to measure, and why one run is not enough |
    | 0:55 - 1:05 | **Step 3**: the final delivery | What to deliver on 18/12, and a must / should / could list |
    | 1:05 - 2:55 | **Step 4**: work in pairs + **1:1 meetings** | Progress on your robot, and feedback on it |
    | 2:55 - 3:00 | Wrap-up | Your plan until Session 4 |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feedback on the proposals

    Common issues worth checking in your own proposal:

    * **States that are actions, not phases.** "Turn left" is usually a
      command inside a state, not a state. A state is a phase of the mission.
    * **Transitions you cannot compute.** "When the room is clean" sounds
      fine, but how does the robot *know*? Every event needs a formula from
      the sensors or the world model.
    * **A world model that nobody uses**, or a state that needs memory the
      world model does not have.
    * **No failure path.** What happens when the robot gets stuck, loses the
      target or the plan fails? Add a timeout (`time_in_state()`) at least.
    * **Too ambitious.** A simple design that works beats a brilliant one that
      doesn't. We will cut scope together in the 1:1 meetings.

    You will get individual feedback on your proposal in the meeting.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 1 · From skeleton to robot

    Build the robot **one state at a time**. Don't write the whole FSM and
    press ▶ hoping for the best.

    1. **Pick the first state** of the mission and make it work on its own:
       set it as the initial state and comment out its transitions.
    2. **Add its exit event** (e.g. `obstacle_ahead(sensors)`). Print the value
       for a while before you trust it: is it ever `True`? Too often?
    3. **Add the next state**, and test the transition between the two.
    4. **Repeat** until the mission runs from start to end.
    5. **Only then**, add the failure paths and tune the parameters.

    /// admonition | Keep a working version
    Each time something works, **save a copy** of the code (`v1_explore.py`,
    `v2_with_return.py`...). The Unibotics editor will not do it for you, and
    you will want to go back.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Debugging toolkit

    Small additions to the template that save hours:

    ```python
    DEBUG_STATE = None        # e.g. "TURN": start here to test one state alone
    state = DEBUG_STATE or "FORWARD"
    ticks = 0
    while True:
        Frequency.tick(LOOP_HZ)
        sensors = read_sensors()
        ticks += 1
        if ticks % LOOP_HZ == 0:              # print once per second, not 20 times
            print(state, f"{time_in_state():.1f}s", obstacle_ahead(sensors))
        ...
    ```

    * **Print events, not raw data.** `obstacle_ahead=True` is readable;
      360 laser readings are not.
    * **Slow down** while debugging, and **change one thing at a time**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## When it doesn't work

    | Symptom | Usually means | Try |
    |---|---|---|
    | It never leaves a state | The exit event is never `True` | Print the event; check the threshold and the units |
    | It flips between two states | Both events are true at the same time | Add a minimum `time_in_state()`, or two thresholds (hysteresis) |
    | It worked once, not again | It depends on the start position or on luck | Run it 5 times before saying "it works" (Step 2) |
    | It hits things it should see | The sensor index or angle is not what you think | Print the reading with the obstacle in front, then at the side |
    | It is slow and laggy | Heavy work every iteration (images, planning) | Plan once and store it in `memory`; lower the image resolution |

    🚀 If a state's function grows beyond ~30 lines, move pieces into helper
    functions (events, controllers) that you can test on their own.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 2 · Experiments: one run is not enough

    Your report needs **experiments**, not a single video of a lucky run.

    Below, a toy robot drives down a corridor to the exit. Its sensors are noisy
    and its wheels slip a little, so **every run is different**: just like in
    Gazebo.

    1. Start with **1 run** and try several speeds. Which speed would you pick?
    2. Now set **30 runs** and repeat. Still the same choice?
    """)
    return


@app.cell
def _():
    # Displayed next to the plot below (a cell cannot read the value of a
    # control it creates itself).
    speed = mo.ui.slider(0.5, 4.0, step=0.25, value=3.0, label="Speed (m/s)", show_value=True)
    noise = mo.ui.slider(0.1, 0.6, step=0.05, value=0.4, label="Sensor noise", show_value=True)
    n_runs = mo.ui.slider(1, 50, value=1, label="Runs", show_value=True)
    rerun = mo.ui.button(value=0, on_click=lambda v: v + 1, label="🎲 Run again")
    return n_runs, noise, rerun, speed


@app.cell
def _(n_runs, noise, rerun, speed):
    _rng = np.random.default_rng(rerun.value)
    trials = [run_trial(speed.value, noise.value, _rng) for _ in range(n_runs.value)]
    return (trials,)


@app.cell
def experiment(n_runs, noise, rerun, speed, trials):
    _times = [t for outcome, t, *_ in trials if outcome == "goal"]
    _n = len(trials)
    _ok = len(_times)
    if _ok == 0:
        _stats = "No run reached the exit."
    elif _ok == 1:
        _stats = f"Time: **{_times[0]:.1f} s**"
    else:
        _stats = (
            f"Time: **{np.mean(_times):.1f} ± {np.std(_times):.1f} s** "
            f"(best {min(_times):.1f} s)"
        )
    _kind = "success" if _ok == _n else ("warn" if _ok >= 0.8 * _n else "danger")
    _banner = mo.callout(
        mo.md(f"**Success: {_ok}/{_n}** ({100 * _ok / _n:.0f}%)<br>{_stats}"), kind=_kind
    )
    mo.vstack(
        [
            plot_trials(trials),
            mo.hstack([mo.vstack([speed, noise]), mo.vstack([n_runs, rerun]), _banner],
                      justify="space-around", align="center", widths=[2, 2, 3]),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What did you learn? (discuss in pairs)

    1. With one run, speed 3-4 looks great. With 30 runs, how often does it
       crash? Would you report the **best** time or the **mean**?
    2. Increase the **noise**. Which speed would you pick now? There is a
       **trade-off** between speed and reliability: your report should show it.
    3. Press **🎲 Run again** with 1 run a few times. How different are two
       "identical" runs?

    /// admonition | Rule of thumb for your report
    For every result: **how many runs**, the **success rate**, and the
    **mean ± spread** of your metric. Show the failures too, and explain them.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Designing your experiments

    1. **Metric.** What does "better" mean in your exercise? Time, coverage %,
       people found, distance travelled, number of collisions...
    2. **Question.** What do you want to find out? For example: *does the
       replanning state help?*, *how fast can we go before it fails?*,
       *does it work from other start positions?*
    3. **Conditions.** Change **one** thing at a time (a parameter, a state on/off,
       the start position) and keep the rest fixed.
    4. **Repetitions.** At least **5 runs** per condition (more if they are short).

    | Condition | Runs | Success | Metric (mean ± spread) | Notes / failures |
    |---|---|---|---|---|
    | Baseline: speed 0.3 | 5 | 5/5 | 92 ± 6 s | |
    | Speed 0.6 | 5 | 3/5 | 51 ± 4 s | 2 collisions at the door |

    A table like this, and two sentences explaining it, is a perfectly good
    experiments section.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 3 · The final delivery

    | When | What |
    |---|---|
    | **Today (10/12)** | 1:1 meeting: proposal feedback, scope agreed |
    | 10/12 - 17/12 | Implementation and experiments |
    | **Session 4 (17/12)** | Last fixes, then **demos**: a short live demo and questions to **both** members |
    | **18/12** | **Final delivery** (in pairs) |
    | 15/01 (optional) | Revised delivery with improvements |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What to deliver on 18/12

    1. **Code**: the `.py` file(s) you run in Unibotics.
    2. **Report** (PDF, ~4-6 pages): architecture, design choices, experiments,
       limitations, and what you would change.
    3. **Video** (2-3 min): the robot running, with a short explanation.

    /// admonition | Don't leave the video for the 18th
    Record the robot **as soon as it completes the mission once**. You can
    always replace it with a better run later.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    report_template = r"""# Final report: <exercise name>

    Pair: <name 1>, <name 2>

    ## 1. Problem
    The mission in a few lines, and how the platform scores it.

    ## 2. Architecture
    The diagram of the layers and the FSM, as implemented. What changed since the
    proposal, and why?

    ## 3. Implementation
    The key design choices: how each event is computed, what goes in the world
    model, the main parameters and how you chose them.

    ## 4. Experiments
    What you measured, under which conditions, and how many runs. A table with the
    success rate and the mean and spread of your metric. Explain the failures.

    ## 5. Limitations and future work
    What does not work, or works only sometimes, and why. What would you change
    with two more weeks?

    ## 6. Contributions
    Who did what.

    Links: video, code.
    """
    mo.vstack(
        [
            mo.md(r"""
    ## The report: template

    Use these sections (download the Markdown file or copy them into the editor
    you prefer). Reuse your proposal: sections 1-2 are mostly there already.
            """),
            mo.download(
                data=report_template.encode(), filename="final_report.md",
                mimetype="text/markdown", label="Download final_report.md",
            ),
            mo.accordion({"📄 final_report.md": mo.md(f"````markdown\n{report_template}````")}),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The video and the demo

    **Video (2-3 min)**

    * 20 s: the mission. 30 s: the architecture (show the diagram).
      The rest: the robot running, from start to end.
    * Show the **console** with the state transitions next to the robot, so the
      viewer sees what it is thinking.
    * Speed up boring parts (and say so). Show one failure if it teaches something.


    **Demo (Session 4)**

    * A short live run, then questions. **Both** of you must be able to explain
      the whole system, not only your half.
    * Have a recorded run ready in case the simulator misbehaves.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Must / should / could

    You have **8 days**. Split your plan into three lists and agree on them
    with me in the 1:1 meeting:

    | | Meaning | Example (Follow Line with a pit stop) |
    |---|---|---|
    | **Must** | The minimum robot that completes a simple version of the mission. Done before Session 4 | Count laps and stop in the pit after lap 3 |
    | **Should** | Makes it reliable and complete. What a good grade needs | Rejoin the race; recover if the green line is lost |
    | **Could** | Extras if time allows | Speed up on straights and slow down in curves |

    Finish every **must** before starting any **should**. A complete simple
    robot is worth more than half of an ambitious one.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 4 · Work in pairs + 1:1 meetings

    While you work, I will call each pair for **10-15 minutes**.

    **Bring to the meeting:**

    1. Your robot **running** in Unibotics, whatever state it is in.
    2. What changed since the proposal, and why.
    3. Your **must / should / could** lists, and who does what.
    4. **One question** or blocker you want help with.

    **What I will ask:** each of you to explain a part of the system *the other
    one* wrote, how you will test it, and what the riskiest part is.

    /// admonition | While you wait for your turn
    Follow Step 1 (one state at a time). If you are stuck for more than 15
    minutes, write the question down for the meeting and move on to another state.
    ///
    """)
    return


@app.cell
def _():
    # Teacher's controls for the schedule (displayed on the next slide).
    n_pairs = mo.ui.number(start=1, stop=20, value=8, label="Pairs")
    slot = mo.ui.number(start=5, stop=20, value=12, label="Minutes per meeting")
    return n_pairs, slot


@app.cell(hide_code=True)
def _(n_pairs, slot):
    _start, _end = 65, 175  # minutes since the session started (1:05 and 2:55)

    def _hm(m):
        return f"{m // 60}:{m % 60:02d}"

    _rows = "\n".join(
        f"| Pair {i + 1} | {_hm(_start + i * slot.value)} - {_hm(_start + (i + 1) * slot.value)} | |"
        for i in range(n_pairs.value)
    )
    _finish = _start + n_pairs.value * slot.value
    _note = (
        mo.callout(mo.md(f"Ends at **{_hm(_finish)}**, before the wrap-up at 2:55."), kind="success")
        if _finish <= _end
        else mo.callout(
            mo.md(f"Ends at **{_hm(_finish)}**, after 2:55: shorten the meetings "
                  f"or start some during Steps 2-3."),
            kind="warn",
        )
    )
    _table = "## Meeting schedule\n\n| Pair | Time | Exercise |\n|---|---|---|\n" + _rows
    mo.hstack(
        [
            mo.md(_table),
            mo.vstack([n_pairs, slot, _note], gap=1),
        ],
        widths=[3, 2], align="start", gap=2,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Before Session 4 (17/12)

    * ✅ All your **must** items working, from start to end of the mission.
    * ✅ At least **one experiment** run (5+ runs of one condition), with its table.
    * ✅ A first **video** recorded, even if it is not the final one.
    * ✅ Report sections 1-3 drafted.

    Session 4 is for the last fixes, more experiments and the **demos**.
    The final delivery is the next day, so there won't be time to start
    anything new.
    """)
    return


if __name__ == "__main__":
    app.run()
