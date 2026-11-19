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
    layout_file="layouts/s2_architecture.slides.json",
    css_file="theme/slides-theme.css",
)

with app.setup:
    import marimo as mo

    # notebook_location() is the notebook folder locally and its URL in the browser (WASM)
    ASSETS = mo.notebook_location() / "public"
    from collections import deque

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch, Rectangle

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

    FSM_TEMPLATE = read_bytes(
        mo.notebook_location().parent / "unibotics" / "fsm_template.py"
    ).decode()

    # ------------------------------------------------------------------
    # Toy grid world (Section 1): a robot must go from S to the goal.
    # The robot's map ("known") may differ from the real world ("real").
    # Its only sensor tells it whether the 4 neighbouring cells are free.
    # ------------------------------------------------------------------
    MOVES = ((0, 1), (-1, 0), (1, 0), (0, -1))  # right, up, down, left
    STATE_COLORS = {
        "REACT": "#7f7f7f", "PLAN": "#ff7f0e", "EXECUTE": "#1f77b4",
        "FOLLOW": "#1f77b4", "GOAL": "#2ca02c", "FAIL": "#d62728",
    }


@app.function
def neighbours(pos, shape):
    """The 4 neighbouring cells of pos that are inside the grid."""
    H, W = shape
    for dr, dc in MOVES:
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < H and 0 <= c < W:
            yield (r, c)


@app.function
def bfs(grid, start, goal):
    """Shortest path on a grid (True = wall), or None if there is none."""
    prev = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for n in neighbours(cur, grid.shape):
            if not grid[n] and n not in prev:
                prev[n] = cur
                queue.append(n)
    return None


@app.function
def make_world(unknown_obstacle):
    """Map with a cup-shaped wall; optionally a wall that is not on the map."""
    H, W = 11, 18
    known = np.zeros((H, W), bool)
    known[0, :] = known[-1, :] = known[:, 0] = known[:, -1] = True
    known[2:9, 10] = True  # cup, open towards the start
    known[2, 6:11] = True
    known[8, 6:11] = True
    start, goal = (5, 2), (5, 15)
    real = known.copy()
    if unknown_obstacle:
        # Block the planned route a bit after half way.
        path = bfs(known, start, goal)
        i = int(len(path) * 0.6)
        (r, c), (r2, _) = path[i], path[i + 1]
        if r == r2:
            cells = [(rr, c) for rr in range(r - 2, r + 3)]
        else:
            cells = [(r, cc) for cc in range(c - 2, c + 3)]
        for cell in cells:
            if 0 < cell[0] < H - 1 and 0 < cell[1] < W - 1 and cell not in (start, goal):
                real[cell] = True
    return known, real, start, goal


@app.function
def run_reactive(known, real, start, goal, max_ticks=100):
    """Reactive: each tick, step to the free neighbour closest to the goal."""
    def dist(a):
        return abs(a[0] - goal[0]) + abs(a[1] - goal[1])

    pos, log = start, []
    for _ in range(max_ticks):
        log.append(dict(pos=pos, state="REACT", plan=None, found=set()))
        if pos == goal:
            return log, "goal"
        free = [n for n in neighbours(pos, real.shape) if not real[n]]
        best = min(free, key=dist, default=None)
        if best is None or dist(best) >= dist(pos):
            return log, "stuck"
        pos = best
    return log, "timeout"


@app.function
def run_deliberative(known, real, start, goal):
    """Deliberative: plan once on the map, then execute the plan blindly."""
    plan = bfs(known, start, goal)
    log = [dict(pos=start, state="PLAN", plan=plan, found=set())]
    for nxt in plan[1:]:
        if real[nxt]:
            return log, "crash"
        log.append(dict(pos=nxt, state="EXECUTE", plan=plan, found=set()))
    return log, "goal"


@app.function
def run_hybrid(known, real, start, goal, max_ticks=300):
    """Hybrid: an FSM that plans on its map and checks the sensor before every move."""
    world = known.copy()  # world model: the map, updated with what the robot sees
    pos, state, plan = start, "PLAN", None
    found, log = set(), []
    for _ in range(max_ticks):
        log.append(dict(pos=pos, state=state, plan=plan, found=set(found)))
        if state == "PLAN":  # deliberative layer
            plan = bfs(world, pos, goal)
            state = "FOLLOW" if plan else "FAIL"
        elif state == "FOLLOW":  # reactive layer: sense before acting
            for n in neighbours(pos, real.shape):
                if real[n] and not world[n]:
                    world[n] = True
                    found.add(n)
            nxt = plan[plan.index(pos) + 1]
            if world[nxt]:
                state = "PLAN"  # the path is blocked: replan
            else:
                pos = nxt
                if pos == goal:
                    state = "GOAL"
        else:  # GOAL or FAIL
            break
    return log, state.lower()


@app.function
def plot_world(known, real, start, goal, log, k):
    """Map, trail and current plan at tick k, plus a timeline of the states."""
    k = min(k, len(log) - 1)
    entry = log[k]
    fig, (ax, axt) = plt.subplots(
        2, 1, figsize=(7.5, 5.4), gridspec_kw={"height_ratios": [10, 1]}
    )
    H, W = known.shape
    img = np.ones((H, W, 3))
    img[known] = (0.27, 0.32, 0.37)
    ax.imshow(img, origin="upper")
    for r, c in zip(*np.nonzero(real & ~known)):
        if (r, c) in entry["found"]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#d62728"))
        else:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   hatch="///", edgecolor="#d62728", lw=0.8))
    if entry["plan"]:
        rest = entry["plan"][entry["plan"].index(entry["pos"]):]
        ax.plot([p[1] for p in rest], [p[0] for p in rest], "--", color="#ff7f0e", lw=2)
    trail = [e["pos"] for e in log[: k + 1]]
    ax.plot([p[1] for p in trail], [p[0] for p in trail], "-", color="#1f77b4", lw=3)
    ax.text(start[1], start[0], "S", ha="center", va="center", fontsize=13, weight="bold")
    ax.plot(goal[1], goal[0], marker="*", color="#2ca02c", markersize=20)
    ax.plot(entry["pos"][1], entry["pos"][0], "o", color="#1f77b4", markersize=14,
            markeredgecolor="white", markeredgewidth=2)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", lw=0.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.legend(
        handles=[
            Patch(color=(0.27, 0.32, 0.37), label="wall on the map"),
            Patch(facecolor="white", edgecolor="#d62728", hatch="///", label="NOT on the map"),
            plt.Line2D([], [], color="#1f77b4", lw=3, label="trail"),
            plt.Line2D([], [], color="#ff7f0e", lw=2, ls="--", label="current plan"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=8, frameon=False,
    )
    # State timeline
    for i, e in enumerate(log):
        axt.axvspan(i, i + 1, color=STATE_COLORS[e["state"]], alpha=1.0 if i <= k else 0.25, lw=0)
    axt.axvline(k + 0.5, color="black", lw=1.5)
    axt.set_xlim(0, len(log))
    axt.set_yticks([])
    axt.set_xlabel("tick", fontsize=8)
    axt.tick_params(labelsize=7)
    axt.legend(
        handles=[Patch(color=STATE_COLORS[s], label=s)
                 for s in dict.fromkeys(e["state"] for e in log)],
        loc="upper center", bbox_to_anchor=(0.5, 2.1), ncol=6, fontsize=8, frameon=False,
    )
    ax.set_title(f"tick {k} · state: {entry['state']}", fontsize=10)
    fig.tight_layout()
    return fig


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Session 2 · Robot architectures
    ### From reacting to planning

    * **Goal for today:** understand the three classic ways to organise a robot's
    "brain" (reactive, deliberative, hybrid), start designing the architecture
    for your problem, and have the FSM template running on it.

    * **Goal for next week:** an **architecture proposal** and a **skeleton** of
    your solution that reads the sensors and moves the robot.

    /// admonition | Work in pairs
    From today you work in **pairs** on the exercise assigned from your top 3.
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
    | 0:00 - 0:15 | Back to Follow Line | What a purely reactive robot cannot do |
    | 0:15 - 0:45 | **Step 1**: three architectures | Reactive, deliberative and hybrid, compared on the same problem |
    | 0:45 - 1:05 | **Step 2**: finite state machines | The FSM template, and how to read and draw an FSM |
    | 1:05 - 1:20 | **Step 3**: meet your exercise | Your exercise running in Unibotics, with its sensors and actuators listed |
    | 1:20 - 2:15 | **Step 4**: design in pairs | A first draft of your architecture |
    | 2:15 - 2:40 | **Step 5**: skeleton and wrap-up | The FSM template running on your exercise |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Back to Follow Line

    Your Follow Line controller is **reactive**: every iteration it maps the
    *current* image directly to a command.

    ```
    image ──► error ──► W = -Kp · error
    ```

    It is simple, fast and robust... but:

    * it has **no memory**: lap 10 is driven exactly like lap 1;
    * it has **no goal** beyond "stay on the line": it cannot decide to stop,
      take a pit lane or overtake;
    * it **cannot anticipate**: it only reacts to what it sees right now.

    /// admonition | Your answers from the homework
    *What could the car do better if it remembered the circuit after the
    first lap? What would that require?* Let's share a few answers.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 1 · Three architectures

    | | **Reactive** | **Deliberative** | **Hybrid** |
    |---|---|---|---|
    | Cycle | sense → act | sense → **plan** → act | plan (slow) + react (fast) |
    | Uses a model of the world? | No | Yes (map, goals) | Yes |
    | Speed of response | Very fast | Slow | Fast where it matters |
    | Unexpected events | Handles them | Breaks if the model is wrong | Detects them and replans |
    | Long-term goals | Hard | Natural | Natural |
    | Examples | Follow Line, Braitenberg | Classic planners (PDDL, A*) | 3T architectures, most real robots |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A **hybrid** robot is usually organised in **layers**, each one working at
    its own pace:
    """)
    return


@app.cell(hide_code=True)
def _():
    _box = ("border:2px solid #073349;border-radius:10px;padding:10px 16px;"
            "background:#eef4f7;color:#1d2327;text-align:center;line-height:1.35")
    _pill = ("border:1.5px solid #8a9aa5;border-radius:999px;padding:6px 14px;"
             "background:#fff;color:#1d2327;text-align:center;white-space:nowrap")
    _arrow = "color:#5b6770;text-align:center;font-size:0.9em"
    mo.Html(f"""
    <div style="display:grid;grid-template-columns:auto auto 1fr auto auto;gap:8px 14px;
                align-items:center;max-width:900px;margin:0 auto;font-size:1.15rem">
      <div style="{_pill}">World model<br><small>map, memory</small></div>
      <div style="{_arrow}">→</div>
      <div style="{_box}"><b>Deliberative layer</b><br>plans paths, task order · <i>slow</i></div>
      <div></div><div></div>

      <div style="{_arrow};grid-row:span 3">↑<br>updates</div>
      <div></div>
      <div style="{_arrow}">↓ plan</div>
      <div></div><div></div>

      <div></div>
      <div style="{_box}"><b>Sequencer</b> (FSM)<br>decides which behaviour runs now</div>
      <div></div><div></div>

      <div></div>
      <div style="{_arrow}">↓ activates &nbsp;&nbsp; ↑ events</div>
      <div></div><div></div>

      <div style="{_pill}">Sensors</div>
      <div style="{_arrow}">→</div>
      <div style="{_box}"><b>Reactive layer</b><br>behaviours: follow, avoid, align · <i>fast</i></div>
      <div style="{_arrow}">→</div>
      <div style="{_pill}">Actuators</div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Three possible solutions to the same problem

    A robot must go from **S** to the **goal ★** on a grid. Its only sensor tells
    it whether the 4 neighbouring cells are free (like a bumper or a short-range
    laser).

    * **Reactive**: each tick, step to the free neighbour that is closest to the goal.
    * **Deliberative**: plan the shortest path on the map once, then execute it blindly.
    * **Hybrid**: plan on the map, check the sensor before every move, and
      **replan** when the path is blocked. The map is updated with what it sees.

    Then add a wall that is **not on the robot's map**.
    """)
    return


@app.cell
def _():
    # Displayed next to the map below (a cell cannot read the value of a
    # control it creates itself).
    agent = mo.ui.radio(
        options=["Reactive", "Deliberative", "Hybrid"], value="Reactive", label="**Robot**"
    )
    surprise = mo.ui.checkbox(label="Add a wall that is not on the map")
    return agent, surprise


@app.cell
def _(agent, surprise):
    known, real, start, goal = make_world(surprise.value)
    _run = {"Reactive": run_reactive, "Deliberative": run_deliberative, "Hybrid": run_hybrid}
    world_log, world_status = _run[agent.value](known, real, start, goal)
    # The tick lives in a state so the slider and the buttons stay in sync.
    get_tick, set_tick = mo.state(len(world_log) - 1)
    return (
        get_tick,
        goal,
        known,
        real,
        set_tick,
        start,
        world_log,
        world_status,
    )


@app.cell
def _(set_tick, world_log):
    # The buttons must live in a different cell from the slider: marimo does not
    # rerun the cell whose element changed the state, and the slider cell must
    # rerun to move the slider.
    _last = len(world_log) - 1
    prev_tick = mo.ui.button(
        label="◀ Previous tick", on_click=lambda _: set_tick(lambda k: max(k - 1, 0))
    )
    next_tick = mo.ui.button(
        label="Next tick ▶", on_click=lambda _: set_tick(lambda k: min(k + 1, _last))
    )
    return next_tick, prev_tick


@app.cell
def _(get_tick, set_tick, world_log):
    _last = len(world_log) - 1
    tick = mo.ui.slider(
        0, _last, value=min(get_tick(), _last), label="Tick", show_value=True,
        on_change=set_tick,
    )
    return (tick,)


@app.cell
def dummy_comparison(
    agent,
    goal,
    known,
    next_tick,
    prev_tick,
    real,
    start,
    surprise,
    tick,
    world_log,
    world_status,
):
    _moves = sum(a["pos"] != b["pos"] for a, b in zip(world_log, world_log[1:]))
    _plans = sum(e["state"] == "PLAN" for e in world_log)
    if world_status == "goal":
        _extra = f", planned {_plans} times" if agent.value == "Hybrid" else ""
        _banner = mo.callout(mo.md(f"🏁 **Goal reached** in {_moves} moves{_extra}"), kind="success")
    elif world_status == "stuck":
        _banner = mo.callout(
            mo.md(f"🪤 **Stuck** after {_moves} moves: no move brings it closer to the goal"),
            kind="warn",
        )
    else:
        _banner = mo.callout(
            mo.md(f"💥 **Crashed** after {_moves} moves into a wall that was not on its map"),
            kind="danger",
        )
    mo.hstack(
        [
            plot_world(known, real, start, goal, world_log, tick.value),
            mo.vstack([agent, surprise, tick, mo.hstack([prev_tick, next_tick], justify="start"), _banner], gap=1.5),
        ],
        justify="center", align="center", gap=2, widths=[3, 2],
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Experiments (discuss in pairs)

    1. **Reactive.** Why does it get stuck? What would *you* do in its place,
       and what would the robot need to do the same?
    2. **Deliberative.** Without the extra wall it finds the shortest path.
       Now add the wall. Why does it crash? Is the plan wrong, or the map?
    3. **Hybrid.** Add the wall and move the **Tick** slider. When does it
       replan, and why? Which parts of the robot are deliberative, and which
       are reactive?
    4. **Back to your car.** Which of the three is your Follow Line? Which one
       would you need to drive a circuit you have never seen as fast as possible?
    """)
    return


@app.cell
def _():
    mo.md(r"""
    /// admonition | 🚀 Not all reactive robots are this naive
    Smarter reactive behaviours (wall following, the *Bug* algorithms) can
    escape the cup. But they still don't build a map, so they cannot find the
    *shortest* path, or remember anything for the next time.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 2 · Finite state machines

    The hybrid robot above is a **finite state machine (FSM)**. An FSM is the
    simplest way to build the **sequencer** of a hybrid architecture:

    * **States**: what the robot is doing now. Each state runs one behaviour.
    * **Transitions**: when to change state, triggered by **events** computed
      from the sensors or the world model (*path blocked*, *goal reached*,
      *timeout*...).
    * Only **one state is active** at a time.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack([mo.md("## The hybrid robot as an FSM"), mo.hstack(
        [
            mo.mermaid(r"""
    stateDiagram-v2
        [*] --> PLAN
        PLAN --> FOLLOW: path found
        PLAN --> FAIL: no path
        FOLLOW --> PLAN: path blocked / add obstacle to map
        FOLLOW --> GOAL: goal reached
        GOAL --> [*]
            """),
            mo.md(r"""
    | State | Behaviour | Layer |
    |---|---|---|
    | `PLAN` | shortest path on the map | deliberative |
    | `FOLLOW` | sense, then step along the path (stays while the next cell is free) | reactive |
    | `GOAL` / `FAIL` | stop | |

    Read a transition label as
    **event / action**: *"path blocked"* triggers the change, and
    *"add obstacle to map"* is done on the way.
            """),
        ],
        justify="center", align="center", gap=2, widths=[1, 1],
    )])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The FSM in code

    Each state is a **function** that acts and returns the **next state**;
    the main loop just calls the current one (no giant `if / elif` chain):

    ```python
    def plan(sensors):
        memory["path"] = shortest_path(memory["map"], memory["pose"], GOAL)
        return "FOLLOW" if memory["path"] else "FAIL"

    def follow(sensors):
        if path_blocked(sensors):
            memory["map"].add_obstacle(sensors)   # update the world model
            return "PLAN"
        step_along(memory["path"])                # send commands to the robot
        return "GOAL" if at_goal(sensors) else "FOLLOW"

    STATES = {"PLAN": plan, "FOLLOW": follow, "GOAL": stop, "FAIL": stop}
    state = "PLAN"
    while True:
        state = STATES[state](read_sensors())
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The FSM template

    This is the starting point for your skeleton. It is a complete Unibotics
    program: a robot that drives forward and turns when it detects an
    obstacle. Replace the sensors, states and transitions with your own.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.download(
                data=FSM_TEMPLATE.encode(), filename="fsm_template.py",
                mimetype="text/x-python", label="Download fsm_template.py",
            ),
            mo.accordion({"📄 fsm_template.py": mo.md(f"```python\n{FSM_TEMPLATE}```")}),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | Good habits with FSMs
    * **One state, one behaviour.** If you need "and" to describe a state, split it.
    * **Name the events.** `obstacle_ahead(sensors)` reads better than `laser[90] < 0.4`,
      and you can test it on its own.
    * **Print every transition.** The template already does it: the console
      tells you what the robot *thinks* it is doing.
    * **Use `time_in_state()`** for timeouts, so the robot never gets stuck in a state forever.
    * **Draw the diagram first.** If you can't draw it, you can't code it.
    ///

    🚀 FSMs grow messy with many states. If yours does, look up **hierarchical
    FSMs** or **behaviour trees**, which are widely used in robotics and games.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 3 · Meet your exercise

    1. Open your **assigned exercise** in Unibotics and run the default code.
    2. Read its page in the
       [RoboticsAcademy exercise list](https://jderobot.github.io/RoboticsAcademy/exercises/),
       in particular the **API** section: your design can only use what the
       robot can actually sense.
    3. Fill in this table together. It will be part of your proposal:

    | | Your answers |
    |---|---|
    | **Mission**: what must the robot do? When is it done? | |
    | **Sensors**: which ones, and what does each return? | |
    | **Actuators**: which commands move the robot? | |
    | **Evaluation**: how does the platform score it (time, coverage, ...)? | |
    | **Difficulties**: what makes it hard? | |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 4 · Design in pairs

    Answer these questions, in this order. Your answers *are* the first draft
    of your architecture:

    1. **Phases.** Which phases does the mission go through? Those are
       candidate **states**.
    2. **Transitions.** What event ends each phase? Can you compute it from the
       sensors or the world model?
    3. **Memory.** What must the robot remember between iterations (a map,
       visited places, targets found, a counter...)? That is your
       **world model**.
    4. **Reactive or deliberative?** Which decisions need a plan or the world
       model, and which ones can be taken from the current sensor readings alone?
    5. **Failures.** What can go wrong (lost, stuck, target not found)? Add a
       state or a transition to recover from each one.
    6. **Why not reactive?** In two sentences: why is a purely reactive design
       not enough for this exercise?

    Then **draw** the FSM and the layers. Paper, a whiteboard or
    [mermaid.live](https://mermaid.live) are all fine.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Worked example: Follow Line with a pit stop

    Variant of Session 1 exercise: *after 3 laps the car must take the pit lane (marked with a green line),
    stop for 5 s and rejoin the race.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.mermaid(r"""
    stateDiagram-v2
        [*] --> RACE
        RACE --> TO_PIT: lap 3 and green line seen
        TO_PIT --> PIT_STOP: stopped at the box
        TO_PIT --> RACE: green line lost
        PIT_STOP --> REJOIN: 5 s in state
        REJOIN --> RACE: red line found
    """).center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Worked example: the architecture

    * **Reactive layer**: follow the red line, follow the green line, stop.
    * **Sequencer**: the FSM on the previous slide.
    * **World model**: a lap counter (+1 each time the start line is crossed).
    * **Sensors**: the camera only (red line, green line, start line).
    * **Why not reactive?** "Pit on lap 3" needs **memory** (the lap count) and
      a **sequence** of behaviours: the image alone can't tell lap 1 from lap 3.
    * **Failure**: if the green line is lost, go back to `RACE` and try next lap.
    """)
    return


@app.cell(hide_code=True)
def _():
    proposal_template = r"""# Architecture proposal: <exercise name>

    Pair: <name 1> (architecture and report), <name 2> (implementation)

    ## 1. Mission
    What the robot must do, and when the mission is complete.

    ## 2. Sensors and actuators
    | Sensor / actuator | What it gives / does | Used by (state or layer) |
    |---|---|---|
    | | | |

    ## 3. Architecture
    Diagram of the layers (or behaviours) and how information flows between
    them. Say which parts are deliberative and which are reactive.

    ## 4. States and transitions
    Diagram of the FSM. For every transition: the event that triggers it and
    how you compute it from the sensors or the world model.

    ## 5. World model
    What the robot remembers (map, visited places, targets, counters...), how
    it is updated, and who uses it.

    ## 6. Why not purely reactive?
    Two or three sentences.

    ## 7. Risks and plan
    What could fail, how you will recover, and what you will implement first.
    """
    mo.vstack(
        [
            mo.md(r"""
    ## Your proposal: template

    **1-2 pages**, diagrams included. Use these sections (download the
    Markdown file or copy them into the editor you prefer):
            """),
            mo.download(
                data=proposal_template.encode(), filename="architecture_proposal.md",
                mimetype="text/markdown", label="Download architecture_proposal.md",
            ),
            mo.accordion({"📄 architecture_proposal.md": mo.md(f"````markdown\n{proposal_template}````")}),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Step 5 · Skeleton

    Before the end of the session, get the FSM template running on **your**
    exercise:

    1. Paste `fsm_template.py` into the Unibotics editor and press ▶.
       Check the motion commands: if your robot is not a wheeled one, replace
       `HAL.setV` / `HAL.setW` with the commands from the API.
    2. Fill in `read_sensors()` with **every sensor** your design uses, and
       `print()` them once. Do the values make sense?
    3. Make **one state move the robot**.
    4. Add your states as **empty functions** that return their own name.
       Wire the transitions one by one.

    /// admonition | ✅ Checkpoint
    Before leaving, every pair should have the template running on their
    exercise and printing at least one sensor.
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Homework: proposal and skeleton

    **Due before Session 3 (10/12)** · in pairs

    1. **Architecture proposal** (PDF, 1-2 pages), following the template:
       a diagram of the layers or behaviours, the states and transitions, the
       sensors used by each part, what goes in the world model, and why this
       needs more than a reactive design.
    2. **Skeleton** (a single `.py` file) that runs without errors and:
        * reads **every sensor** your design needs;
        * **moves** the robot;
        * has the **states of your FSM**, even if most of them are still empty.

    The skeleton doesn't have to solve anything yet: it proves that you can
    read the sensors and move the robot, so Session 3 can go to implementation.
    """)
    return


if __name__ == "__main__":
    app.run()
