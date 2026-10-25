"""Figures for 'Curiosity, Part 2'. All numbers are copied from LAB_LOG / paper tables / RESULTS.md files."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__)) + "/"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY, GREEN, RED = "#3b6ea8", "#e08a2c", "#9a9a9a", "#4a9a5b", "#c2453c"
LIGHT = {"blue": "#dce7f3", "orange": "#fbe6cf", "green": "#dcefe0", "red": "#f6dcda", "grey": "#eeeeee"}


def box(ax, x, y, w, h, text, fc, ec="#555", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color="#555", text=None, fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", va="bottom", fontsize=fs, color=color)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.axis("off"); return fig, ax


# 1. Overall flow -----------------------------------------------------------------------------------
fig, ax = canvas(12, 4.6)
steps = [
    ("1. Where to spend\nexploration?", "Fork at high-entropy tokens\n(creative writing, ALFWorld)",
     "Entropy finds consequence,\nnot value", LIGHT["blue"]),
    ("2. Notice what the\nplan missed?", "Reflection prompt when terminal\nsurfaces planted solution.sh",
     "Sharpens an existing\ninclination; can't create one", LIGHT["orange"]),
    ("3. When is a test\nworth its cost?", "Priced tests in a DiscoveryWorld-\nstyle env; checked VOI traces",
     "Works, and transfers to\nreal DiscoveryWorld", LIGHT["green"]),
    ("4. When is a question\nneeded?", "Why-traces, sim episodes, dose\nsweep, prompting (QuestBench)",
     "Fails: models can't tell\nwhen they know enough", LIGHT["red"]),
]
w, gap = 0.2, 0.065
for i, (q, how, res, c) in enumerate(steps):
    x = 0.01 + i * (w + gap)
    box(ax, x, 0.68, w, 0.25, q, c, bold=True, fs=11)
    box(ax, x, 0.37, w, 0.22, how, "white", fs=8.8)
    box(ax, x, 0.06, w, 0.22, res, c, fs=9.5)
    arrow(ax, x + w / 2, 0.68, x + w / 2, 0.595); arrow(ax, x + w / 2, 0.37, x + w / 2, 0.285)
    if i < 3:
        arrow(ax, x + w + 0.003, 0.805, x + w + gap - 0.003, 0.805)
for y, lab in [(0.805, "question"), (0.48, "experiment"), (0.17, "finding")]:
    ax.text(-0.005, y, lab, rotation=90, ha="right", va="center", fontsize=9, color=GREY)
ax.text(0.5, 0.995, "From spending compute on consequential turns to judging whether information is sufficient",
        ha="center", va="top", fontsize=11, color="#333")
fig.savefig(OUT + "fig1_flow.png"); plt.close(fig)

# 2. Fork experiment ---------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
a1.set_xlim(-0.03, 1.03); a1.set_ylim(-0.03, 1.03); a1.axis("off")
box(a1, 0.0, 0.42, 0.3, 0.16, "response prefix\n(held fixed)", LIGHT["grey"], fs=9)
box(a1, 0.38, 0.68, 0.28, 0.14, "default arm:\nmost likely next token", LIGHT["blue"], fs=9)
box(a1, 0.38, 0.18, 0.28, 0.14, "tail arm: forced token\nfrom the low-probability tail", LIGHT["orange"], fs=9)
arrow(a1, 0.30, 0.52, 0.38, 0.75); arrow(a1, 0.30, 0.48, 0.38, 0.25)
for yc, c in [(0.75, BLUE), (0.25, ORANGE)]:
    for k in range(5):
        a1.plot([0.67, 0.80], [yc, yc - 0.12 + k * 0.06], color=c, lw=1, alpha=0.7)
    a1.text(0.81, yc, "x25 continuations\n-> Ocsai originality\n(1-5)", va="center", fontsize=8.5, color=c)
a1.text(0.5, 0.02, "fork advantage = mean(tail) - mean(default), at 8 positions per response",
        ha="center", fontsize=9.5, fontweight="bold")
a1.set_title("How a fork is measured", fontsize=11, loc="left")
bars = [0.33, 0.19, -0.03]
labs = ["entropy vs\nsize of effect", "same, controlling\nfor position", "entropy vs\ndirection of effect"]
a2.bar(labs, bars, color=[BLUE, BLUE, GREY], width=0.6)
for i, v in enumerate(bars):
    a2.text(i, v + (0.012 if v >= 0 else -0.035), f"{v:+.2f}", ha="center", fontsize=10)
a2.axhline(0, color="#333", lw=0.8); a2.set_ylim(-0.1, 0.4); a2.set_ylabel("Spearman rho")
a2.set_title("Uncertainty finds consequence, not value", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(OUT + "fig2_fork.png"); plt.close(fig)

# 3. Terminal-Bench planted hint -------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
a1.set_xlim(-0.03, 1.03); a1.set_ylim(-0.03, 1.03); a1.axis("off")
box(a1, 0.0, 0.66, 0.28, 0.26, "Working directory\n\ntask files ...\nsolution.sh  <- planted,\nnever mentioned", LIGHT["grey"], fs=9)
box(a1, 0.36, 0.66, 0.28, 0.26, "Agent runs  ls / git status\n\nterminal output shows\nsolution.sh\n= DISCOVERY", LIGHT["blue"], fs=9)
box(a1, 0.72, 0.66, 0.28, 0.26, "Trigger compares output\nwith last written plan:\n\"solution.sh\" not in plan\n-> fire", LIGHT["orange"], fs=9)
box(a1, 0.18, 0.08, 0.46, 0.34, "Reflection prompt appended:\n\"Before continuing, note that the terminal surfaced\nsomething your last plan did not account for:\nsolution.sh. Decide whether it changes what you\nshould do next, then continue.\"", LIGHT["orange"], fs=8.5)
box(a1, 0.72, 0.13, 0.28, 0.24, "Agent opens / runs it?\n= INTERACTION\n\nmetric: interaction\ngiven discovery", LIGHT["green"], fs=9)
arrow(a1, 0.28, 0.79, 0.36, 0.79); arrow(a1, 0.64, 0.79, 0.72, 0.79)
arrow(a1, 0.86, 0.66, 0.6, 0.42); arrow(a1, 0.64, 0.25, 0.72, 0.25)
a1.text(0.0, 0.0, "Control arm: the same number of reflection prompts, inserted at random steps.", fontsize=8.5, color="#555")
a1.set_title("Planting a hint in Terminal-Bench", fontsize=11, loc="left")
groups = ["git-workflow-hack\n(exploratory)", "regime test\n(2nd preregistration)"]
rnd, tgt = [0 / 29, 0 / 54], [6 / 27, 8 / 54]
xs = range(2)
a2.bar([x - 0.18 for x in xs], rnd, 0.34, color=GREY, label="random reflection")
a2.bar([x + 0.18 for x in xs], tgt, 0.34, color=ORANGE, label="targeted reflection")
for x, (r, t, rl, tl) in enumerate(zip(rnd, tgt, ["0/29", "0/54"], ["6/27", "8/54"])):
    a2.text(x - 0.18, r + 0.006, rl, ha="center", fontsize=9); a2.text(x + 0.18, t + 0.006, tl, ha="center", fontsize=9)
a2.set_xticks(list(xs)); a2.set_xticklabels(groups, fontsize=9)
a2.set_ylabel("sightings acted on"); a2.set_ylim(0, 0.3); a2.legend(frameon=False, fontsize=9)
a2.text(1, 0.255, "p = 0.003\n(tasks chosen after\nseeing replication)", ha="center", fontsize=8, color="#555")
a2.set_title("Acting on the surfaced file", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(OUT + "fig3_terminal_hint.png"); plt.close(fig)

# 4. Cost-of-information environment -----------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
a1.set_xlim(-0.03, 1.03); a1.set_ylim(-0.03, 1.03); a1.axis("off")
samples, instr = ["berry A", "berry B", "berry C"], ["thermometer", "gas chrom.", "pH meter"]
x0, y0, cw, ch = 0.22, 0.25, 0.17, 0.15
for j, s in enumerate(instr):
    a1.text(x0 + j * cw + cw / 2, y0 + 3 * ch + 0.03, s, ha="center", fontsize=9)
for i, s in enumerate(samples):
    a1.text(x0 - 0.02, y0 + (2 - i) * ch + ch / 2, s, ha="right", va="center", fontsize=9)
    for j in range(3):
        hit = (i == 1 and j == 1)
        a1.add_patch(plt.Rectangle((x0 + j * cw, y0 + (2 - i) * ch), cw, ch, fc=LIGHT["red"] if hit else "white", ec="#777"))
        a1.text(x0 + j * cw + cw / 2, y0 + (2 - i) * ch + ch / 2, "+" if hit else "-", ha="center", va="center",
                fontsize=12, color=RED if hit else GREY)
a1.text(x0 + 1.5 * cw, 0.13, "Hidden: one contaminated sample, one instrument that detects it.\n"
        "Only that cell reads positive.", ha="center", fontsize=8.5, color="#555")
box(a1, 0.76, 0.6, 0.26, 0.2, "TEST sample | instrument\ncosts c points\n(added by us)", LIGHT["orange"], fs=9)
box(a1, 0.76, 0.28, 0.26, 0.2, "GUESS sample\nends episode\n+100 if correct", LIGHT["green"], fs=9)
a1.set_title("Space Sick, with a price on each test", fontsize=11, loc="left")
import numpy as np
c = np.linspace(0, 100, 200)
a2.plot(c, 100 - c, color=ORANGE, lw=2, label="test first: 100 - c")
a2.plot(c, 0 * c + 50, color=GREEN, lw=2, label="guess now: 50")
a2.axvline(50, color=GREY, ls="--", lw=1); a2.text(51, 88, "break-even\nc = 50", fontsize=9, color="#555")
a2.fill_between(c, 100 - c, 50, where=c < 50, color=ORANGE, alpha=0.12)
a2.fill_between(c, 50, 100 - c, where=c > 50, color=GREEN, alpha=0.12)
a2.text(18, 60, "investigate", color=ORANGE, fontsize=10); a2.text(68, 38, "guess", color=GREEN, fontsize=10)
a2.set_xlabel("cost per test, c"); a2.set_ylabel("expected score"); a2.set_ylim(0, 105)
a2.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.75))
a2.set_title("Toy example: two samples, one perfect test", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(OUT + "fig4_cost_env.png"); plt.close(fig)

# 5. Two training pipelines + results ------------------------------------------------------------------
fig = plt.figure(figsize=(12.5, 7.2))
a0 = fig.add_axes([0, 0.52, 1, 0.46]); a0.set_xlim(-0.03, 1.03); a0.set_ylim(-0.03, 1.03); a0.axis("off")
row = [(0.62, "Attempt 1: outcome ranking", ["rollouts", "rank by\nscore", "SFT on top\nrollouts", "learns base rate:\nguess ~65% at any cost"], LIGHT["red"]),
       (0.12, "Attempt 2: checked VOI rationale",
        ["rollout writes\nVOI comparison\n(guess vs investigate)", "check: arithmetic right\nAND action follows", "SFT, template\nremoved from prompt", "learns the\ncost threshold"], LIGHT["green"])]
for y, title, cells, c in row:
    a0.text(0.0, y + 0.31, title, fontsize=11, fontweight="bold")
    for k, t in enumerate(cells):
        x = 0.0 + k * 0.255
        box(a0, x, y, 0.19, 0.26, t, c if k == 3 else "white", fs=9)
        if k < 3:
            arrow(a0, x + 0.205, y + 0.13, x + 0.25, y + 0.13)
a1 = fig.add_axes([0.06, 0.07, 0.42, 0.36]); a2 = fig.add_axes([0.57, 0.07, 0.42, 0.36])
labs = ["base", "always\nguess", "outcome\nranked", "VOI\nround 1", "VOI\nround 2"]
vals = [1.921, 0.118, 0.174, 0.076, 0.046]
cols = [GREY, GREY, RED, GREEN, GREEN]
a1.bar(labs, vals, color=cols, width=0.6); a1.set_yscale("log"); a1.set_ylabel("held-out regret (log, lower = better)")
for i, v in enumerate(vals):
    a1.text(i, v * 1.12, f"{v:.3f}", ha="center", fontsize=9)
a1.set_title("Cost env, held-out configurations", fontsize=11, loc="left")
costs = ["0.03", "0.10", "0.25"]; base = [0.84, 1.42, 3.23]; r1 = [0.03, 0, 0]
xs = np.arange(3)
a2.bar(xs - 0.18, base, 0.34, color=GREY, label="base (success 0.36)")
a2.bar(xs + 0.18, r1, 0.34, color=GREEN, label="VOI round 1 (success 0.69)")
for x, b, r in zip(xs, base, r1):
    a2.text(x - 0.18, b + 0.05, f"{b:.2f}", ha="center", fontsize=9); a2.text(x + 0.18, r + 0.05, f"{r:g}", ha="center", fontsize=9)
a2.set_xticks(xs); a2.set_xticklabels([f"cost {c}" for c in costs]); a2.set_ylabel("regret")
a2.legend(frameon=False, fontsize=9); a2.set_title("Transfer to real DiscoveryWorld (unseen instruments)", fontsize=11, loc="left")
fig.savefig(OUT + "fig5_cost_training.png"); plt.close(fig)

# 6. QuestBench training arms ------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
labs = ["base", "why-traces\n(lr 1e-4)", "sim episodes", "base rate\n(no question\nneeded)"]
vals = [0.144, 0.072, 0.63, 0.63]
a1.bar(labs, vals, color=[GREY, RED, ORANGE, "white"], edgecolor=["none", "none", "none", "#555"], hatch=[None, None, None, "//"], width=0.6)
for i, v in enumerate(vals):
    a1.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
a1.set_ylabel("Planning-Q accuracy (Qwen3-8B)"); a1.set_ylim(0, 0.75)
a1.text(2.5, 0.69, "learned to stop every time", ha="center", fontsize=9, color="#555")
a1.set_title("Imitation copies the form or the base rate", fontsize=11, loc="left")
share = [0, 25, 50, 75, 100]
p_right = [0.106, 0.205, 0.155, 0.217, 0.199]; p_wrong = [0.079, 0.197, 0.180, 0.185, 0.208]; logic = [0.156, 0.124, 0.110, 0.096, 0.074]
a2.plot(share, p_right, "o-", color=GREEN, label="stops when stopping is right")
a2.plot(share, p_wrong, "o-", color=RED, label="stops when a question is needed")
a2.plot(share, logic, "s--", color=BLUE, label="Logic-Q accuracy")
a2.set_xticks(share); a2.set_xticklabels(["base", "25%", "50%", "75%", "100%"])
a2.set_xlabel("share of stop examples in training"); a2.set_ylabel("rate"); a2.set_ylim(0, 0.3)
a2.legend(frameon=False, fontsize=9, loc="upper left")
a2.set_title("Dose sweep: a stopping frequency, not a condition", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(OUT + "fig6_questbench_arms.png"); plt.close(fig)

# 7. Solve vs tell -----------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
labs = ["solve fully\nspecified\n(thinking off)", "solve fully\nspecified\n(thinking on)", "pick the right\nquestion"]
vals = [0.91, 0.98, 0.295]
a1.bar(labs, vals, color=[GREEN, GREEN, RED], width=0.6)
for i, v in enumerate(vals):
    a1.text(i, v + 0.015, f"{v:.3g}", ha="center", fontsize=10)
a1.set_ylim(0, 1.1); a1.set_ylabel("accuracy"); a1.set_title("Gemini 2.5 Flash, QuestBench Logic-Q", fontsize=11, loc="left")
labs = ["Gemini\nthinking off", "Gemini\nthinking on", "Qwen3-8B", "Qwen3-8B +\ndeterminacy SFT"]
over = [0.40, 0.41, 0.81, 0.75]; under = [0.32, 0.27, 0.16, 0.10]
xs = np.arange(4)
a2.bar(xs - 0.18, over, 0.34, color=RED, label="over-claims (Yes/No when undetermined)")
a2.bar(xs + 0.18, under, 0.34, color=GREY, label="under-claims (Not sure when determined)")
for x, o, u in zip(xs, over, under):
    a2.text(x - 0.18, o + 0.015, f"{o:.2f}", ha="center", fontsize=9); a2.text(x + 0.18, u + 0.015, f"{u:.2f}", ha="center", fontsize=9)
a2.set_xticks(xs); a2.set_xticklabels(labs, fontsize=9); a2.set_ylim(0, 1.05); a2.legend(frameon=False, fontsize=9)
a2.set_title("Telling whether the goal is determined (SAT-checked)", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(OUT + "fig7_solve_vs_tell.png"); plt.close(fig)

# 8. Why picking a question needs repeated determinacy judgments ------------------------------------------
fig, ax = canvas(12, 4.2)
box(ax, 0.0, 0.38, 0.2, 0.26, "Rules + known facts\n\nIs the goal already\ndetermined?", LIGHT["blue"], fs=9.5)
box(ax, 0.0, 0.0, 0.2, 0.2, "yes -> no question needed", LIGHT["green"], fs=9)
arrow(ax, 0.1, 0.38, 0.1, 0.2)
for k, L in enumerate("XYZ"):
    f = f"candidate fact {L}"
    y = 0.72 - k * 0.32
    box(ax, 0.3, y, 0.16, 0.2, f, "white", fs=9)
    arrow(ax, 0.2, 0.51, 0.3, y + 0.1, text="no" if k == 0 else None)
    box(ax, 0.53, y + 0.12, 0.2, 0.08, f"if {L} true: determined?", LIGHT["blue"], fs=8.5)
    box(ax, 0.53, y - 0.0, 0.2, 0.08, f"if {L} false: determined?", LIGHT["blue"], fs=8.5)
    arrow(ax, 0.46, y + 0.12, 0.53, y + 0.155); arrow(ax, 0.46, y + 0.08, 0.53, y + 0.045)
box(ax, 0.8, 0.36, 0.2, 0.3, "Ask the fact for which\nBOTH answers settle\nthe goal", LIGHT["green"], fs=9.5)
for k in range(3):
    arrow(ax, 0.73, 0.77 - k * 0.32, 0.8, 0.51, color=GREY)
ax.text(0.5, 0.995, "Every blue box is the same determinacy judgment - the one models get wrong 40-81% of the time",
        ha="center", va="top", fontsize=10.5, color="#333")
fig.savefig(OUT + "fig8_question_tree.png"); plt.close(fig)

# 9. Proposed witness check -------------------------------------------------------------------------
fig, ax = canvas(12, 3.8)
box(ax, 0.0, 0.35, 0.18, 0.3, "Problem:\nrules + facts\n+ goal", LIGHT["grey"], fs=9.5)
box(ax, 0.26, 0.62, 0.3, 0.3, "\"Undetermined\" + two worlds:\nW1 consistent, goal TRUE\nW2 consistent, goal FALSE", LIGHT["orange"], fs=9)
box(ax, 0.26, 0.08, 0.3, 0.3, "\"Yes\" / \"No\" + derivation:\nfacts -> rule -> ... -> goal", LIGHT["blue"], fs=9)
box(ax, 0.64, 0.35, 0.14, 0.3, "SAT solver\nchecks the\nevidence", "white", fs=9.5)
box(ax, 0.86, 0.6, 0.14, 0.3, "verified ->\nkeep for SFT\nor reward +1", LIGHT["green"], fs=9)
box(ax, 0.86, 0.1, 0.14, 0.3, "fails ->\ndrop / reward 0", LIGHT["red"], fs=9)
arrow(ax, 0.18, 0.55, 0.26, 0.75); arrow(ax, 0.18, 0.45, 0.26, 0.23)
arrow(ax, 0.56, 0.77, 0.64, 0.55); arrow(ax, 0.56, 0.23, 0.64, 0.45)
arrow(ax, 0.78, 0.55, 0.86, 0.75); arrow(ax, 0.78, 0.45, 0.86, 0.25)
ax.text(0.5, 0.995, "Proposed next experiment (not yet run): make the judgment explicit and checkable, as in the cost env",
        ha="center", va="top", fontsize=10.5, color="#333")
fig.savefig(OUT + "fig9_witness_proposal.png"); plt.close(fig)
print("ok")
