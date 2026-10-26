"""Figures for the reframed 'Curiosity, Part 2'. Dates come from session transcripts and lab logs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import date

from make_part2_figs import box, BLUE, ORANGE, GREY, GREEN, RED, LIGHT  # noqa: E402  (regenerates part2 figs too)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

# 10. Who did what, over time -----------------------------------------------------------------------
human = [(date(2026, 7, 25), "Is there a measure\nfor inspiration?"),
         (date(2026, 8, 10), "Entropy, surprise and\ncuriosity are conflated"),
         (date(2026, 9, 6), "Build a harness that\nmakes agents curious"),
         (date(2026, 9, 13), "Test decisions, not\nsentence boundaries"),
         (date(2026, 10, 5), "This benchmark needs\nno curiosity"),
         (date(2026, 10, 15), "Mine why-traces from\nreal research"),
         (date(2026, 10, 25), "Is solution.sh a cheat?\nIsn't this a formula?")]
ai = [(date(2026, 7, 14), "ALFWorld: entropy vs\nbranch-early ablation"),
      (date(2026, 7, 28), "Fork experiment\n(creative writing)"),
      (date(2026, 8, 24), "Survey; physics signals\non ALFWorld"),
      (date(2026, 9, 9), "Terminal-Bench trigger,\npreregistered tests"),
      (date(2026, 10, 7), "Cost-of-information\nenvironment"),
      (date(2026, 10, 18), "QuestBench training;\nSAT determinacy probe")]
t0 = date(2026, 7, 5).toordinal()
x = lambda d: d.toordinal() - t0
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.set_xlim(-4, 118); ax.set_ylim(-1.35, 1.45); ax.axis("off")
ax.axhline(0, color="#555", lw=1.2)
for m, lab in [(date(2026, 7, 10), "Jul"), (date(2026, 8, 1), "Aug"), (date(2026, 9, 1), "Sep"), (date(2026, 10, 1), "Oct")]:
    ax.plot([x(m)] * 2, [-0.05, 0.05], color="#555"); ax.text(x(m), -0.12, lab, ha="center", va="top", fontsize=9, color=GREY)
for i, (d, t) in enumerate(human):
    y = 0.62 if i % 2 == 0 else 1.12
    c = RED if d.month == 10 and d.day == 25 else BLUE
    ax.plot([x(d)] * 2, [0, y - 0.17], color=c, lw=1)
    ax.text(x(d), y, t, ha="center", va="center", fontsize=8.6, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT["red"] if c == RED else LIGHT["blue"], ec=c, lw=1))
for i, (d, t) in enumerate(ai):
    y = -0.6 if i % 2 == 0 else -1.08
    ax.plot([x(d)] * 2, [0, y + 0.17], color=GREEN, lw=1)
    ax.text(x(d), y, t, ha="center", va="center", fontsize=8.6, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT["green"], ec=GREEN, lw=1))
ax.text(-4, 1.42, "Human: questions, redirections, objections", fontsize=9.5, color=BLUE, fontweight="bold", va="top")
ax.text(-4, -1.33, "AI agent: surveys, environments, experiments, analysis, drafts", fontsize=9.5, color=GREEN,
        fontweight="bold", va="bottom")
fig.savefig(OUT + "fig10_timeline.png"); plt.close(fig)

# 11. Which errors each side caught ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.axis("off")
box(ax, 0.0, 0.82, 0.47, 0.15, "Is the measurement correct?\n(internal validity)", LIGHT["green"], fs=11, bold=True)
box(ax, 0.53, 0.82, 0.47, 0.15, "Does it measure curiosity?\n(construct validity)", LIGHT["red"], fs=11, bold=True)
left = ["Wrong working-directory diagnosis → one runtime check",
        "Scorer matched \"od\" inside Git output → audit, result unchanged",
        "GLM arm lost its reflections → parity gate failed the run",
        "\"Task ordering transfers\" → retracted when DeepSeek reversed it",
        "Exploratory p = 0.03 → not counted; replication not rescued"]
right = ["Entropy, surprise, inspiration treated as one quantity",
         "A benchmark where nothing was hidden, proposed for curiosity",
         "Ignoring solution.sh may be integrity, not lack of curiosity",
         "Cost environment: we supplied the rule; training distilled a prompt",
         "Sufficiency diagnostic run only after five training attempts*"]
for i, (l, r) in enumerate(zip(left, right)):
    y = 0.66 - i * 0.12
    ax.text(0.01, y, "•  " + l, fontsize=9.6, va="center")
    ax.text(0.54, y, "•  " + r, fontsize=9.6, va="center")
ax.text(0.235, 0.0, "Caught by the agent's own checks", ha="center", fontsize=10.5, color=GREEN, fontweight="bold")
ax.text(0.765, 0.0, "Caught by the human asking plain questions", ha="center", fontsize=10.5, color=RED,
        fontweight="bold")
ax.text(1.0, -0.06, "*caught by the agent, but late", ha="right", fontsize=8.5, color=GREY)
fig.savefig(OUT + "fig11_who_caught_what.png"); plt.close(fig)
print("ok")
