"""
compare_agents.py — Multi-agent comparison dashboard v2.

Run:
    python compare_agents.py                   # all mazes
    python compare_agents.py --maze 1          # single maze
    python compare_agents.py --window 100      # change smoothing window

Outputs → Results/comparison/
    comparison_maze<N>.png       — 6-panel per-maze dashboard
    cross_maze_summary.png       — cross-maze analysis (solve rate, efficiency)
    ppo_training_dynamics.png    — PPO update-level loss/entropy curves

═══════════════════════════════════════════════════════════════════════════════
WHY SOME AGENTS ARE COMPARED DIFFERENTLY
═══════════════════════════════════════════════════════════════════════════════

PPO (Proximal Policy Optimisation) — rollout-averaged data
    PPO is an on-policy algorithm that collects fixed-size rollout batches
    before performing gradient updates.  The episode CSV logs the *mean*
    reward and steps of each rollout batch, not individual environment
    episodes.  Because PPO converged within the first few updates on all
    three mazes, every logged "episode" already reflects the final policy —
    there is no visible learning curve.

    Indicators in the data:
      • Reward std ≈ 0 across all rows (e.g. maze1: all rows = 25.10).
      • solved = 1 for every single row across all three mazes.
      • Very few rows: 340 (maze1), 60 (maze2), 20 (maze3), versus 1000–5000
        for episode-based agents.

    Treatment applied:
      • PPO is EXCLUDED from learning-curve and steps-over-time line plots.
        Its final-phase mean is shown as a horizontal dashed reference line.
      • PPO IS included in final-performance bar charts (reward, solve %).
      • The updates CSV (training_PPO_maze<N>_updates.csv) is plotted
        separately to show PPO-internal dynamics (loss, entropy, KL).

SAC (Soft Actor-Critic) — insufficient data on maze 2 & 3
    SAC ran for only 160 episodes on maze 2 and 159 episodes on maze 3,
    compared with 5000 for SARSA and 1500 for DQN3.  No episodes were solved
    on either maze, so SAC cannot be fairly compared there.

    Treatment:
      • SAC bars on maze2/3 are annotated "< 165 eps" with hatching.
      • SAC is excluded from convergence-speed analysis on maze2/3.

Cross-maze reward comparison — intentionally not done
    Raw reward magnitudes are not comparable across mazes.  A 35×35 maze
    accumulates far more step penalties over longer timeout episodes than a
    16×16 maze, even if policy quality is identical.  For example, DQN3 peaks
    at +79 on maze1 but only +24 on maze3.  PPO inverts this, scoring higher
    on larger mazes — suggesting reward structures differ per maze.

    Treatment:
      • Cross-maze plots use *solve rate (%)* and *avg steps-when-solved*.
      • Raw reward plots are valid within a single maze dashboard.

Training budget disparity
    Agents were not given equal episode budgets:
      SARSA 5000 | DQN3 1500 | SAC 160–4422 | PPO 20–340.
    Learning-speed comparisons based on episode count are biased toward
    agents with larger budgets.  A training-budget panel is included to
    make this visible.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────────────────────

AGENT_STYLES: dict[str, dict] = {
    "DQN3":  {"color": "#E74C3C", "ls": "-",  "label": "DQN3 (Double DQN)"},
    "SARSA": {"color": "#2ECC71", "ls": "--", "label": "SARSA"},
    "SAC":   {"color": "#F39C12", "ls": "-",  "label": "SAC"},
    "PPO":   {"color": "#9B59B6", "ls": ":",  "label": "PPO"},
}

MAZE_LABELS = {
    1: "Maze 1 (16×16)",
    2: "Maze 2 (25×25)",
    3: "Maze 3 (35×35)",
}

AGENTS = ["DQN3", "SARSA", "SAC", "PPO"]

# Fewer than this many episodes = "early stop / insufficient data"
EARLY_STOP_THRESHOLD = 200


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class AgentData:
    """Normalised per-episode data for one agent on one maze."""
    agent:    str
    maze_id:  int
    x:        np.ndarray   # episode indices
    rewards:  np.ndarray   # total reward per episode
    steps:    np.ndarray   # steps per episode
    solved:   np.ndarray   # bool per episode

    # Flags set during loading
    is_ppo_rollout: bool = False  # True → PPO rollout means, no learning curve
    early_stop:     bool = False  # True → very few episodes, results unreliable

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def _tail(self) -> int:
        """Last 10% of training."""
        return max(1, self.n // 10)

    def final_solve_pct(self) -> float:
        return float(self.solved[-self._tail:].mean() * 100)

    def final_reward(self) -> float:
        return float(self.rewards[-self._tail:].mean())

    def final_steps_when_solved(self) -> float:
        """Avg steps in solved episodes (last 10%). NaN if none solved."""
        mask  = self.solved[-self._tail:]
        steps = self.steps[-self._tail:]
        return float(steps[mask].mean()) if mask.any() else float("nan")

    def total_env_steps(self) -> int:
        return int(self.steps.sum())

    def converged_at(self, threshold: float = 80.0, window: int = 10) -> Optional[int]:
        """Episode index at which rolling solve rate first hit threshold. None if never."""
        rolling = (pd.Series(self.solved.astype(float))
                   .rolling(window, min_periods=1).mean() * 100)
        idx = next((i for i, v in enumerate(rolling) if v >= threshold), None)
        return int(self.x[idx]) if idx is not None else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().values

def _rolling_pct(bool_array: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(bool_array.astype(float)).rolling(window, min_periods=1).mean().values * 100

def _style(agent: str) -> dict:
    return AGENT_STYLES.get(agent, {"color": "#888", "ls": "-", "label": agent})

def _legend_patch(agent: str) -> mpatches.Patch:
    st = _style(agent)
    return mpatches.Patch(facecolor=st["color"], label=st["label"], alpha=0.85)


# ── CSV loading ───────────────────────────────────────────────────────────────

def _load_csv(path: str, agent: str, maze_id: int) -> Optional[AgentData]:
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None

        for col in ("episode", "total_reward", "steps", "solved"):
            if col not in df.columns:
                print(f"    Warning: {path} missing column '{col}', skipping.")
                return None

        rewards = df["total_reward"].values.astype(float)
        steps   = df["steps"].values.astype(float)
        solved  = df["solved"].values.astype(bool)
        x       = df["episode"].values

        # PPO rollout detection: reward std near 0 and all solved.
        # PPO logs batch means, not individual episodes, so variance collapses.
        is_ppo_rollout = (
            agent == "PPO"
            and rewards.std() < 1.0
            and solved.mean() > 0.95
        )

        # Early-stop detection: non-PPO agents with very few episodes.
        early_stop = (not is_ppo_rollout) and (len(x) < EARLY_STOP_THRESHOLD)

        return AgentData(
            agent=agent, maze_id=maze_id,
            x=x, rewards=rewards, steps=steps, solved=solved,
            is_ppo_rollout=is_ppo_rollout,
            early_stop=early_stop,
        )
    except Exception as exc:
        print(f"    Warning: could not load {path}: {exc}")
        return None


def _load_ppo_updates(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_results(results_root: str, maze_filter: Optional[int] = None) -> list[AgentData]:
    records: list[AgentData] = []
    for agent in AGENTS:
        agent_dir = os.path.join(results_root, agent)
        if not os.path.isdir(agent_dir):
            continue
        for maze_id in [1, 2, 3]:
            if maze_filter is not None and maze_id != maze_filter:
                continue
            pattern = os.path.join(agent_dir, f"training_{agent}_maze{maze_id}.csv")
            matches = glob.glob(pattern)
            if not matches:
                continue
            data = _load_csv(matches[0], agent, maze_id)
            if data is None:
                continue
            flag = " [PPO rollout means — no learning curve]" if data.is_ppo_rollout else ""
            flag += " [early stop]"                            if data.early_stop     else ""
            print(f"  Loaded {agent:6s} maze{maze_id}: {data.n:>5} episodes{flag}")
            records.append(data)
    return records


def discover_ppo_updates(results_root: str) -> dict[int, pd.DataFrame]:
    ppo_dir = os.path.join(results_root, "PPO")
    result: dict[int, pd.DataFrame] = {}
    if not os.path.isdir(ppo_dir):
        return result
    for maze_id in [1, 2, 3]:
        pattern = os.path.join(ppo_dir, f"training_PPO_maze{maze_id}_updates.csv")
        for path in glob.glob(pattern):
            df = _load_ppo_updates(path)
            if df is not None:
                result[maze_id] = df
    return result


# ── KPI table ─────────────────────────────────────────────────────────────────

def _build_kpi(records: list[AgentData]) -> pd.DataFrame:
    rows = []
    for d in records:
        if d.is_ppo_rollout:
            conv_str = "pre-logged *"
            note = "rollout means *"
        elif d.early_stop:
            conv_str = "N/A"
            note = f"only {d.n} eps †"
        else:
            ep = d.converged_at()
            conv_str = str(ep) if ep is not None else "—"
            note = ""

        rows.append({
            "Agent":            d.agent,
            "Maze":             d.maze_id,
            "Episodes":         d.n,
            "Total Steps":      f"{d.total_env_steps():,}",
            "Final Solve %":    f"{d.final_solve_pct():.0f}",
            "Final Reward":     f"{d.final_reward():.1f}",
            "Steps/Win (final)": (f"{d.final_steps_when_solved():.0f}"
                                  if not np.isnan(d.final_steps_when_solved()) else "N/A"),
            "Conv. @ ep":       conv_str,
            "Notes":            note,
        })
    return pd.DataFrame(rows)


def print_kpi(records: list[AgentData]) -> None:
    df = _build_kpi(records)
    if df.empty:
        return
    sep = "=" * 115
    print(f"\n{sep}\n  KPI Summary\n{sep}")
    print(df.to_string(index=False))
    print(sep)
    print("  * PPO episodes are rollout batch means logged after convergence — no learning trajectory.")
    print("  † Fewer than 200 episodes trained; results not representative.\n")


# ── Per-maze 6-panel dashboard ────────────────────────────────────────────────

def plot_maze_comparison(
    maze_id: int,
    records:  list[AgentData],
    out_dir:  str,
    window:   int,
) -> None:
    maze_recs    = [r for r in records if r.maze_id == maze_id]
    if not maze_recs:
        return

    # Separate PPO (rollout means) from agents with real learning curves
    episode_recs = [r for r in maze_recs if not r.is_ppo_rollout]
    ppo_recs     = [r for r in maze_recs if r.is_ppo_rollout]

    maze_lbl = MAZE_LABELS.get(maze_id, f"Maze {maze_id}")

    # Maze-level alerts
    no_ep_solver  = all(r.final_solve_pct() < 5 for r in episode_recs) if episode_recs else True
    has_early_sac = any(r.agent == "SAC" and r.early_stop for r in maze_recs)

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        f"Agent Comparison — {maze_lbl}",
        fontsize=16, fontweight="bold", y=0.995,
    )

    alerts = []
    if no_ep_solver and ppo_recs:
        alerts.append("Only PPO solved this maze — episode-based agents did not converge")
    if has_early_sac:
        alerts.append("SAC training stopped early (< 165 episodes); result not representative")
    if alerts:
        fig.text(0.5, 0.968,
                 "  |  ".join(f"⚠  {a}" for a in alerts),
                 ha="center", fontsize=9.5, color="#C0392B",
                 fontweight="bold", style="italic")

    gs = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.36)
    ax_lr  = fig.add_subplot(gs[0, 0])
    ax_sr  = fig.add_subplot(gs[0, 1])
    ax_st  = fig.add_subplot(gs[0, 2])
    ax_br  = fig.add_subplot(gs[1, 0])
    ax_bs  = fig.add_subplot(gs[1, 1])
    ax_tbl = fig.add_subplot(gs[1, 2])

    # ── [0,0] Learning curve (reward) ────────────────────────────────────────
    # Episode-based agents only.
    # PPO is excluded here (no learning trajectory) and shown as a reference line.
    for d in episode_recs:
        st  = _style(d.agent)
        sm  = _smooth(d.rewards, window)
        lbl = st["label"] + (f"  [{d.n} eps only]" if d.early_stop else "")
        ax_lr.plot(d.x, sm, color=st["color"], ls=st["ls"], lw=2, label=lbl)
        lo = pd.Series(d.rewards).rolling(window, min_periods=1).min().values
        hi = pd.Series(d.rewards).rolling(window, min_periods=1).max().values
        ax_lr.fill_between(d.x, lo, hi, alpha=0.07, color=st["color"])
        if d.early_stop:
            ax_lr.axvspan(d.x[0], d.x[-1], alpha=0.05, color=st["color"])

    for d in ppo_recs:
        st     = _style(d.agent)
        mean_r = float(d.rewards.mean())
        ax_lr.axhline(mean_r, color=st["color"], ls="--", lw=2.0, alpha=0.85,
                      label=f"{st['label']} — final mean = {mean_r:.1f} *")

    ax_lr.set_title(f"Learning Curve  (smoothed reward, w={window})", fontweight="bold")
    ax_lr.set_xlabel("Episode")
    ax_lr.set_ylabel("Total Reward per Episode")
    ax_lr.legend(fontsize=7.5, loc="lower right")
    ax_lr.grid(True, alpha=0.22)
    ax_lr.axhline(0, color="black", lw=0.5, alpha=0.35)
    if ppo_recs:
        ax_lr.text(0.01, 0.01,
                   "* PPO line = rollout-mean reward; no learning curve available\n"
                   "  (PPO data covers post-convergence rollouts only)",
                   transform=ax_lr.transAxes, fontsize=6, color="#777", style="italic")

    # ── [0,1] Rolling solve rate ──────────────────────────────────────────────
    # All agents.  PPO → horizontal line at 100% (all its episodes are solved).
    for d in maze_recs:
        st  = _style(d.agent)
        lbl = st["label"]
        if d.is_ppo_rollout:
            # 100% line — PPO logged only post-convergence rollouts
            ax_sr.axhline(100, color=st["color"], ls="--", lw=2.0, alpha=0.85,
                          label=f"{lbl} — 100% (rollout data) *")
        else:
            if d.early_stop:
                lbl += f"  [{d.n} eps only]"
            ax_sr.plot(d.x, _rolling_pct(d.solved, window),
                       color=st["color"], ls=st["ls"], lw=2, label=lbl)

    ax_sr.axhline(80, color="#7F8C8D", ls=":", lw=1.1, alpha=0.65,
                  label="80% threshold")
    ax_sr.set_title(f"Rolling Solve Rate  (w={window})", fontweight="bold")
    ax_sr.set_xlabel("Episode")
    ax_sr.set_ylabel("Solve Rate (%)")
    ax_sr.set_ylim(-2, 110)
    ax_sr.legend(fontsize=7.5, loc="lower right")
    ax_sr.grid(True, alpha=0.22)
    if ppo_recs:
        ax_sr.text(0.01, 0.01,
                   "* PPO at 100% because only post-convergence rollouts were logged",
                   transform=ax_sr.transAxes, fontsize=6, color="#777", style="italic")

    # ── [0,2] Steps per episode ───────────────────────────────────────────────
    # Episode-based agents only.  PPO → horizontal reference line.
    # Decreasing steps = agent learning shorter paths = better policy.
    for d in episode_recs:
        st = _style(d.agent)
        ax_st.plot(d.x, _smooth(d.steps, window),
                   color=st["color"], ls=st["ls"], lw=2, label=st["label"])

    for d in ppo_recs:
        st     = _style(d.agent)
        mean_s = float(d.steps.mean())
        ax_st.axhline(mean_s, color=st["color"], ls="--", lw=2.0, alpha=0.85,
                      label=f"{st['label']} — {mean_s:.0f} steps/ep *")

    ax_st.set_title(f"Steps per Episode  (smoothed, w={window})", fontweight="bold")
    ax_st.set_xlabel("Episode")
    ax_st.set_ylabel("Steps  (lower = more efficient)")
    ax_st.legend(fontsize=7.5)
    ax_st.grid(True, alpha=0.22)
    if ppo_recs:
        ax_st.text(0.01, 0.01,
                   "* PPO line = final-policy step count (rollout mean)",
                   transform=ax_st.transAxes, fontsize=6, color="#777", style="italic")

    # ── [1,0] Final avg reward bar chart ─────────────────────────────────────
    _bar_panel(ax_br, maze_recs,
               values=[d.final_reward() for d in maze_recs],
               ylabel="Avg Reward",
               title=f"Final Avg Reward  (last 10% of training)",
               fmt=".1f",
               has_ppo=bool(ppo_recs))

    # ── [1,1] Final solve rate bar chart ─────────────────────────────────────
    _bar_panel(ax_bs, maze_recs,
               values=[d.final_solve_pct() for d in maze_recs],
               ylabel="Solve Rate (%)",
               title=f"Final Solve Rate  (last 10% of training)",
               fmt=".0f", suffix="%",
               ylim=(0, 115),
               has_ppo=bool(ppo_recs))

    # ── [1,2] KPI table ───────────────────────────────────────────────────────
    ax_tbl.axis("off")
    kdf = _build_kpi(maze_recs).drop(columns=["Maze"])

    if not kdf.empty:
        tbl = ax_tbl.table(
            cellText=kdf.values.tolist(),
            colLabels=list(kdf.columns),
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.8)
        tbl.scale(1.02, 1.65)

        n_cols = len(kdf.columns)
        for c in range(n_cols):
            tbl[0, c].set_facecolor("#2C3E50")
            tbl[0, c].set_text_props(color="white", fontweight="bold")

        for row_i, row_vals in enumerate(kdf.values.tolist()):
            agent_name = row_vals[0]
            c = _style(agent_name)["color"]
            for col_i in range(n_cols):
                tbl[row_i + 1, col_i].set_facecolor(c + "20")

    ax_tbl.set_title("KPI Summary", fontweight="bold", pad=12)
    ax_tbl.text(
        0.5, -0.04,
        "* PPO data = rollout-batch means (post-convergence only) — not comparable as a\n"
        "  learning trajectory.  † Early stop: < 200 episodes; results unreliable.",
        transform=ax_tbl.transAxes, ha="center",
        fontsize=6.2, color="#666", style="italic",
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"comparison_maze{maze_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def _bar_panel(
    ax, records: list[AgentData], values: list[float],
    ylabel: str, title: str, fmt: str = ".1f", suffix: str = "",
    ylim: Optional[tuple] = None, has_ppo: bool = False,
) -> None:
    """Shared helper for the two bar-chart panels."""
    agents  = [d.agent for d in records]
    colors  = [_style(a)["color"] for a in agents]
    x_pos   = np.arange(len(agents))

    bars = ax.bar(x_pos, values, color=colors, alpha=0.85, width=0.6,
                  edgecolor="white", linewidth=0.5)

    for bar, val, d in zip(bars, values, records):
        # Hatch early-stop bars
        if d.early_stop:
            bar.set_hatch("//")
            bar.set_alpha(0.65)
        # Value label
        sign = 1 if val >= 0 else -1
        label_str = f"{val:{fmt}}{suffix}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + sign * max(abs(val) * 0.04, 0.5),
            label_str, ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8, fontweight="bold",
        )
        # Mark PPO bars with asterisk
        if d.is_ppo_rollout and abs(val) > 3:
            mid_y = val / 2
            ax.text(bar.get_x() + bar.get_width() / 2, mid_y,
                    "*", ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22)
    ax.axhline(0, color="black", lw=0.6)
    if ylim:
        ax.set_ylim(*ylim)

    legend_patches = []
    if has_ppo:
        legend_patches.append(
            mpatches.Patch(facecolor="white", edgecolor="#555",
                           label="* = PPO rollout mean"))
    if any(d.early_stop for d in records):
        legend_patches.append(
            mpatches.Patch(facecolor="white", edgecolor="#555",
                           hatch="//", label="/// = early stop"))
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=7, loc="best")


# ── Cross-maze summary ────────────────────────────────────────────────────────

def plot_cross_maze_summary(records: list[AgentData], out_dir: str) -> None:
    """
    Cross-maze comparison.

    Raw reward is NOT used across mazes — see module docstring.
    Metrics used:
      • Solve rate (%): primary measure of policy success.
      • Avg steps when solved: measures efficiency (path length).
      • Solve rate heatmap: quick visual overview.
      • Training budget (log scale): exposes the unequal episode budgets.
    """
    if not records:
        return

    mazes  = sorted({r.maze_id for r in records})
    agents = list(dict.fromkeys(r.agent for r in records))

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Cross-Maze Performance Summary", fontsize=15, fontweight="bold")
    gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.36)

    ax_sr  = fig.add_subplot(gs[0, 0])   # solve rate grouped bar
    ax_eff = fig.add_subplot(gs[0, 1])   # steps when solved
    ax_hm  = fig.add_subplot(gs[1, 0])   # heatmap
    ax_bgt = fig.add_subplot(gs[1, 1])   # training budget

    maze_xlbls = [MAZE_LABELS[m] for m in mazes]
    n_agents   = len(agents)
    bar_w      = 0.75 / max(1, n_agents)
    x          = np.arange(len(mazes))

    # ── Solve rate grouped bar ─────────────────────────────────────────────────
    # PPO shows 100% on all mazes because it logs only post-convergence rollouts.
    # This is a valid final-performance result but not comparable as a learning outcome.
    legend_handles_sr = []
    for i, agent in enumerate(agents):
        st   = _style(agent)
        vals, hatches = [], []
        for maze_id in mazes:
            d = _find(records, agent, maze_id)
            vals.append(d.final_solve_pct() if d else 0.0)
            hatches.append("//" if (d and d.early_stop) else "")

        offset = (i - n_agents / 2 + 0.5) * bar_w
        for j, (val, hatch) in enumerate(zip(vals, hatches)):
            ax_sr.bar(x[j] + offset, val, bar_w * 0.9,
                      color=st["color"], alpha=0.85,
                      edgecolor="white", linewidth=0.3, hatch=hatch)
            if val > 3:
                ax_sr.text(x[j] + offset, val + 0.8, f"{val:.0f}%",
                           ha="center", va="bottom", fontsize=6.5)
        legend_handles_sr.append(mpatches.Patch(facecolor=st["color"],
                                                 label=st["label"], alpha=0.85))

    ax_sr.set_xticks(x)
    ax_sr.set_xticklabels(maze_xlbls, fontsize=9)
    ax_sr.set_ylabel("Final Solve Rate (%)")
    ax_sr.set_ylim(0, 120)
    ax_sr.set_title("Final Solve Rate by Agent × Maze\n(last 10% of each training run)",
                    fontweight="bold")
    ax_sr.legend(handles=legend_handles_sr, fontsize=8, loc="upper left")
    ax_sr.grid(True, axis="y", alpha=0.22)
    ax_sr.text(0.01, -0.13,
               "PPO shows 100% because only post-convergence rollout data was logged.\n"
               "/// = agent trained < 200 episodes — not representative.",
               transform=ax_sr.transAxes, fontsize=7, color="#777", style="italic")

    # ── Efficiency: avg steps when solved ─────────────────────────────────────
    # Absent bar = agent never solved (0 wins in final 10%). Lower = better.
    legend_handles_eff = []
    for i, agent in enumerate(agents):
        st   = _style(agent)
        offset = (i - n_agents / 2 + 0.5) * bar_w
        added_to_legend = False
        for j, maze_id in enumerate(mazes):
            d = _find(records, agent, maze_id)
            if d is None or d.final_solve_pct() < 5:
                continue
            val = d.final_steps_when_solved()
            if np.isnan(val):
                continue
            ax_eff.bar(x[j] + offset, val, bar_w * 0.9,
                       color=st["color"], alpha=0.85,
                       edgecolor="white", linewidth=0.3,
                       hatch="//" if d.early_stop else "")
            ax_eff.text(x[j] + offset, val + 1, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=6.5)
            if not added_to_legend:
                legend_handles_eff.append(
                    mpatches.Patch(facecolor=st["color"], label=st["label"], alpha=0.85))
                added_to_legend = True

    ax_eff.set_xticks(x)
    ax_eff.set_xticklabels(maze_xlbls, fontsize=9)
    ax_eff.set_ylabel("Avg Steps per Solved Episode  (lower = better)")
    ax_eff.set_title("Policy Efficiency — Steps When Solved\n(last 10%; missing bar = never solved)",
                     fontweight="bold")
    ax_eff.legend(handles=legend_handles_eff, fontsize=8, loc="upper left")
    ax_eff.grid(True, axis="y", alpha=0.22)
    ax_eff.text(0.01, -0.10,
                "Missing bar: agent did not solve the maze in its final training phase.",
                transform=ax_eff.transAxes, fontsize=7, color="#777", style="italic")

    # ── Solve rate heatmap ─────────────────────────────────────────────────────
    hm = np.full((len(agents), len(mazes)), np.nan)
    for i, agent in enumerate(agents):
        for j, maze_id in enumerate(mazes):
            d = _find(records, agent, maze_id)
            if d is not None:
                hm[i, j] = d.final_solve_pct()

    im = ax_hm.imshow(hm, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax_hm, label="Final Solve Rate (%)")
    ax_hm.set_xticks(range(len(mazes)))
    ax_hm.set_yticks(range(len(agents)))
    ax_hm.set_xticklabels(maze_xlbls, fontsize=8)
    ax_hm.set_yticklabels([_style(a)["label"] for a in agents], fontsize=8)
    ax_hm.set_title("Solve Rate Heatmap  (final 10%)\nGreen = solved · Red = failed · Grey = no data",
                    fontweight="bold")
    for i in range(len(agents)):
        for j in range(len(mazes)):
            val = hm[i, j]
            if np.isnan(val):
                ax_hm.text(j, i, "N/A", ha="center", va="center",
                           fontsize=9, color="#aaa")
            else:
                text_col = "black" if 20 < val < 80 else "white"
                ax_hm.text(j, i, f"{val:.0f}%", ha="center", va="center",
                           fontsize=11, fontweight="bold", color=text_col)

    # ── Training budget (episodes, log scale) ──────────────────────────────────
    # Agents trained for very different numbers of episodes.  Any comparison of
    # *learning speed* is biased unless you normalise by episodes or timesteps.
    legend_handles_bgt = []
    for i, agent in enumerate(agents):
        st   = _style(agent)
        offset = (i - n_agents / 2 + 0.5) * bar_w
        added = False
        for j, maze_id in enumerate(mazes):
            d = _find(records, agent, maze_id)
            if d is None:
                continue
            ax_bgt.bar(x[j] + offset, d.n, bar_w * 0.9,
                       color=st["color"], alpha=0.85,
                       edgecolor="white", linewidth=0.3)
            ax_bgt.text(x[j] + offset, d.n * 1.12,
                        f"{d.n:,}", ha="center", va="bottom",
                        fontsize=5.8, rotation=50)
            if not added:
                legend_handles_bgt.append(
                    mpatches.Patch(facecolor=st["color"], label=st["label"], alpha=0.85))
                added = True

    ax_bgt.set_xticks(x)
    ax_bgt.set_xticklabels(maze_xlbls, fontsize=9)
    ax_bgt.set_ylabel("Episodes Trained  (log scale)")
    ax_bgt.set_yscale("log")
    ax_bgt.set_title("Training Budget per Agent × Maze\n(log scale — unequal budgets bias learning-speed comparison)",
                     fontweight="bold")
    ax_bgt.legend(handles=legend_handles_bgt, fontsize=8, loc="upper right")
    ax_bgt.grid(True, axis="y", alpha=0.22, which="both")
    ax_bgt.text(0.01, -0.10,
                "Agents were given different episode budgets. Learning-speed comparisons\n"
                "by episode count are not fair without normalisation.",
                transform=ax_bgt.transAxes, fontsize=7, color="#777", style="italic")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_maze_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def _find(records: list[AgentData], agent: str, maze_id: int) -> Optional[AgentData]:
    match = [r for r in records if r.agent == agent and r.maze_id == maze_id]
    return match[0] if match else None


# ── PPO training dynamics ─────────────────────────────────────────────────────

def plot_ppo_dynamics(ppo_updates: dict[int, pd.DataFrame], out_dir: str) -> None:
    """
    PPO update-level training metrics.

    These are NOT comparable to episode-based learning curves — they show
    what happens inside each gradient update step, not per-environment-episode.

    Metrics:
      policy_loss  — should decrease as the policy improves
      value_loss   — should decrease as the value function converges
      entropy      — should decrease (less random exploration) as policy focuses
      kl           — KL divergence; the PPO clip keeps this small
      clip_frac    — fraction of policy ratios clipped by the ε bound;
                     should stay low; spikes suggest large policy updates
    """
    if not ppo_updates:
        return

    maze_ids = sorted(ppo_updates.keys())
    metrics  = [
        ("policy_loss", "Policy Loss",    "↓ as policy improves"),
        ("value_loss",  "Value Loss",     "↓ as value fn converges"),
        ("entropy",     "Entropy",        "↓ as policy concentrates"),
        ("kl",          "KL Divergence",  "stays small (PPO clip constraint)"),
        ("clip_frac",   "Clip Fraction",  "fraction of ratios clipped by ε"),
    ]

    n_cols = len(maze_ids)
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "PPO Training Dynamics (update-level)\n"
        "These metrics are internal to PPO and not comparable to episode-based agents.",
        fontsize=13, fontweight="bold",
    )

    maze_colors = {1: "#9B59B6", 2: "#7D3C98", 3: "#5B2C6F"}

    for col, maze_id in enumerate(maze_ids):
        df       = ppo_updates[maze_id]
        updates  = df["update"].values
        color    = maze_colors.get(maze_id, "#888")
        maze_lbl = MAZE_LABELS.get(maze_id, f"Maze {maze_id}")

        for row, (col_name, title, note) in enumerate(metrics):
            ax = axes[row][col]

            if row == 0:
                ax.set_title(maze_lbl, fontweight="bold", fontsize=11, pad=8)

            if col_name not in df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="#aaa")
                ax.axis("off")
                continue

            vals     = df[col_name].values.astype(float)
            smoothed = _smooth(vals, min(10, len(vals)))

            ax.plot(updates, vals, color=color, lw=1.0, alpha=0.45)
            ax.plot(updates, smoothed, color=color, lw=2.0, alpha=0.95,
                    label="smoothed")

            ax.set_ylabel(title, fontsize=9)
            ax.set_xlabel("PPO Update", fontsize=8)
            ax.grid(True, alpha=0.18)
            ax.text(0.02, 0.97, note,
                    transform=ax.transAxes, fontsize=6.5,
                    va="top", color="#555", style="italic")

            # Eval checkpoints (sparse rows with rollout_success present)
            if "eval_reached" in df.columns:
                eval_mask = df["eval_reached"].notna()
                for u in updates[eval_mask]:
                    ax.axvline(u, color="gray", ls=":", lw=0.7, alpha=0.5)
                if col == 0 and row == 0:
                    ax.axvline(u, color="gray", ls=":", lw=0.7, alpha=0.5,
                               label="eval checkpoint")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ppo_training_dynamics.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RL agent results (v2).")
    parser.add_argument("--maze",   type=int, default=None, choices=[1, 2, 3],
                        help="Restrict to one maze (default: all).")
    parser.add_argument("--window", type=int, default=50,
                        help="Smoothing window for time-series plots (default: 50).")
    args = parser.parse_args()

    results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    out_dir      = os.path.join(results_root, "comparison")

    print(f"\nScanning: {results_root}\n")
    records = discover_results(results_root, maze_filter=args.maze)

    if not records:
        print("\nNo training CSVs found.")
        print("Expected: Results/<AGENT>/training_<AGENT>_maze<N>.csv")
        return

    print_kpi(records)

    mazes = sorted({r.maze_id for r in records})
    print(f"\nGenerating per-maze dashboards: maze(s) {mazes}")
    for maze_id in mazes:
        plot_maze_comparison(maze_id, records, out_dir, args.window)

    if len(mazes) > 1:
        print("\nGenerating cross-maze summary (solve rate + efficiency)...")
        plot_cross_maze_summary(records, out_dir)

    ppo_updates = discover_ppo_updates(results_root)
    if ppo_updates:
        print("\nGenerating PPO update-level dynamics...")
        plot_ppo_dynamics(ppo_updates, out_dir)

    print(f"\nAll figures saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
