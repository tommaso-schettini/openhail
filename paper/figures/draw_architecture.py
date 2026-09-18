"""Generate the manuscript's vector architecture diagram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

OUTPUT = Path(__file__).resolve().parent


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set(xlim=(0, 7.4), ylim=(0, 4.4), aspect="equal")
    ax.axis("off")

    def box(x, y, w, h, title, subtitle, external=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.055",
                edgecolor="#334155",
                facecolor="#edf3f7" if external else "white",
                linewidth=0.9,
                zorder=3,
            )
        )
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", zorder=4)
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.7,
            zorder=4,
        )

    def arrow(points, style="-|>", dashed=False):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        ax.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle=style,
                mutation_scale=9,
                color="#475569",
                linewidth=0.9,
                linestyle="--" if dashed else "-",
                zorder=2,
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.16, 0.2),
            7.08,
            2.77,
            boxstyle="round,pad=0.025,rounding_size=0.075",
            edgecolor="#94a3b8",
            facecolor="#f8fafc",
            linewidth=0.8,
            zorder=0,
        )
    )
    ax.text(0.32, 0.36, "OpenHail components", fontsize=8, color="#475569")

    box(
        0.35,
        3.58,
        2.65,
        0.62,
        "Configuration",
        "Demand, fleet, charging, decision epochs",
        True,
    )
    box(
        4.47,
        3.58,
        2.55,
        0.62,
        "Control policy",
        "Assignment, repositioning, charging",
        True,
    )
    box(0.4, 2.05, 2.03, 0.65, "OpenhailInstance", "Instance data and parameters")
    box(3.03, 2.05, 2.08, 0.65, "OpenhailEnv", "reset / step and event loop")
    box(0.39, 0.74, 1.57, 0.63, "RequestManager", "Demand and request buffer")
    box(2.08, 0.74, 1.57, 0.63, "VehicleManager", "Jobs, batteries, queues")
    box(3.77, 0.74, 1.57, 0.63, "StateObserver", "Policy observations")
    box(5.46, 0.74, 1.57, 0.63, "SummaryManager", "Operational statistics")

    arrow([(1.42, 3.56), (1.42, 2.72)])
    arrow([(2.7, 3.56), (2.7, 3.17), (3.53, 3.17), (3.53, 2.72)], dashed=True)
    ax.text(2.82, 3.23, "Decision epochs", ha="left", va="bottom", fontsize=7.6)
    arrow([(2.45, 2.37), (3.01, 2.37)])

    arrow([(4.91, 3.56), (4.91, 2.72)])
    ax.text(4.85, 3.15, "Action", ha="right", fontsize=7.8)
    arrow([(5.13, 2.38), (6.35, 2.38), (6.35, 3.56)])
    ax.text(
        6.28, 3.19, "Observation,\nreward, info", ha="right", va="center", fontsize=7.8
    )

    ax.plot([4.07, 4.07], [2.03, 1.7], color="#475569", linewidth=0.9)
    ax.plot([1.175, 6.245], [1.7, 1.7], color="#475569", linewidth=0.9)
    ax.text(3.89, 1.86, "Coordinates", ha="right", fontsize=7.8)
    for x in (1.175, 2.865, 4.555, 6.245):
        arrow([(x, 1.7), (x, 1.39)])

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUTPUT / f"architecture.{suffix}",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
