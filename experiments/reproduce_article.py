# =============================================================================
# Script for reproducing all figures from the Physica Scripta article on
# the Quantum Sure Thing Principle (QSTP) in the EWL quantum Prisoner's Dilemma.
# Only comments were added; the scientific implementation is unchanged.
# =============================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from qstp_ewl.configs import (
    ORIGINAL_STANDARD,
    ORIGINAL_COMPLEMENTARY,
    FIG6_STANDARD,
    FIG7_COMPLEMENTARY,
)
from qstp_ewl.core import ExperimentPoint, evaluate
from qstp_ewl.plotting import article_original_panel, perfect_panel, save

# Project root directory.
ROOT = Path(__file__).resolve().parents[1]
# Directory where article figures are stored.
OUT = ROOT / "figures" / "article"



# Convert representative theta values into LaTeX labels.
def _theta_text(theta: float) -> str:
    values = [
        (np.pi / 2, r"\pi/2"),
        (2 * np.pi / 5, r"2\pi/5"),
        (3 * np.pi / 10, r"3\pi/10"),
        (17 * np.pi / 30, r"17\pi/30"),
        (19 * np.pi / 30, r"19\pi/30"),
    ]
    for target, text in values:
        if np.isclose(theta, target):
            return text
    return rf"{theta / np.pi:.3g}\pi"



# Generate one standalone EWL panel reproducing the published article.
def standalone_original_panel(panel, thetas, filename, complementary=False, phi=np.pi, n=701):
    """Article-style standalone EWL panel with curve labels in grayscale."""
    # Sweep the entanglement parameter gamma.
    gammas = np.linspace(0.0, np.pi / 2.0, n)
    # Compute conditional reference probabilities.
    reference = [
        evaluate(
            ExperimentPoint(
                panel.t_a, panel.t_b, 1.0, thetas[0], phi, gamma, panel.entangler
            )
        )
        for gamma in gammas
    ]

    # Create the publication-quality figure.
    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    # Green hypothesis region exactly as displayed in the article panel.
    if complementary:
        ax.axhspan(0.5, 1.0, color="#c8f7c5", alpha=0.75, zorder=0)

    ax.plot(
        gammas,
        [p.p_a1_b1 for p in reference],
        color="#6064ff",
        linewidth=2.25,
        zorder=3,
    )
    ax.plot(
        gammas,
        [p.p_a1_b0 for p in reference],
        color="#ff6666",
        linewidth=2.25,
        zorder=3,
    )
    ax.plot(
        gammas,
        [p.p_a1_unconditional for p in reference],
        color="#ff7600",
        linewidth=3.0,
        linestyle=":",
        zorder=3,
    )

    # Gray colors matching the article.
    grays = ["#000000", "#555555", "#929292"]
    curves = []
    # Evaluate P_D for each representative theta.
    for theta, gray in zip(thetas, grays):
        probabilities = [
            evaluate(
                ExperimentPoint(
                    panel.t_a, panel.t_b, 1.0, theta, phi, gamma, panel.entangler
                )
            )
            for gamma in gammas
        ]
        values = np.array([p.p_d for p in probabilities])
        curves.append(values)
        ax.plot(gammas, values, color=gray, linewidth=2.05, zorder=4)

    # Standard panel: shade only the actual lower violation region.
    if not complementary:
        # Envelope used to shade the violation region.
        lower = np.minimum.reduce(curves)
        mask = lower < 0.5
        ax.fill_between(
            gammas,
            lower,
            0.5,
            where=mask,
            interpolate=True,
            color="#c8f7c5",
            alpha=0.75,
            zorder=0,
        )

    # Labels close to the right boundary, matching the supplied article image.
    x_label = 1.27 if complementary else 1.31
    offsets = [0.015, 0.0, -0.015] if complementary else [0.02, 0.0, -0.012]
    for theta, values, offset in zip(thetas, curves, offsets):
        y_label = float(np.interp(x_label, gammas, values)) + offset
        ax.text(
            x_label,
            y_label,
            rf"$\Theta={_theta_text(theta)}$",
            fontsize=10.5,
            color="black",
            clip_on=True,
        )

    ax.set_xlim(0.0, np.pi / 2.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\gamma$", fontsize=18, labelpad=18)
    ax.set_ylabel(r"$P_D$", fontsize=18, rotation=0, labelpad=25)
    ax.set_title(r"EWL ($-+$)", fontsize=19, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=11)
    ax.minorticks_on()
    fig.tight_layout()
    save(fig, OUT / filename)



# Execute the complete article-reproduction pipeline.
def main():
    # Theta ordering for the standard panel.
    standard_thetas = [np.pi / 2, 2 * np.pi / 5, 3 * np.pi / 10]
    # Theta ordering for the complementary panel.
    complementary_thetas = [19 * np.pi / 30, 17 * np.pi / 30, np.pi / 2]

    # Standalone panels used directly in the article and thesis.
    standalone_original_panel(
        ORIGINAL_STANDARD,
        standard_thetas,
        "Fig5a_EWL_standard",
        complementary=False,
        phi=np.pi,
    )
    standalone_original_panel(
        ORIGINAL_COMPLEMENTARY,
        complementary_thetas,
        "Fig5b_EWL_complementary",
        complementary=True,
        phi=0.0,
    )

    # Combined original-EWL figure.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    article_original_panel(axes[0], ORIGINAL_STANDARD, standard_thetas, phi=np.pi)
    article_original_panel(axes[1], ORIGINAL_COMPLEMENTARY, complementary_thetas, phi=0.0)
    axes[1].set_ylabel("")
    fig.tight_layout()
    save(fig, OUT / "Fig5_original_EWL")

    # Perfect entanglers, standard QSTP.
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for ax, panel in zip(axes, FIG6_STANDARD):
        perfect_panel(ax, panel)
    fig.tight_layout()
    save(fig, OUT / "Fig6_perfect_standard")

    # Perfect entanglers, complementary QSTP.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, panel in zip(axes.flat, FIG7_COMPLEMENTARY):
        perfect_panel(ax, panel)
    fig.tight_layout()
    save(fig, OUT / "Fig7_perfect_complementary")


if __name__ == "__main__":
    main()
