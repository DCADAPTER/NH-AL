"""Persist acquisition replay results and their diagnostic visualizations."""
import csv
import json
from pathlib import Path

import numpy as np

from replay import predict_moments, score_acquisition_many


def save_replay_results(replay, results, output_dir, n_mc_samples, random_seed,
                        plot_limits, figure_dpi, figsize_field, figsize_distribution,
                        colormap, font_sizes, elapsed_seconds):
    """Write repetition results, summary metadata, and acquisition plots."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = replay.problem
    model = replay.model
    acquisition = replay.acquisition
    run_config = replay.run_config
    evaluation = replay.evaluation
    dimension = int(problem.d)
    if dimension < 2:
        raise ValueError("Acquisition field plot requires at least two dimensions")
    if not results:
        raise ValueError("At least one acquisition repetition result is required")

    acquisition_config = run_config["acquisition"]
    acquisition_name = str(
        acquisition_config.get(
            "type", type(acquisition).__name__.removesuffix("Acquisition")
        )
    )
    direction_label = "Maximum" if acquisition.direction == "max" else "Minimum"
    values = np.asarray([row["acquisition_value"] for row in results], dtype=float)
    best_points = np.asarray([row["x"] for row in results], dtype=float)
    if best_points.shape != (len(results), dimension) or not np.isfinite(values).all():
        raise ValueError("Acquisition results must contain finite scores and dimension-matched points")

    mean_value = float(np.mean(values))
    std_value = float(np.std(values))
    median_value = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    iqr_value = float(q3 - q1)
    run_name = replay.run_data_dir.parent.name
    problem_name = getattr(problem, "kind", type(problem).__name__)
    font_title = font_sizes["title"]
    font_label = font_sizes["label"]
    font_tick = font_sizes["tick"]
    font_legend = font_sizes["legend"]
    font_annotation = font_sizes["annotation"]

    x1, x2 = np.meshgrid(
        np.linspace(*plot_limits, 150),
        np.linspace(*plot_limits, 150),
    )
    grid = np.zeros((x1.size, dimension), dtype=float)
    grid[:, 0], grid[:, 1] = x1.ravel(), x2.ravel()
    scores = score_acquisition_many(acquisition, grid).reshape(x1.shape)
    mean, _ = predict_moments(model, grid)
    mean = mean.reshape(x1.shape)
    true_g = np.fromiter(
        (problem.g(point.tolist()) for point in grid),
        dtype=float,
        count=len(grid),
    ).reshape(x1.shape)

    fig, ax = plt.subplots(figsize=figsize_field, dpi=figure_dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f9fc")
    contour = ax.contourf(
        x1, x2, scores, levels=36, cmap=colormap, extend="max", zorder=0
    )
    if float(np.min(true_g)) < 0:
        ax.contourf(
            x1, x2, true_g,
            levels=[float(np.min(true_g)), 0.0],
            colors=["#f28e72"], alpha=0.18, zorder=1,
        )
    cbar = fig.colorbar(contour, ax=ax, pad=0.025, fraction=0.046, shrink=0.92)
    cbar.set_label(
        f"{acquisition_name} acquisition score", fontsize=font_label, labelpad=12
    )
    cbar.ax.tick_params(labelsize=font_tick, length=0, pad=6)
    cbar.outline.set_linewidth(0.8)
    if float(np.min(mean)) <= 0 <= float(np.max(mean)):
        ax.contour(
            x1, x2, mean, levels=[0.0], colors=["#fff8e7"],
            linewidths=3.6, zorder=3,
        )
        ax.contour(
            x1, x2, mean, levels=[0.0], colors=["#25324a"],
            linewidths=1.8, zorder=4,
        )
    if float(np.min(true_g)) <= 0 <= float(np.max(true_g)):
        ax.contour(
            x1, x2, true_g, levels=[0.0], colors=["white"],
            linestyles=":", linewidths=4.0, zorder=4,
        )
        ax.contour(
            x1, x2, true_g, levels=[0.0], colors=["#e76f51"],
            linewidths=2.0, zorder=5,
        )
    ax.scatter(
        replay.X_train[:, 0], replay.X_train[:, 1], s=38, c="white",
        edgecolors="#263859", linewidths=0.85, alpha=0.9, zorder=6,
    )
    ax.scatter(
        best_points[0, 0], best_points[0, 1], s=260, c="#ffe066",
        edgecolors="#9c2f2f", marker="*", linewidths=1.4, zorder=7,
    )
    ax.set_title(
        f"{str(problem_name).title()} {acquisition_name} acquisition | "
        f"{run_name}, iteration {replay.iteration}",
        fontsize=font_title, pad=16, fontweight="bold", color="#263859",
    )
    ax.set_xlabel("$x_1$", fontsize=font_label, labelpad=8)
    ax.set_ylabel("$x_2$", fontsize=font_label, labelpad=8)
    ax.set_xlim(*plot_limits)
    ax.set_ylim(*plot_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(True, color="white", linewidth=0.8, alpha=0.55)
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_color("#d5dce5")
        spine.set_linewidth(0.9)
    field_legend = [
        Line2D([0], [0], color="#25324a", lw=2,
               label=r"Metamodel mean boundary ($\mu(x)=0$)"),
        Line2D([0], [0], color="#e76f51", lw=2,
               label=r"True problem boundary ($G(x)=0$)"),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="white",
            markeredgecolor="#263859", markersize=8,
            label=f"Training data (n={len(replay.X_train)})",
        ),
        Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="#ffe066",
            markeredgecolor="#9c2f2f", markersize=13,
            label=f"First-run {direction_label.lower()} ({values[0]:.2e})",
        ),
    ]
    fig.subplots_adjust(left=0.11, right=0.88, top=0.88, bottom=0.22)
    fig.legend(
        handles=field_legend, loc="lower center",
        bbox_to_anchor=(0.47, 0.025), ncol=2,
        frameon=True, fancybox=True, framealpha=0.98, facecolor="white",
        edgecolor="#d5dce5", fontsize=font_legend,
    )
    field_path = output_dir / "acquisition_field_plot.png"
    fig.savefig(field_path, dpi=figure_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, (ax, stats_ax) = plt.subplots(
        1, 2, figsize=figsize_distribution, dpi=figure_dpi,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    fig.patch.set_facecolor("white")
    ax.boxplot(
        values, positions=[1], widths=0.42, patch_artist=True, showmeans=True,
        boxprops=dict(facecolor="#b9d8e8", color="#356b83", linewidth=1.8),
        whiskerprops=dict(color="#356b83", linewidth=1.5),
        capprops=dict(color="#356b83", linewidth=1.5),
        medianprops=dict(color="#d1495b", linewidth=2.4),
        meanprops=dict(
            marker="D", markerfacecolor="#f4a261",
            markeredgecolor="#8a4f16", markersize=7,
        ),
        flierprops=dict(
            marker="o", markerfacecolor="#e76f51",
            markeredgecolor="white", markersize=5, alpha=0.8,
        ),
    )
    jitter = np.random.default_rng(random_seed + 1).normal(0, 0.055, size=len(values))
    ax.scatter(
        np.ones_like(values) + jitter, values, color="#356b83", alpha=0.48,
        edgecolor="white", linewidth=0.55, s=34, zorder=4,
    )
    ax.set_title(
        f"{direction_label} {acquisition_name} score ({len(results)} repetitions)\n"
        f"{run_name}, iteration {replay.iteration}",
        fontsize=font_title, pad=14, fontweight="bold", color="#263859",
    )
    ax.set_ylabel(
        f"Best {acquisition_name} acquisition score",
        fontsize=font_label, labelpad=10,
    )
    ax.set_xticks([1])
    ax.set_xticklabels(["Monte Carlo runs"])
    ax.set_xlim(0.55, 1.45)
    ax.set_axisbelow(True)
    ax.grid(True, color="#e7edf2", linewidth=0.9, axis="y")
    for spine in ax.spines.values():
        spine.set_color("#d5dce5")
        spine.set_linewidth(0.9)
    stats_text = (
        f"Mean        {mean_value:.2e}\n"
        f"Std. dev.   {std_value:.2e}\n"
        f"Median      {median_value:.2e}\n"
        f"IQR         {iqr_value:.2e}\n"
        f"Minimum     {float(np.min(values)):.2e}\n"
        f"Maximum     {float(np.max(values)):.2e}"
    )
    stats_ax.set_facecolor("#f6f8fb")
    stats_ax.set_xticks([])
    stats_ax.set_yticks([])
    for spine in stats_ax.spines.values():
        spine.set_color("#d5dce5")
        spine.set_linewidth(0.9)
    stats_ax.text(
        0.08, 0.91, f"Summary statistics (N={len(results)})",
        transform=stats_ax.transAxes, fontsize=font_label,
        fontweight="bold", color="#263859", va="top",
    )
    stats_ax.text(
        0.08, 0.77, stats_text, transform=stats_ax.transAxes,
        fontsize=font_annotation, fontfamily="monospace",
        linespacing=1.8, color="#344054", va="top",
    )
    ax.legend(
        handles=[
            Patch(facecolor="#b9d8e8", edgecolor="#356b83", label="Interquartile range"),
            Line2D([0], [0], color="#d1495b", lw=2.4,
                   label=f"Median: {median_value:.2e}"),
            Line2D(
                [0], [0], marker="D", color="w", markerfacecolor="#f4a261",
                markeredgecolor="#8a4f16", markersize=7,
                label=f"Mean: {mean_value:.2e}",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#356b83",
                markeredgecolor="white", markersize=7, label="Individual runs",
            ),
        ],
        loc="upper left", frameon=True, fancybox=True, framealpha=0.98,
        facecolor="white", edgecolor="#d5dce5", fontsize=font_legend,
    )
    fig.tight_layout(pad=2.0, w_pad=2.0)
    distribution_path = output_dir / "acquisition_extrema_distribution.png"
    fig.savefig(
        distribution_path, dpi=figure_dpi, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    results_csv = output_dir / "acquisition_repetitions.csv"
    columns = ["repetition", "best_acquisition_value"] + [
        f"x{index}" for index in range(dimension)
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "repetition": row["repetition"],
                "best_acquisition_value": row["acquisition_value"],
                **{f"x{index}": value for index, value in enumerate(row["x"])},
            })

    theta = getattr(model, "theta", None)
    summary = {
        "run_data_dir": str(replay.run_data_dir),
        "iteration": replay.iteration,
        "model_snapshot": evaluation.get("model_snapshot"),
        "problem": run_config["problem"],
        "metamodel_factory": run_config["metamodel"].get("factory"),
        "theta": None if theta is None else np.asarray(theta).tolist(),
        "training_samples": len(replay.X_train),
        "acquisition": acquisition_config,
        "acquisition_direction": acquisition.direction,
        "mc_samples_per_repetition": int(n_mc_samples),
        "repetitions": len(results),
        "random_seed": random_seed,
        "best_acquisition": {
            "mean": mean_value,
            "std": std_value,
            "median": median_value,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": iqr_value,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
        "elapsed_seconds": float(elapsed_seconds),
    }
    summary_path = output_dir / "replay_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    return {
        "field_plot": field_path,
        "distribution_plot": distribution_path,
        "results_csv": results_csv,
        "summary_json": summary_path,
        "summary": summary,
    }
