"""
Plotting utilities for evaluation results.
- Scatter plot
- Cactus plot
- Box plot
- Violin plot
- Heat map
"""

import math, os, random
from typing import *

import matplotlib.pyplot as plt, matplotlib.gridspec
import numpy as np
from matplotlib import rcParams

# Global plotting configuration
PLOT_CONFIG = {
    "font_family": "serif",
    "font_serif": ["Times New Roman"],
    "use_tex": False,
    "dpi": 300,
    "figure_format": "png",
    "default_figsize": (9, 4),
    "user_fontsize": 12,
}

# Color schemes
COLORS = {
    "standard": ["b", "g", "r", "c", "m", "y", "k"],
    "pastel": ["pink", "lightblue", "lightgreen", "lightyellow", "red"],
    "markers": [".", ",", "o", "<", ">", "s", "p", "*", "x", "h", "H", "D", "|"],
}


def setup_plot_style() -> None:
    """Configure global matplotlib settings."""
    rcParams["font.family"] = PLOT_CONFIG["font_family"]
    rcParams["font.serif"] = PLOT_CONFIG["font_serif"]
    rcParams["text.usetex"] = PLOT_CONFIG["use_tex"]
    plt.rcParams["font.size"] = PLOT_CONFIG["user_fontsize"]


def find_csv(path: str) -> List[str]:
    """Find all CSV files in given directory and subdirectories."""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(path)
        for f in files
        if f.endswith(".csv")
    ]


class BasePlot:
    """Base class for all plotting functionality."""

    def __init__(self):
        setup_plot_style()
        self.use_log_scale: bool = False
        self.upper_bound: float = 1.0

    def save_plot(self, fig: plt.Figure, output_dir: str, filename: str) -> None:
        """Save plot to file with standard parameters."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_file = os.path.join(
            output_dir, f"{filename}.{PLOT_CONFIG['figure_format']}"
        )
        fig.savefig(out_file, dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
        plt.close(fig)

    def _apply_log_scale(self, values: List[float]) -> List[float]:
        """Apply log scale to values if enabled."""
        return [math.log(v, 10) for v in values] if self.use_log_scale else values


class ScatterPlot(BasePlot):
    """Scatter plot implementation with support for single and multi-group data."""

    def __init__(self, name_a: str = "tool a", name_b: str = "tool b"):
        super().__init__()
        self.tool_a_name = name_a
        self.tool_b_name = name_b

    def plot(
        self,
        data: Union[Tuple[List, List], Tuple[Tuple[List, List], Tuple[List, List]]],
        output_dir: str = "",
        filename: str = "scatter",
        save: bool = False,
        multi_group: bool = False,
    ) -> None:
        """Create scatter plot with optional multi-group support."""
        fig, ax = plt.subplots()

        if multi_group:
            self._plot_multi_groups(ax, data)
        else:
            self._plot_single_group(ax, data)

        self._setup_axes(ax)

        if save:
            self.save_plot(fig, output_dir, filename)
        else:
            plt.show()

    def _plot_single_group(self, ax: plt.Axes, data: Tuple[List, List]) -> None:
        x, y = data
        x = self._apply_log_scale(x)
        y = self._apply_log_scale(y)
        ax.scatter(x, y, alpha=0.5, marker="x", c="blue")

    def _plot_multi_groups(
        self, ax: plt.Axes, data: Tuple[Tuple[List, List], Tuple[List, List]]
    ) -> None:
        colors = ("blue", "green")
        markers = ("x", "s")
        for (x, y), color, marker in zip(data, colors, markers):
            x = self._apply_log_scale(x)
            y = self._apply_log_scale(y)
            ax.scatter(x, y, alpha=0.5, marker=marker, c=color)

    def _setup_axes(self, ax: plt.Axes) -> None:
        ax.set_xlabel(f"Result of {self.tool_a_name}", fontsize=11)
        ax.set_ylabel(f"Result of {self.tool_b_name}", fontsize=12)
        bound = (
            math.log(self.upper_bound, 10) if self.use_log_scale else self.upper_bound
        )
        ax.plot([0, bound], [0, bound], "k", linewidth=0.7)
        ax.set_title("")


class CactusPlot(BasePlot):
    """Cactus plot implementation for comparing multiple tools/methods."""

    def __init__(self):
        super().__init__()
        self.upper_bound = 9

    def plot(
        self,
        data: List[List[float]],
        output_dir: str = "",
        filename: str = "cactus",
        save: bool = False,
    ) -> None:
        """Create cactus plot from multiple data series."""
        fig, ax = plt.subplots()
        processed_data = self._process_data(data)
        self._plot_series(ax, processed_data)
        self._setup_axes(ax)

        if save:
            self.save_plot(fig, output_dir, filename)
        else:
            plt.show()

    def _process_data(self, data: List[List[float]]) -> List[List[float]]:
        # Remove timeouts and calculate cumulative sums
        cleaned_data = [[x for x in series if x < self.upper_bound] for series in data]
        return [list(np.cumsum(sorted(series))) for series in cleaned_data]

    def _plot_series(self, ax: plt.Axes, data: List[List[float]]) -> None:
        for i, series in enumerate(data):
            color = random.choice(COLORS["standard"])
            marker = random.choice(COLORS["markers"])
            ax.plot(
                series, color=color, marker=marker, label=f"{i}-th tool", markevery=3
            )

    def _setup_axes(self, ax: plt.Axes) -> None:
        ax.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
        ax.set_xlabel("#solved instances")
        ax.set_ylabel("Runtime [sec]" + (" log scale" if self.use_log_scale else ""))
        ax.legend(loc="lower right")


class BoxPlot(BasePlot):
    """Box plot implementation with support for single and multi-group data."""

    def plot(
        self,
        data: Union[List[List[float]], List[List[List[float]]]],
        output_dir: str = "",
        filename: str = "box",
        save: bool = False,
        multi_group: bool = False,
    ) -> None:
        """Create box plot with optional multi-group support."""
        if multi_group:
            self._plot_multi_groups(data)
        else:
            self._plot_single_group(data)

        if save:
            self.save_plot(plt.gcf(), output_dir, filename)
        else:
            plt.show()

    def _plot_single_group(self, data: List[List[float]]) -> None:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["default_figsize"])
        bplot = ax.boxplot(data, vert=True, patch_artist=True)

        colors = random.sample(COLORS["pastel"], len(data))
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)

        self._setup_single_axes(ax, len(data))

    def _plot_multi_groups(self, data: List[List[List[float]]]) -> None:
        fig, axes = plt.subplots(
            nrows=1, ncols=len(data), figsize=PLOT_CONFIG["default_figsize"]
        )

        colors = random.sample(COLORS["pastel"], len(data[0]))
        for ax, group in zip(axes, data):
            bplot = ax.boxplot(group, vert=True, patch_artist=True)
            for patch, color in zip(bplot["boxes"], colors):
                patch.set_facecolor(color)
            self._setup_single_axes(ax, len(group))

    def _setup_single_axes(self, ax: plt.Axes, num_boxes: int) -> None:
        ax.set_xlabel("xlabel")
        ax.set_ylabel("ylabel")
        ax.set_xticklabels([f"x{i}" for i in range(num_boxes)])


class ViolinPlot(BasePlot):
    """Violin plot implementation with support for single and multi-group data."""

    def plot(
        self,
        data: Union[List[List[float]], List[List[List[float]]]],
        output_dir: str = "",
        filename: str = "violin",
        save: bool = False,
        multi_group: bool = False,
    ) -> None:
        """Create violin plot with optional multi-group support."""
        if multi_group:
            self._plot_multi_groups(data)
        else:
            self._plot_single_group(data)

        if save:
            self.save_plot(plt.gcf(), output_dir, filename)
        else:
            plt.show()

    def _plot_single_group(self, data: List[List[float]]) -> None:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["default_figsize"])
        parts = ax.violinplot(data, showmeans=True, showmedians=True)

        colors = random.sample(COLORS["pastel"], len(data))
        for patch, color in zip(parts["bodies"], colors):
            patch.set_facecolor(color)

        self._setup_single_axes(ax, len(data))

    def _plot_multi_groups(self, data: List[List[List[float]]]) -> None:
        fig, axes = plt.subplots(
            nrows=1, ncols=len(data), figsize=PLOT_CONFIG["default_figsize"]
        )

        colors = random.sample(COLORS["pastel"], len(data[0]))
        for ax, group in zip(axes, data):
            parts = ax.violinplot(group, showmeans=True, showmedians=True)
            for patch, color in zip(parts["bodies"], colors):
                patch.set_facecolor(color)
            self._setup_single_axes(ax, len(group))

    def _setup_single_axes(self, ax: plt.Axes, num_boxes: int) -> None:
        ax.set_xlabel("xlabel")
        ax.set_ylabel("ylabel")
        ax.set_xticklabels([f"x{i}" for i in range(num_boxes)])


class HeatMap(BasePlot):
    """Heat map implementation with support for single and multi-group data."""

    def plot(
        self,
        data: Union[List[List[float]], List[List[List[float]]]],
        output_dir: str = "",
        filename: str = "heatmap",
        save: bool = False,
        multi_group: bool = False,
    ) -> None:
        """Create heat map with optional multi-group support."""
        if multi_group:
            self._plot_multi_groups(data)
        else:
            self._plot_single_group(data)

        if save:
            self.save_plot(plt.gcf(), output_dir, filename)
        else:
            plt.show()
        # TBD


def generate_scatter_data() -> Tuple[List[float], List[float]]:
    """Generate sample data for scatter plot."""
    return ([0.4, 0.6, 0.33, 0.2], [0.3, 0.7, 0.6, 0.33])


def generate_multi_group_data() -> (
    Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]]]
):
    """Generate sample data for multi-group scatter plot."""
    sat_data = ([2, 3, 4, 8], [1, 5, 6, 6])
    unsat_data = ([2, 3, 4, 5], [1, 2, 6, 8])
    return (sat_data, unsat_data)


def generate_cactus_data() -> List[List[float]]:
    """Generate sample data for cactus plot."""
    return [
        [1, 2, 3, 3.5, 2.5, 4, 6],
        [5, 2, 2, 5, 8, 2, 10],
        [1, 7, 3, 3, 2, 4, 10, 12, 15],
        [5, 6, 1, 2, 3, 8, 7, 9, 12, 10],
    ]


def generate_box_data() -> List[List[float]]:
    """Generate sample data for box plot."""
    return [[1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]]


def generate_multi_group_box_data() -> List[List[List[float]]]:
    """Generate sample data for multi-group box plot."""
    return [
        [[1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]],
        [[2, 2, 3, 4, 3, 5, 6], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]],
        [[2, 2, 3, 4, 3, 5, 6], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]],
    ]


def test_scatter_plot():
    """Demonstrate basic scatter plot usage."""
    data = generate_scatter_data()
    plotter = ScatterPlot("Tool A", "Tool B")
    plotter.plot(data)


def test_scatter_plot_multi_groups():
    """Demonstrate scatter plot with multiple groups."""
    data = generate_multi_group_data()
    plotter = ScatterPlot("SAT Solver 1", "SAT Solver 2")
    plotter.plot(data, multi_group=True)


def test_cactus_plot():
    """Demonstrate cactus plot usage."""
    data = generate_cactus_data()
    plotter = CactusPlot()
    plotter.plot(data)


def test_box_plot():
    """Demonstrate box plot usage."""
    data = generate_box_data()
    plotter = BoxPlot()
    plotter.plot(data)


def test_box_plot_multi_groups():
    """Demonstrate box plot with multiple groups."""
    data = generate_multi_group_box_data()
    plotter = BoxPlot()
    plotter.plot(data, multi_group=True)


if __name__ == "__main__":
    # Uncomment tests to run
    test_scatter_plot()
    # test_scatter_plot_multi_groups()
    # test_cactus_plot()
    # test_box_plot()
    # test_box_plot_multi_groups()
