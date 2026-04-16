import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends import backend_pgf
from matplotlib.ticker import MaxNLocator


#### TIKZPLOTLIB MONKEYPATCH ####
def _tex_escape(string):
    return backend_pgf._tex_escape(string).replace("&", "\\&")


backend_pgf.common_texification = _tex_escape

import webcolors

webcolors.CSS3_HEX_TO_NAMES = webcolors._definitions._CSS3_HEX_TO_NAMES
####END MONKEYPATCH###

import tikzplotlib

TITLE_FONTSIZE = 10


def _ensure_dir(path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def scatterplot_with_hist(x, y, save_path, xlabel=None, ylabel=None, title=None, clip_percentile=None):
    """Create a scatterplot with histogram of x values where y > 1 below."""
    x, y = np.array(x), np.array(y)
    x_violated = x[y > 1]

    fig, (ax_scatter, ax_hist) = plt.subplots(
        2, 1, figsize=(3.5, 4),
        height_ratios=[3, 1],
        sharex=True
    )

    if clip_percentile is not None:
        low = 100 - clip_percentile
        x_low, x_high = np.percentile(x, low), np.percentile(x, clip_percentile)
        y_low, y_high = np.percentile(y, low), np.percentile(y, clip_percentile)
        ax_scatter.set_xlim(x_low, x_high)
        ax_scatter.set_ylim(y_low, y_high)

    ax_scatter.scatter(x, y, alpha=0.6, s=15, edgecolors='none')
    if title:
        ax_scatter.set_title(title, fontsize=11, pad=10)
    if ylabel:
        ax_scatter.set_ylabel(ylabel)
    ax_scatter.tick_params(labelbottom=True)

    ax_hist.hist(x_violated, bins='auto', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_hist.set_ylabel('Number of EJR+ violations')
    if xlabel:
        ax_hist.set_xlabel(xlabel)

    plt.subplots_adjust(hspace=0.1)

    _ensure_dir(save_path)

    # Create separate figures w TikZ
    base_path = os.path.splitext(save_path)[0]

    # Scatter plot
    fig_scatter = plt.figure(figsize=(3.5, 3))
    ax_s = fig_scatter.add_subplot(111)
    ax_s.scatter(x, y, alpha=0.6, s=15, edgecolors='none')
    if clip_percentile is not None:
        ax_s.set_xlim(x_low, x_high)
        ax_s.set_ylim(y_low, y_high)
    if title:
        ax_s.set_title(title, fontsize=11, pad=10)
    if ylabel:
        ax_s.set_ylabel(ylabel)
    if xlabel:
        ax_s.set_xlabel(xlabel)
    ax_s.margins(x=0.05, y=0.05)  # 5% padding on each side
    ax_s.yaxis.set_major_locator(MaxNLocator(nbins=8))

    plt.tight_layout()
    tikzplotlib.save(
        base_path + '_scatter.tex'
    )
    plt.close()

    # Histogram
    fig_hist = plt.figure(figsize=(3.5, 1))
    ax_h = fig_hist.add_subplot(111)
    ax_h.hist(x_violated, bins='auto', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax_h.set_ylabel('Number of EJR+ violations')
    if xlabel:
        ax_h.set_xlabel(xlabel)
    if clip_percentile is not None:
        ax_h.set_xlim(x_low, x_high)

    plt.tight_layout()
    tikzplotlib.save(
        base_path + '_hist.tex'
    )
    plt.close()


def histogram(data, save_path, bins='auto', xlabel=None, ylabel=None, title=None):
    """Create a histogram."""
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=TITLE_FONTSIZE, wrap=True)
    plt.tight_layout()
    _ensure_dir(save_path)
    save_path_tikz = os.path.splitext(save_path)[0] + '.tex'
    tikzplotlib.save(save_path_tikz)
    plt.savefig(save_path)
    plt.close()


def price_system_measures_plot(
        all_max_deviation_money: list,
        avg_residuals: list,
        budget_non_uniformities: list,
        avg_payment_stds: list,
        save_path: str,
        title: str = None
):
    """Create a 2x2 subplot with histograms of price system measures."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: max_deviation_money histogram (aggregated over all c and instances) in [0,1]
    ax = axes[0, 0]
    ax.hist(all_max_deviation_money, bins=20, range=(0, 1), alpha=0.7, edgecolor='black')
    ax.set_xlabel('Available Money for Deviation')
    ax.set_ylabel('Count (all candidates)')
    ax.set_title('Deviation Money (Stability Margin)')

    # Top-right: avg_residual histogram
    ax = axes[0, 1]
    ax.hist(avg_residuals, bins='auto', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Average Residual')
    ax.set_ylabel('Instances')
    ax.set_title('Avg Residual Distribution')

    # Bottom-left: budget_non_uniformity histogram
    ax = axes[1, 0]
    ax.hist(budget_non_uniformities, bins='auto', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Sum of Pairwise Budget Diffs')
    ax.set_ylabel('Instances')
    ax.set_title('Budget Non-Uniformity Distribution')

    # Bottom-right: avg_payment_std histogram
    ax = axes[1, 1]
    ax.hist(avg_payment_stds, bins='auto', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Avg Payment Std per Candidate')
    ax.set_ylabel('Instances')
    ax.set_title('Payment Std Distribution')

    if title:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE + 2)

    plt.tight_layout()
    _ensure_dir(save_path)
    save_path_tikz = os.path.splitext(save_path)[0] + '.tex'
    tikzplotlib.save(save_path_tikz)
    plt.savefig(save_path)
    plt.close()
