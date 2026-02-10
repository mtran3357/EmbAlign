import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List

class SpatialVisualizer:
    def __init__(self, atlas):
        self.atlas = atlas
        
    def plot_alignment(self, frame, result, ax=None, title=None):
        """
        Plots 3D alignment with wireframe ellipsoids and displacement lines.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
        inferred_labels = result.get('labels', [])
        aligned_coords = result.get('coords', np.array([]))
        n_cells = len(inferred_labels)
        
        # 1. Plot Atlas Gaussians (Wireframes for transparency)
        for label in set(inferred_labels):
            if label in self.atlas.means:
                mu = self.atlas.means[label]
                cov = self.atlas.covs[label]
                ex, ey, ez = self._generate_ellipsoid(mu, cov, scale=1.0)
                ax.plot_wireframe(ex, ey, ez, color='gray', alpha=0.5, linewidth=0.3)

        # 2. Extract Labels and Accuracy
        true_labels = []
        if frame.valid_df is not None and 'cell_name' in frame.valid_df.columns:
            true_labels = frame.valid_df['cell_name'].astype(str).tolist()

        if true_labels:
            is_correct = [inf == tru for inf, tru in zip(inferred_labels, true_labels)]
            colors = ['#2ca02c' if c else '#d62728' for c in is_correct]
            label_suffix = f" | Acc: {np.mean(is_correct):.1%}"
        else:
            colors = '#2ca02c'
            label_suffix = ""

        if n_cells > 0:
            # 2. Draw Displacement Lines and Centroids
            for i, label in enumerate(inferred_labels):
                if label in self.atlas.means:
                    atlas_mu = self.atlas.means[label]
                    ax.plot([aligned_coords[i, 0], atlas_mu[0]],
                            [aligned_coords[i, 1], atlas_mu[1]],
                            [aligned_coords[i, 2], atlas_mu[2]],
                            color='black', linestyle='--', linewidth=1, alpha=0.3)

            ax.scatter(aligned_coords[:, 0], aligned_coords[:, 1], aligned_coords[:, 2], 
                       c=colors, s=40, edgecolors='white', linewidth=0.5, alpha=0.9)

            # 3. Combined Annotation Logic
            if n_cells < 60:
                for i in range(n_cells):
                    pred = inferred_labels[i]
                    x, y, z = aligned_coords[i]
                    
                    if true_labels and not is_correct[i]:
                        # --- ERROR: Single Red Combined Label ---
                        gt = true_labels[i]
                        combined_label = f"P:{pred} (GT:{gt})"
                        ax.text(x, y, z, combined_label, color='#d62728', 
                                fontsize=8, fontweight='bold', alpha=0.9)
                    else:
                        # --- CORRECT: Standard Label ---
                        ax.text(x, y, z, pred, color='black', fontsize=8, alpha=0.7)

        # 6. Formatting
        default_title = f"ID: {frame.embryo_id} | T: {frame.time_idx} | N: {n_cells}{label_suffix}"
        ax.set_title(title if title else default_title, fontsize=12, pad=20)
        
        if n_cells > 0:
            max_range = np.ptp(aligned_coords, axis=0).max() / 2.0
            mid = np.mean(aligned_coords, axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        return ax

    def plot_multi_alignment(self, frames, results, ncols=3, title="Temporal Alignment Sequence"):
        """Faceted version of plot_alignment."""
        n_plots = len(results)
        nrows = int(np.ceil(n_plots / ncols))
        
        fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        for i, (frame, result) in enumerate(zip(frames, results)):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
            self.plot_alignment(frame, result, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _generate_ellipsoid(self, center, cov, scale=1.0):
        """Generates coordinates for a 3D ellipsoid based on a covariance matrix."""
        vals, vecs = np.linalg.eigh(cov)
        vals = 2.0 * scale * np.sqrt(np.maximum(vals, 0))
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        ellipsoid = np.stack((x, y, z), axis=-1) @ (vecs @ np.diag(vals)).T + center
        return ellipsoid[:,:,0], ellipsoid[:,:,1], ellipsoid[:,:,2]
    
# Benchmark visualization

def plot_sweep_dashboard(df, bin_size=5):
    """
    Unified 2x2 dashboard for alignment performance.
    
    Parameters:
    - df: The result dataframe from the BenchmarkingSuite.
    - bin_size: The temporal width (in minutes) for the checkpoint plots.
    """
    df = df.copy()
    sns.set_style("whitegrid")
    
    # 1. Data Preparation
    # Reconstruct 'n_correct' for the cumulative calculations
    df['n_correct'] = (df['frame_accuracy'] * df['N_valid']).round()
    df['time_bin'] = (df['canonical_time'] // bin_size) * bin_size
    
    # Initialize the 2x2 Figure
    # We disable sharex because the top-right plot is continuous (raw time)
    # while the others are binned (categorical time).
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    axes_flat = axes.flatten()

    # --- PLOT 1: Binned Frame Accuracy (Top-Left) ---
    sns.pointplot(
        data=df, x='time_bin', y='frame_accuracy', hue='config_name',
        join=False, dodge=0.4, errorbar=('ci', 95), capsize=0.1, 
        markers=["o", "s", "D", "^"], ax=axes_flat[0]
    )
    axes_flat[0].set_title(f"Frame Accuracy Checkpoints (±95% CI, {bin_size}m bins)", fontsize=13, weight='bold')
    axes_flat[0].set_ylabel("Mean Accuracy")
    axes_flat[0].set_ylim(-0.05, 1.05)

    # --- PLOT 2: Cumulative Cell Identification (Top-Right) ---
    # Aggregate correct counts per minute across all embryos
    ts = df.groupby(['canonical_time', 'config_name']).agg(
        min_correct=('n_correct', 'sum'), 
        min_total=('N_valid', 'sum')
    ).reset_index()
    
    # Ensure strict sorting for cumsum
    ts = ts.sort_values(['config_name', 'canonical_time']).reset_index(drop=True)
    
    # Running sums calculated strictly within each config group
    ts['cum_correct'] = ts.groupby('config_name')['min_correct'].cumsum()
    ts['cum_total'] = ts.groupby('config_name')['min_total'].cumsum()
    ts['running_acc'] = ts['cum_correct'] / ts['cum_total']
    
    sns.lineplot(
        data=ts, x='canonical_time', y='running_acc', 
        hue='config_name', linewidth=2.5, ax=axes_flat[1]
    )
    axes_flat[1].set_title("Cumulative Lineage Recovery (Running Sum)", fontsize=13, weight='bold')
    axes_flat[1].set_ylabel("Total Correct / Total Seen")
    axes_flat[1].set_ylim(ts['running_acc'].min() - 0.05, 1.02)

    # --- PLOT 3: Alignment Cost (Bottom-Left) ---
    sns.pointplot(
        data=df, x='time_bin', y='total_mahalanobis_cost', hue='config_name',
        join=False, dodge=0.4, errorbar=('ci', 95), capsize=0.1, 
        markers=["o", "s", "D", "^"], ax=axes_flat[2]
    )
    axes_flat[2].set_title("Mahalanobis Alignment Cost ($D^2$)", fontsize=13, weight='bold')
    axes_flat[2].set_ylabel("Mean Cost")

    # --- PLOT 4: Computational Load (Bottom-Right) ---
    sns.pointplot(
        data=df, x='time_bin', y='runtime_sec', hue='config_name',
        estimator="sum", join=False, dodge=0.4, errorbar=('ci', 95), 
        capsize=0.1, markers=["o", "s", "D", "^"], ax=axes_flat[3]
    )
    axes_flat[3].set_title(f"Total Computational Load (Sum per {bin_size}m bin)", fontsize=13, weight='bold')
    axes_flat[3].set_ylabel("Total Execution Time (s)")

    # --- FINAL FORMATTING ---
    for i, ax in enumerate(axes_flat):
        # Remove individual legends to avoid clutter
        if ax.get_legend():
            ax.get_legend().remove()
        
        # Clean X-Axis Ticks: Label only every 20 MPF for binned plots
        if i in [0, 2, 3]:
            all_bins = sorted(df['time_bin'].unique())
            # Map tick positions to labels, showing only multiples of 20
            labels = [int(v) if v % 20 == 0 else "" for v in all_bins]
            ax.set_xticklabels(labels)
        
        ax.set_xlabel("Minutes Post-Fertilization (MPF)")

    # Create one global unified legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='center right', title="Pipeline Configs", 
        bbox_to_anchor=(1.12, 0.5), frameon=True, shadow=True, fontsize=11
    )

    plt.suptitle("Engine Performance Sweep: Baseline vs. Topological Consensus", 
                 fontsize=18, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_sweep_dashboard(df, bin_size=5):
    """
    Unified 2x2 dashboard with specifically positioned color-coded annotations.
    - Top Row: Lower Right
    - Bottom Row: Upper Left
    """
    df = df.copy()
    sns.set_style("whitegrid")
    
    # 1. Data Preparation
    df['n_correct'] = (df['frame_accuracy'] * df['N_valid']).round()
    df['time_bin'] = (df['canonical_time'] // bin_size) * bin_size
    configs = list(df['config_name'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    axes_flat = axes.flatten()

    # --- HELPER: Position-Aware Annotation Logic ---
    def add_stat_box(ax, config_label, color, idx, text, position='lower right'):
        if position == 'lower right':
            # Stack upwards from bottom right
            x_pos, y_pos = 0.98, 0.05 + (idx * 0.06)
            ha = 'right'
        else:
            # Stack downwards from top left
            x_pos, y_pos = 0.02, 0.95 - (idx * 0.06)
            ha = 'left'
            
        ax.text(
            x_pos, y_pos, 
            text,
            transform=ax.transAxes,
            fontsize=10,
            color=color,
            weight='bold',
            horizontalalignment=ha,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
        )

    # --- PLOT 1: Binned Frame Accuracy (Top-Left) -> Lower Right ---
    sns.pointplot(
        data=df, x='time_bin', y='frame_accuracy', hue='config_name',
        join=False, dodge=0.4, errorbar=('ci', 95), capsize=0.1, 
        markers=["o", "s", "D", "^"], ax=axes_flat[0]
    )
    axes_flat[0].set_title(f"Frame Accuracy (±95% CI, {bin_size}m bins)", fontsize=13, weight='bold')
    axes_flat[0].set_ylim(-0.05, 1.05)
    
    for i, cfg in enumerate(configs):
        avg_f_acc = df[df['config_name'] == cfg]['frame_accuracy'].mean()
        add_stat_box(axes_flat[0], cfg, sns.color_palette()[i], i, f"{cfg}: {avg_f_acc:.1%} Avg", 'lower right')

    # --- PLOT 2: Cumulative Cell Identification (Top-Right) -> Lower Right ---
    ts = df.groupby(['canonical_time', 'config_name']).agg(
        min_correct=('n_correct', 'sum'), min_total=('N_valid', 'sum')
    ).reset_index()
    ts = ts.sort_values(['config_name', 'canonical_time']).reset_index(drop=True)
    ts['cum_correct'] = ts.groupby('config_name')['min_correct'].cumsum()
    ts['cum_total'] = ts.groupby('config_name')['min_total'].cumsum()
    ts['running_acc'] = ts['cum_correct'] / ts['cum_total']
    
    sns.lineplot(data=ts, x='canonical_time', y='running_acc', hue='config_name', linewidth=2.5, ax=axes_flat[1])
    axes_flat[1].set_title("Total Cumulative Cell Accuracy", fontsize=13, weight='bold')
    
    for i, cfg in enumerate(configs):
        final_row = ts[ts['config_name'] == cfg].iloc[-1]
        total_cell_acc = final_row['cum_correct'] / final_row['cum_total']
        add_stat_box(axes_flat[1], cfg, sns.color_palette()[i], i, f"{cfg}: {total_cell_acc:.1%} Total", 'lower right')

    # --- PLOT 3: Alignment Cost (Bottom-Left) -> Upper Left ---
    sns.pointplot(
        data=df, x='time_bin', y='total_mahalanobis_cost', hue='config_name',
        join=False, dodge=0.4, errorbar=('ci', 95), capsize=0.1, 
        markers=["o", "s", "D", "^"], ax=axes_flat[2]
    )
    axes_flat[2].set_title("Mahalanobis Alignment Cost ($D^2$)", fontsize=13, weight='bold')
    
    for i, cfg in enumerate(configs):
        cfg_df = df[df['config_name'] == cfg]
        avg_cost = cfg_df['total_mahalanobis_cost'].mean()
        total_cost = cfg_df['total_mahalanobis_cost'].sum()
        add_stat_box(axes_flat[2], cfg, sns.color_palette()[i], i, f"{cfg}: Avg {avg_cost:.1f} | Tot {total_cost:,.0f}", 'upper left')

    # --- PLOT 4: Computational Load (Bottom-Right) -> Upper Left ---
    sns.pointplot(
        data=df, x='time_bin', y='runtime_sec', hue='config_name',
        estimator="sum", join=False, dodge=0.4, errorbar=('ci', 95), 
        capsize=0.1, markers=["o", "s", "D", "^"], ax=axes_flat[3]
    )
    axes_flat[3].set_title(f"Total Computational Load (Sum per {bin_size}m bin)", fontsize=13, weight='bold')
    
    for i, cfg in enumerate(configs):
        cfg_df = df[df['config_name'] == cfg]
        avg_rt = cfg_df['runtime_sec'].mean()
        total_rt = cfg_df['runtime_sec'].sum()
        add_stat_box(axes_flat[3], cfg, sns.color_palette()[i], i, f"{cfg}: Avg {avg_rt:.2f}s | Tot {total_rt:.1f}s", 'upper left')

    # --- FINAL FORMATTING ---
    for i, ax in enumerate(axes_flat):
        if ax.get_legend(): ax.get_legend().remove()
        
        if i in [0, 2, 3]:
            all_bins = sorted(df['time_bin'].unique())
            labels = [int(v) if v % 20 == 0 else "" for v in all_bins]
            ax.set_xticklabels(labels)
        
        ax.set_xlabel("Minutes Post-Fertilization (MPF)")

    plt.suptitle("Benchmarking Sweep: Accuracy, Lineage Recovery, and Computational Efficiency", 
                 fontsize=18, weight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.close(fig)
    return fig
    


def plot_faceted_embryo_performance(df):
    """
    4x3 Facet Grid with per-embryo stats and a Global Summary block 
    in the bottom right.
    """
    df = df.copy()
    # Ensure columns exist for calculations
    df['slice_match'] = df['slice_match'].fillna(False).astype(bool)
    df['n_correct'] = (df['frame_accuracy'] * df['N_valid']).round()
    
    # 1. Setup FacetGrid (4 rows, 3 columns)
    g = sns.FacetGrid(
        df, col="embryo_id", hue="config_name", col_wrap=3,
        height=4, aspect=1.4, sharex=True, sharey=True
    )
    
    # 2. Internal plotting function
    def plot_with_local_stats(data, color, label, **kwargs):
        # Sort for clean lines
        data = data.sort_values('canonical_time')
        x, y = data['canonical_time'], data['frame_accuracy']
        
        # Plotting - using the explicit plt submodule
        plt.plot(x, y, color=color, alpha=0.3, linewidth=1.5, zorder=1)
        
        alphas = data['slice_match'].map({True: 1.0, False: 0.2}).values
        plt.scatter(x, y, color=color, alpha=alphas, s=35, edgecolors='white', zorder=2)

        # Per-facet local annotation
        ax = plt.gca()
        configs = list(df['config_name'].unique())
        idx = configs.index(label)
        stat_line = f"{label}: {y.mean():.1%} Acc"
        
        ax.text(0.05, 0.05 + (idx * 0.1), stat_line, transform=ax.transAxes,
                fontsize=8, color=color, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Map the logic
    g.map_dataframe(plot_with_local_stats)

    # 3. Calculate Global Summary Stats
    summary_text = "GLOBAL SWEEP SUMMARY\n" + "="*21 + "\n"
    configs = list(df['config_name'].unique())
    
    for cfg in configs:
        c_df = df[df['config_name'] == cfg]
        
        avg_f_acc = c_df['frame_accuracy'].mean()
        cell_acc = c_df['n_correct'].sum() / c_df['N_valid'].sum() if c_df['N_valid'].sum() > 0 else 0
        slice_acc = c_df['slice_match'].mean()
        
        summary_text += (f"\n{cfg.upper()}\n"
                         f"  Avg Frame:  {avg_f_acc:.1%}\n"
                         f"  Total Cell: {cell_acc:.1%}\n"
                         f"  Slice Hit:  {slice_acc:.1%}\n")

    # 4. Place Global Summary Box
    fig = plt.gcf()
    fig.text(
        0.98, 0.02, 
        summary_text,
        fontsize=10,
        family='monospace',
        color='black',
        weight='bold',
        ha='right', 
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.8')
    )

    # 5. Formatting & Cleanup
    g.set_axis_labels("Canonical Time (min)", "Frame Accuracy")
    g.set_titles("Embryo: {col_name}", size=12, weight='bold')
    g.set(ylim=(-0.05, 1.05))
    g.map(lambda **kwargs: plt.axhline(1.0, color='gray', linestyle='--', alpha=0.15))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.close(fig) # Prevent double plotting in notebooks
    
    return fig
