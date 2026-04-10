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

import matplotlib.pyplot as plt
import seaborn as sns

VERSION_PALETTE = {
    "V0.0": "#9b59b6",  # Purple
    "V1.0": "#e74c3c",  # Red
    "V1.1": "#2ecc71",  # Green
    "V2.0": "#f39c12",  # Orange
    "V2.1": "#3498db",  # Blue
    "production_model": "#3498db"
    #"V3.0": "#34495e"   # Dark Gray
}

def plot_embryo_performance(df, palette=VERSION_PALETTE):
    """
    Facet Grid tracking Positional Accuracy over Time.
    - 3-column grid for embryos.
    - Tabular Global Summary block positioned closer to the plots.
    """
    df = df.copy()
    df['set_match'] = df['set_match'].fillna(0.0).astype(bool)
    df['n_correct'] = (df['positional_accuracy'] * df['num_gt_cells']).round()
    
    x_col = 'canonical_time' 
    y_col = 'positional_accuracy'
    
    # 1. Setup FacetGrid (3 columns for 11 embryos)
    g = sns.FacetGrid(
        df, col="embryo_id", hue="config_name", col_wrap=3,
        height=3.5, aspect=1.4, sharex=True, sharey=True, palette=palette
    )
    
    # 2. Internal plotting function
    def plot_with_local_stats(data, color, label, **kwargs):
        data = data.sort_values(x_col)
        x, y = data[x_col], data[y_col]
        plt.plot(x, y, color=color, alpha=0.3, linewidth=1.5, zorder=1)
        
        alphas = data['set_match'].map({True: 1.0, False: 0.2}).values
        plt.scatter(x, y, color=color, alpha=alphas, s=35, edgecolors='white', zorder=2)

        ax = plt.gca()
        configs = list(df['config_name'].unique())
        idx = configs.index(label)
        stat_line = f"{label}: {y.mean():.1%} Acc"
        
        ax.text(0.05, 0.05 + (idx * 0.1), stat_line, transform=ax.transAxes,
                fontsize=8, color=color, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    g.map_dataframe(plot_with_local_stats)

    # 3. Create Tabular Global Summary
    configs = sorted(df['config_name'].unique())
    
    header = "GLOBAL SWEEP SUMMARY\n" + "="*70 + "\n"
    table_header = f"{'CONFIG VERSION':<25} | {'AVG FRAME':>12} | {'TOTAL CELL':>12} | {'SET MATCH':>12}\n"
    divider = "-" * len(table_header) + "\n"
    
    summary_text = header + table_header + divider
    
    for cfg in configs:
        c_df = df[df['config_name'] == cfg]
        avg_f_acc = c_df[y_col].mean()
        total_gt = c_df['num_gt_cells'].sum()
        cell_acc = c_df['n_correct'].sum() / total_gt if total_gt > 0 else 0
        set_match_rate = c_df['set_match'].mean()
        
        row = (f"{cfg.upper()[:24]:<25} | "
               f"{avg_f_acc:>12.1%} | "
               f"{cell_acc:>12.1%} | "
               f"{set_match_rate:>12.1%}\n")
        summary_text += row

    # 4. Place Global Summary Box
    # Adjusted y-coordinate (0.08) and va='top' to move it closer
    fig = plt.gcf()
    fig.text(
        0.5, 0.08, 
        summary_text,
        fontsize=10,
        family='monospace',
        color='black',
        weight='bold',
        ha='center', 
        va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=1.0')
    )

    # 5. Formatting & Cleanup
    g.set_axis_labels("Canonical Time", "Frame Accuracy")
    g.set_titles("Embryo: {col_name}", size=12, weight='bold')
    g.set(ylim=(-0.05, 1.05))
    g.map(lambda **kwargs: plt.axhline(1.0, color='gray', linestyle='--', alpha=0.15))
    
    # rect=[left, bottom, right, top]
    # We increase the bottom margin slightly (0.12) to clear the space for the box
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])
    plt.close(fig) 
    
    return fig
    
    
def plot_binned_accuracy(df, bin_size=5, palette = VERSION_PALETTE):
    """
    Plots binned Positional Accuracy over Canonical Time across configurations.
    Features a consolidated Global Mean Accuracy summary box.
    """
    df = df.copy()
    sns.set_style("whitegrid")
    
    # 1. Data Preparation
    df['time_bin'] = (df['canonical_time'] // bin_size) * bin_size
    
    # 2. Plotting
    plt.figure(figsize=(8, 8))
    
    # Using lineplot with error bars
    ax = sns.lineplot(
        data=df, 
        x='time_bin', 
        y='positional_accuracy', 
        hue='config_name',
        marker='o',
        markersize=8,
        linewidth=2.5,
        errorbar=('ci', 95),
        err_style="bars",
        err_kws={'capsize': 5},
        palette=palette
    )

    # 3. Build Global Summary Box
    configs = df['config_name'].unique()
    summary_parts = ["GLOBAL MEAN ACCURACY", "="*21]
    
    for cfg in configs:
        c_df = df[df['config_name'] == cfg]
        global_mean = c_df['positional_accuracy'].mean()
        
        # Formatted with padding for alignment: CONFIG_NAME: 00.0%
        summary_parts.append(f"{cfg.upper():<12}: {global_mean:>6.1%}")

    # 4. Final Formatting
    #plt.title(f"Binned Performance Over Time ({bin_size}m intervals)", fontsize=14, weight='bold')
    plt.xlabel(f"Canonical Time ({bin_size}m bins)", fontsize=12)
    plt.ylabel("Frame Accuracy", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.2)
    
    # Draw the summary box in the bottom right
    summary_text = "\n".join(summary_parts)
    plt.gca().text(
        0.98, 0.02, 
        summary_text,
        transform=ax.transAxes,
        fontsize=10, 
        family='monospace', 
        weight='bold',
        ha='right', 
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.8')
    )

    plt.tight_layout()
    
# HTML Report stuff
def plot_optimization_landscape(slice_landscape, slice_id=None):
    """
    Plots the Coarse Roll Angle vs Cost alongside the ICP Convergence
    traces for the selected tournament valleys.
    
    Args:
        slice_landscape (dict): A single dictionary from report['landscape'][slice_id]
        slice_id (int, optional): The ID of the slice for the plot title.
    """
    coarse_history = slice_landscape.get('coarse', [])
    tournament = slice_landscape.get('tournament', [])
    
    if not coarse_history:
        print("No coarse trace history found. Did you run with trace=True?")
        return

    # --- 1. Process Coarse Data ---
    df_coarse = pd.DataFrame(coarse_history)
    df_plus = df_coarse[df_coarse['sign'] == 1.0].sort_values('angle')
    df_minus = df_coarse[df_coarse['sign'] == -1.0].sort_values('angle')

    # --- 2. Setup Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_style("whitegrid")
    
    title_prefix = f"Slice ID {slice_id} | " if slice_id is not None else ""
    fig.suptitle(f"{title_prefix}Alignment Optimization Diagnostics", fontsize=16, fontweight='bold')

    # ==========================================
    # LEFT PANEL: Coarse Roll Landscape
    # ==========================================
    ax1 = axes[0]
    
    # Plot the two orientation sweeps
    ax1.plot(df_plus['angle'], df_plus['cost'], label='+PC1 Axis', color='#3498db', linewidth=2, alpha=0.8)
    ax1.plot(df_minus['angle'], df_minus['cost'], label='-PC1 Axis', color='#e74c3c', linewidth=2, alpha=0.8)

    # Highlight the local minima selected for the tournament
    for i, finalist in enumerate(tournament):
        init_angle = finalist['init_angle']
        rank = finalist['start_rank']
        
        # Look up the exact starting cost of this valley from the coarse history
        start_point = df_coarse[np.isclose(df_coarse['angle'], init_angle, atol=1e-5)]
        if not start_point.empty:
            start_cost = start_point['cost'].min()
            ax1.scatter(init_angle, start_cost, color='#f1c40f', s=250, marker='*', edgecolor='black', zorder=5)
            ax1.text(init_angle, start_cost, f" R{rank}", fontsize=12, fontweight='bold', ha='left', va='bottom', color='black')

    ax1.set_title("Coarse Roll Initialization Landscape", fontsize=14)
    ax1.set_xlabel("Rotation Angle (Degrees)", fontsize=12)
    ax1.set_ylabel("Sinkhorn Approximation Cost", fontsize=12)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))
    ax1.legend(loc='upper right')

    # ==========================================
    # RIGHT PANEL: ICP Convergence Traces
    # ==========================================
    ax2 = axes[1]
    colors = sns.color_palette("Set2", n_colors=len(tournament))

    for i, finalist in enumerate(tournament):
        rank = finalist['start_rank']
        angle = finalist['init_angle']
        icp_history = finalist.get('icp_history', [])

        if icp_history:
            df_icp = pd.DataFrame(icp_history)
            ax2.plot(
                df_icp['iter'], df_icp['cost'], 
                marker='o', markersize=5, 
                label=f'Rank {rank} (Init: {angle:.1f}°)', 
                color=colors[i], linewidth=2.5
            )

    ax2.set_title("ICP Refinement Convergence", fontsize=14)
    ax2.set_xlabel("ICP Iteration", fontsize=12)
    ax2.set_ylabel("Refined Registration Cost", fontsize=12)
    
    # Force integers on the X-axis for iterations
    from matplotlib.ticker import MaxNLocator
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
import plotly.graph_objects as go

def get_plotly_temporal_context(growth_df, observed_n, map_t):
    """
    Generates an interactive Plotly figure comparing the inferred embryo 
    against the empirical population growth curve.
    """
    fig = go.Figure()

    # 1. Plot the 95% Confidence Interval (Shaded Band)
    fig.add_trace(go.Scatter(
        x=pd.concat([growth_df['time_bin'], growth_df['time_bin'][::-1]]),
        y=pd.concat([growth_df['ci_upper'], growth_df['ci_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)', # Light blue
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Biological Variance',
        showlegend=True
    ))

    # 2. Plot the Mean Curve
    fig.add_trace(go.Scatter(
        x=growth_df['time_bin'],
        y=growth_df['mean_n'],
        mode='lines',
        line=dict(color='#2c3e50', width=3),
        name='Empirical Mean',
        hovertemplate='Time: %{x}m<br>Avg Cells: %{y:.1f}<extra></extra>'
    ))

    # 3. Project the Inference Observation (The Red Star)
    fig.add_trace(go.Scatter(
        x=[map_t],
        y=[observed_n],
        mode='markers',
        marker=dict(
            color='#e74c3c', 
            size=18, 
            symbol='star', 
            line=dict(color='black', width=1)
        ),
        name='Current Embryo',
        hovertemplate=f'<b>Observation</b><br>MAP Time: {map_t:.1f}m<br>Observed Cells: {observed_n}<extra></extra>'
    ))

    # 4. Add Crosshairs to pinpoint the star
    fig.add_hline(y=observed_n, line_dash="dot", line_color="#e74c3c", opacity=0.5)
    fig.add_vline(x=map_t, line_dash="dot", line_color="#e74c3c", opacity=0.5)

    # 5. Formatting for HTML
    fig.update_layout(
        title="MAP Time Estimate",
        xaxis_title="Canonical Time (minutes)",
        yaxis_title="Total Number of Cells",
        plot_bgcolor='white',
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Add subtle gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. Ensure the plotting function is defined
def get_plotly_temporal_context(growth_df, observed_n, map_t):
    """
    Generates an interactive Plotly figure comparing the inferred embryo 
    against the empirical population growth curve.
    """
    fig = go.Figure()

    # 1. Plot the 95% Confidence Interval (Shaded Band)
    fig.add_trace(go.Scatter(
        x=pd.concat([growth_df['time_bin'], growth_df['time_bin'][::-1]]),
        y=pd.concat([growth_df['ci_upper'], growth_df['ci_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Biological Variance',
        showlegend=True
    ))

    # 2. Plot the Mean Curve
    fig.add_trace(go.Scatter(
        x=growth_df['time_bin'],
        y=growth_df['mean_n'],
        mode='lines',
        line=dict(color='#2c3e50', width=3),
        name='Empirical Mean',
        hovertemplate='Time: %{x}m<br>Avg Cells: %{y:.1f}<extra></extra>'
    ))

    # 3. Project the Inference Observation (The Red Star)
    fig.add_trace(go.Scatter(
        x=[map_t],
        y=[observed_n],
        mode='markers',
        marker=dict(
            color='#e74c3c', 
            size=18, 
            symbol='star', 
            line=dict(color='black', width=1)
        ),
        name='Current Embryo',
        hovertemplate=f'<b>Observation</b><br>MAP Time: {map_t:.1f}m<br>Observed Cells: {observed_n}<extra></extra>'
    ))

    # 4. Add Crosshairs
    fig.add_hline(y=observed_n, line_dash="dot", line_color="#e74c3c", opacity=0.5)
    fig.add_vline(x=map_t, line_dash="dot", line_color="#e74c3c", opacity=0.5)

    # 5. Formatting
    fig.update_layout(
        title="Embryogenesis Curve vs. MAP Time Estimation",
        xaxis_title="Canonical Time (minutes)",
        yaxis_title="Total Number of Cells",
        plot_bgcolor='white',
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig


import plotly.graph_objects as go
import numpy as np
from scipy.linalg import eigh

def plot_inference_alignment_interactive(result: dict, title=None):
    """
    Interactive Plotly visualizer for 3D alignment.
    Renders the Atlas as solid, translucent 3D ellipsoids and the 
    observed cells as solid spheres, connected by residual lines.
    Features a physical plotting stage and togglable cell labels.
    """
    predicted_labels = result.get('labels', [])
    aligned_coords = result.get('coords', np.array([]))
    ref_frame = result.get('ref_frame', None)
    
    if len(predicted_labels) == 0 or ref_frame is None:
        print("Alignment results missing or 'ref_frame' not returned.")
        return

    # 1. Setup Colors (Matching your lineage palette)
    lineage_colors = {
        'AB': '#1f77b4', 'MS': '#ff7f0e', 'E': '#2ca02c', 
        'C': '#d62728', 'D': '#9467bd', 'P': '#8c564b'
    }
    def get_color(name):
        for key, hex_val in lineage_colors.items():
            if str(name).startswith(key): return hex_val
        return '#7f7f7f' # Gray

    fig = go.Figure()
    atlas_lookup = {}

    # ==========================================
    # 2. Add Atlas Ellipsoids (Solid 3D Meshes)
    # ==========================================
    for label, mu, cov in zip(ref_frame.labels, ref_frame.means, ref_frame.covs):
        color = get_color(label)
        atlas_lookup[label] = mu
        
        # Ensure covariance matrix format
        v = np.atleast_1d(cov)
        cov_matrix = np.eye(3) * v[0] if v.shape == (1,) else (np.diag(v) if v.ndim == 1 else v)
        
        # Calculate eigenvalues/vectors
        vals, vecs = eigh(cov_matrix)
        vals = 2.0 * np.sqrt(np.maximum(vals, 0)) # 2.0 scale factor (95% bounds)
        
        # Generate spherical coordinates
        u = np.linspace(0, 2 * np.pi, 20)
        v_angle = np.linspace(0, np.pi, 15)
        x = np.outer(np.cos(u), np.sin(v_angle))
        y = np.outer(np.sin(u), np.sin(v_angle))
        z = np.outer(np.ones_like(u), np.cos(v_angle))
        
        # Transform sphere to oriented ellipsoid
        ellipsoid = np.stack((x, y, z), axis=-1) @ (vecs @ np.diag(vals)).T + mu
        
        # Add as a solid translucent convex hull
        fig.add_trace(go.Mesh3d(
            x=ellipsoid[:,:,0].flatten(), 
            y=ellipsoid[:,:,1].flatten(), 
            z=ellipsoid[:,:,2].flatten(),
            alphahull=0,     # Creates a tight wrap around the points
            color=color,
            opacity=0.15,    # Glass-like transparency
            name=f"Atlas {label}",
            hoverinfo='name',
            showlegend=False
        ))

    # ==========================================
    # 3. Add Displacement Lines (Highly Optimized)
    # ==========================================
    # We use a single trace with 'None' breaks to prevent HTML bloat
    line_x, line_y, line_z = [], [], []
    for label, coord in zip(predicted_labels, aligned_coords):
        if label in atlas_lookup:
            mu = atlas_lookup[label]
            line_x.extend([coord[0], mu[0], None])
            line_y.extend([coord[1], mu[1], None])
            line_z.extend([coord[2], mu[2], None])

    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color='black', width=3, dash='dot'),
        opacity=0.4,
        showlegend=False,
        hoverinfo='none',
        name="Residuals"
    ))

    # ==========================================
    # 4. Add Aligned Experimental Points
    # ==========================================
    colors_inf = [get_color(l) for l in predicted_labels]
    
    # Extract confidence scores if available for hover data
    hovers = []
    if 'diagnostics' in result:
        conf_scores = result['diagnostics']['confidence_score'].values
        for l, c in zip(predicted_labels, conf_scores):
            hovers.append(f"<b>{l}</b><br>Confidence: {c:.1%}")
    else:
        hovers = [f"<b>{l}</b>" for l in predicted_labels]

    fig.add_trace(go.Scatter3d(
        x=aligned_coords[:, 0], 
        y=aligned_coords[:, 1], 
        z=aligned_coords[:, 2],
        mode='markers+text', # Labels ON by default
        text=predicted_labels,
        textposition="top center",
        textfont=dict(size=10, color='black', weight='bold'),
        marker=dict(
            size=7, 
            color=colors_inf, 
            line=dict(color='black', width=1.5), 
            symbol='circle'
        ),
        name='Inference Data',
        hovertext=hovers,
        hoverinfo='text',
        showlegend=False
    ))

    # Calculate the exact index of the Inference Data trace for the toggle button
    # It is added after N meshes + 1 residual line trace
    inference_trace_idx = len(ref_frame.labels) + 1

    # ==========================================
    # 5. Layout, Stage, and Interactivity
    # ==========================================
    if title is None:
        t_map = result.get('map_time', np.nan)
        cost = result.get('cost', 0.0)
        title = f"3D Alignment Map | N={len(predicted_labels)} | t_MAP={t_map:.1f} | Cost={cost:.1f}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='black')),
        
        # The Plotting Stage (Colored background panels with crisp grid lines)
        scene=dict(
            xaxis=dict(title='X (μm)', showbackground=True, backgroundcolor="#e5e5e5", gridcolor="white", showspikes=True),
            yaxis=dict(title='Y (μm)', showbackground=True, backgroundcolor="#e5e5e5", gridcolor="white", showspikes=True),
            zaxis=dict(title='Z (μm)', showbackground=True, backgroundcolor="#e5e5e5", gridcolor="white", showspikes=True),
            aspectmode='data' 
        ),
        
        # UI Buttons (The Label Toggle)
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        label="Labels ON",
                        method="restyle",
                        args=[{"mode": "markers+text"}, [inference_trace_idx]]
                    ),
                    dict(
                        label="Labels OFF",
                        method="restyle",
                        args=[{"mode": "markers"}, [inference_trace_idx]]
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.08,
                yanchor="top",
                bgcolor="white",
                bordercolor="gray"
            ),
        ],
        margin=dict(l=10, r=10, b=10, t=80),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    #fig.show()
    return fig



def plot_spatial_confidence_interactive(result: dict, title=None):
    """
    Interactive Plotly visualizer for Spatial Confidence.
    Plots the aligned experimental cells colored by the Oracle's 
    predicted confidence score using a Red-to-Green heatmap.
    Includes a 10% bounding box padding to prevent wall-clipping.
    """
    predicted_labels = result.get('labels', [])
    aligned_coords = result.get('coords', np.array([]))
    
    if len(predicted_labels) == 0:
        print("Alignment results missing.")
        return
        
    if 'diagnostics' not in result or 'confidence_score' not in result['diagnostics'].columns:
        print("Confidence scores not found. Did you run the result through the Oracle?")
        return
        
    conf_scores = result['diagnostics']['confidence_score'].values
    fig = go.Figure()

    hovers = [f"<b>{l}</b><br>Confidence: {c:.1%}" for l, c in zip(predicted_labels, conf_scores)]

    fig.add_trace(go.Scatter3d(
        x=aligned_coords[:, 0], 
        y=aligned_coords[:, 1], 
        z=aligned_coords[:, 2],
        mode='markers+text',
        text=predicted_labels,
        textposition="top center",
        textfont=dict(size=10, color='black', weight='bold'),
        marker=dict(
            size=10, 
            color=conf_scores, 
            colorscale='RdYlGn', 
            cmin=0.0, cmax=1.0,
            showscale=True,
            colorbar=dict(title="Confidence", tickformat=".0%", thickness=15, len=0.7),
            line=dict(color='black', width=1.5) 
        ),
        name='Confidence Map',
        hovertext=hovers,
        hoverinfo='text',
        showlegend=False
    ))

    # ==========================================
    # Calculate Padding for the 3D Stage Walls
    # ==========================================
    x_min, x_max = aligned_coords[:, 0].min(), aligned_coords[:, 0].max()
    y_min, y_max = aligned_coords[:, 1].min(), aligned_coords[:, 1].max()
    z_min, z_max = aligned_coords[:, 2].min(), aligned_coords[:, 2].max()

    # Add 10% padding to each axis
    pad_x = (x_max - x_min) * 0.10
    pad_y = (y_max - y_min) * 0.10
    pad_z = (z_max - z_min) * 0.10

    if title is None:
        mean_conf = result.get('mean_confidence', np.mean(conf_scores))
        title = f"Spatial Confidence Map | N={len(predicted_labels)} | Mean Conf={mean_conf:.1%}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='black')),
        
        scene=dict(
            # Apply the padded ranges to push the walls outward
            xaxis=dict(title='X (μm)', range=[x_min - pad_x, x_max + pad_x], showbackground=True, backgroundcolor="#e5e5e5", gridcolor="white", showspikes=True),
            yaxis=dict(title='Y (μm)', range=[y_min - pad_y, y_max + pad_y], showbackground=True, backgroundcolor="#e5e5e5", gridcolor="white", showspikes=True),
            zaxis=dict(title='Z (μm)', range=[z_min - pad_z, z_max + pad_z], showbackground=True, backgroundcolor="#e5e5e5", gridcolor="white", showspikes=True),
            aspectmode='data' 
        ),
        
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(label="Labels ON", method="restyle", args=[{"mode": "markers+text"}, [0]]),
                    dict(label="Labels OFF", method="restyle", args=[{"mode": "markers"}, [0]])
                ]),
                pad={"r": 10, "t": 10},
                showactive=True, x=0.01, xanchor="left", y=1.08, yanchor="top",
                bgcolor="white", bordercolor="gray"
            ),
        ],
        margin=dict(l=10, r=10, b=10, t=80),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    #fig.show()
    return fig