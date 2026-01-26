import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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
                # Reverted to wireframe for better scannability
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
            # 3. Draw Dashed Displacement Lines
            # These connect the aligned centroid to where the atlas thinks it should be
            for i, label in enumerate(inferred_labels):
                if label in self.atlas.means:
                    atlas_mu = self.atlas.means[label]
                    ax.plot([aligned_coords[i, 0], atlas_mu[0]],
                            [aligned_coords[i, 1], atlas_mu[1]],
                            [aligned_coords[i, 2], atlas_mu[2]],
                            color='black', linestyle='--', linewidth=1, alpha=0.5)

            # 4. Plot Centroids
            ax.scatter(aligned_coords[:, 0], aligned_coords[:, 1], aligned_coords[:, 2], 
                       c=colors, s=40, edgecolors='white', linewidth=0.5, alpha=0.9)

            # 5. Annotate early stages (N < 50)
            if n_cells < 50:
                for i, label in enumerate(inferred_labels):
                    ax.text(aligned_coords[i, 0], aligned_coords[i, 1], aligned_coords[i, 2], 
                            label, fontsize=8, fontweight='bold', alpha=0.8)

        # 6. Formatting
        default_title = f"ID: {frame.embryo_id} | T: {frame.time_idx} | N: {n_cells}{label_suffix}"
        ax.set_title(title if title else default_title, fontsize=12, pad=20)
        
        if n_cells > 0:
            max_range = np.ptp(aligned_coords, axis=0).max() / 2.0
            mid = np.mean(aligned_coords, axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        #ax.set_axis_off() 
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