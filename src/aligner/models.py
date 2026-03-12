import pandas as pd
import numpy as np
from typing import List, Tuple, Any
from scipy.spatial.distance import pdist

class EmbryoFrame:
    """
    Represents a single time-point of an experimental embryo's developmental trajectory.
    Holds raw coordinates and handles normalization/PCA for alignment.
    """
    
    def __init__(self, coords: np.ndarray, embryo_id: Any, time_idx: int, metadata: pd.DataFrame = None):
        self.embryo_id = str(embryo_id)
        self.time_idx = int(time_idx)
        self.coords = coords
        self.valid_df = metadata
        
        # State placeholders populated by .prepare()
        self.normalized_coords = None
        self.center_of_mass = None
        self.pc1_axis = None
        self.pc2_axis = None
        self.pc3_axis = None

    def __len__(self):
        return len(self.coords)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, embryo_id: Any, time_idx: int):
        """Extracts a specific timepoint for a specific embryo from the main dataset."""
        subset = df[
            (df['embryo_id'].astype(str) == str(embryo_id)) & 
            (df['time_idx'].astype(int) == int(time_idx))
        ]
        
        # Filter for valid observations
        valid_subset = subset[subset['valid'].astype(int) == 1].copy()
            
        if valid_subset.empty:
            raise ValueError(f"No valid data found for Embryo {embryo_id} at t={time_idx}")
            
        coords = valid_subset[['x_um', 'y_um', 'z_um']].values.astype(float)
        return cls(coords=coords, embryo_id=embryo_id, time_idx=time_idx, metadata=valid_subset)

    def prepare(self):
        if self.normalized_coords is not None:
            return
            
        self.center_of_mass = np.mean(self.coords, axis=0)
        centered = self.coords - self.center_of_mass
        
        # Restore the Legacy Scale Normalization
        if len(centered) > 1:
            med_dist = np.median(pdist(centered))
            scale = med_dist if med_dist > 0 else 1.0
        else:
            scale = 1.0
            
        self.normalized_coords = centered / scale
        self.pc1_axis, self.pc2_axis, self.pc3_axis = self._calculate_pca(self.normalized_coords)

    def _calculate_pca(self, centered_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates all three principal axes via SVD."""
        U, S, Vt = np.linalg.svd(centered_coords, full_matrices=False)
        
        axes = []
        for i in range(3):
            axis = Vt[i]
            norm = np.linalg.norm(axis)
            
            # Fallback to standard basis if an axis is undefined (e.g. perfectly planar)
            if norm < 1e-8 or not np.isfinite(norm):
                fallback = np.zeros(3)
                fallback[i] = 1.0
                axes.append(fallback)
            else:
                axes.append(axis / norm)
                
        return axes[0], axes[1], axes[2]


class ReferenceFrame:
    """
    Represents a biological hypothesis (a slice of the atlas). 
    Acts as the target for the EmbryoFrame during the alignment process.
    """
    def __init__(self, labels: List[str], atlas: Any):
        """
        Args:
            labels: List of cell names in this hypothesis.
            atlas: An instance of StaticGaussianAtlas or GPToStaticAdapter.
        """
        self.labels = labels
        self.n_real = len(labels)
        
        # 1. Fetch geometric parameters from the unified atlas interface
        self.means, self.inv_covs, self.covs = atlas.get_params(self.labels)
        
        # 2. Calculate the target Center of Mass
        self.center_of_mass = np.mean(self.means, axis=0)
        
        # 3. Calculate target PCA for the coarse scan
        self.pc1_axis = self._calculate_pc1(self.means - self.center_of_mass)

    def _calculate_pc1(self, centered_means: np.ndarray) -> np.ndarray:
        """Extracts just the primary axis of variance for the reference frame."""
        if len(centered_means) < 2:
            return np.array([1.0, 0.0, 0.0])
            
        U, S, Vt = np.linalg.svd(centered_means, full_matrices=False)
        axis = Vt[0]
        norm = np.linalg.norm(axis)
        
        if norm < 1e-8 or not np.isfinite(norm):
            return np.array([1.0, 0.0, 0.0])
        return axis / norm