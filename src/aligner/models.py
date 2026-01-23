import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

class EmbryoFrame:
    """
    Represents a single time-point of an embryo's developmental trajectory,
    filtered for valid experimental observations.
    """
    
    def __init__(self, coords: np.ndarray, embryo_id: str, time_idx:int, metadata: pd.DataFrame = None):
        """
        Internal constructor: use from_dataframe()  or from_matrix() for general_use
        """
        self.embryo_id = embryo_id
        self.time_idx = time_idx
        self.coords = coords
        self.valid_df = metadata
        
        # State placeholders
        self.normalized_coords = None
        self.scale_factor = 1.0
        self.pc1_axis = None
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, embryo_id: int, time_idx: int):
        """
        Extract a specific timepoint for a specific embryo.
        """
        subset = df[
            (df['embryo_id'].astype(str) == str(embryo_id)) & 
            (df['time_idx'].astype(int) == int(time_idx))
        ]
        # Filter for valid observations
        valid_subset = subset[subset['valid'].astype(int) == 1].copy()
            
        if valid_subset.empty:
            raise ValueError(f"No valid data found for Embryo {embryo_id} at T={time_idx}")
        coords = valid_subset[['x_um', 'y_um', 'z_um']].values
        return cls(coords, str(embryo_id), time_idx, metadata=valid_subset)
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, embryo_id: str = "Test_Inference",
                    px_xy: float = 1.0, px_z: float = 1.0, mirror_lr: bool = False):
        """Directly handles raw X,Y,Z matrices encountered during inference."""
        
        # Convert pixels to microns
        phys_coords = matrix.copy(). astype(float)
        phys_coords[:, 0:2] *= px_xy
        phys_coords[:, 2] *= px_z
        
        # Mirror if requested
        if mirror_lr:
            phys_coords[:, 0] *= -1.0
            
        return cls(phys_coords, embryo_id, 0, None)
        
    def __repr__(self)-> str:
        """Returns a string representation for easy debugging."""
        return f"EmbryoFrame(ID={self.embryo_id}, T={self.time_idx}, N={len(self)})"
    
    def __len__(self) -> int:
        """Allows usage of len(frame) to get cell count."""
        return self.coords.shape[0]
     
    def prepare(self) -> None:
        """
        Standardizes the frame by scaling by median distance and centering. Also calculates primary axis for rotation.
        """
        self.scale_factor = self._calculate_median_dist()
        # Normalization
        norm = self.coords / self.scale_factor
        self.normalized_coords = norm - norm.mean(axis=0)
        self.pc1_axis = self._calculate_pc1(self.normalized_coords)
    
    def _calculate_median_dist(self) -> float:
        """
        Helper to find the median pairwise distance between all cells.
        """
        if len(self) < 2: 
            return 1.0
        dists = pdist(self.coords)
        med = np.median(dists)
        #handle zero or non-finite distances
        if med <= 0 or not np.isfinite(med):
            return 1.0
        
        return med
    
    def _calculate_pc1(self, coords: np.ndarray) -> np.ndarray:
        """Calculate first principal axis via SVD."""
        U, S, Vt = np.linalg.svd(coords, full_matrices=False)
        axis = Vt[0]
        norm = np.linalg.norm(axis)
        # Match legacy fallback for degenerate variance
        if norm < 1e-8 or not np.isfinite(norm):
            return np.array([1.0, 0.0, 0.0])
        return axis / norm
    

class ReferenceFrame:
    """
    A biological hypothesis representing a valid cell-type combination.
    
    This class 'inflates' a slice from the SliceAtlas using 3D parameters 
    from the StaticGaussianAtlas to create a schematic embryo for alignment.
    """

    def __init__(self, labels: tuple, atlas: 'StaticGaussianAtlas'):
        self.labels = labels
        self.n_cells = len(labels)
        
        # Pull 3d params from gaussian atlas
        self.means, self.inv_covs, self.covs = atlas.get_params(list(labels))
        
        # 3. Calculate center and PC1
        self.center_of_mass = self.means.mean(axis=0)
        self.centered_means = self.means - self.center_of_mass
        self.pc1_axis = self._calculate_pc1(self.centered_means)
        # Note: Why don't we scale atlas params? Already scaled?
        
    def _calculate_pc1(self, coords: np.ndarray) -> np.ndarray:
        """Calculate first principal axis via SVD."""
        U, S, Vt = np.linalg.svd(coords, full_matrices=False)
        axis = Vt[0]
        norm = np.linalg.norm(axis)
        # Match legacy fallback for degenerate variance
        if norm < 1e-8 or not np.isfinite(norm):
            return np.array([1.0, 0.0, 0.0])
        return axis / norm
    
    def __repr__(self):
        return f"ReferenceFrame(N={self.n_cells}, labels={self.labels[:3]}...)"
        
        