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
        self.pc2_axis = None
        self.pc3_axis = None
        self.singular_values = None
    
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
        axes = self._calculate_pca(self.normalized_coords)
        self.pc1_axis, self.pc2_axis, self.pc3_axis = axes
        #self.pc1_axis = self._calculate_pc1(self.normalized_coords)
        #axes = self._calculate_pca(self.normalized_coords)
    
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
    def _calculate_pca(self, coords: np.ndarray):
        """
        Calculates all three principal axes via SVD.
        Returns: (pc1, pc2, pc3)
        """
        # 1. Coordinates should already be centered at (0,0,0) 
        # if they are 'normalized_coords', but we'll ensure it for PCA
        centered = coords - np.mean(coords, axis=0)
        
        # 2. Compute SVD
        # Vt rows are the principal components (eigenvectors)
        # S contains the singular values (sqrt of eigenvalues)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # 3. Extract and normalize axes
        axes = []
        for i in range(3):
            axis = Vt[i]
            norm = np.linalg.norm(axis)
            
            # Handle degenerate cases (e.g., planar or linear distributions)
            if norm < 1e-8 or not np.isfinite(norm):
                # Fallback to standard basis if an axis is undefined
                fallback = np.zeros(3)
                fallback[i] = 1.0
                axes.append(fallback)
            else:
                axes.append(axis / norm)
                
        self.pc1_axis = axes[0]
        self.pc2_axis = axes[1]
        self.pc3_axis = axes[2]
        
        # Important for your report: Explained Variance
        # Tells us if the embryo is a sphere (S1 ~= S2) or a cigar (S1 >> S2)
        self.singular_values = S
        return axes
    

class ReferenceFrame:
    """
    A biological hypothesis representing a valid cell-type combination.
    
    This class 'inflates' a slice from the SliceAtlas using 3D parameters 
    from the StaticGaussianAtlas to create a schematic embryo for alignment.
    """

    def __init__(self, labels: tuple, atlas: 'StaticGaussianAtlas'):
        self.labels = labels
        self.n_real = len(labels)
        
        # Pull 3d params from gaussian atlas
        self.means, self.inv_covs, self.covs = atlas.get_params(list(labels))
        
        # 3. Calculate center and PC1
        self.center_of_mass = self.means.mean(axis=0)
        self.centered_means = self.means - self.center_of_mass
        self.pc1_axis = self._calculate_pc1(self.centered_means)
        self.pc2_axis = None
        self.pc3_axis = None
        self.singular_values = None
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
    
    def _calculate_pca(self, coords: np.ndarray):
        """
        Calculates all three principal axes via SVD.
        Returns: (pc1, pc2, pc3)
        """
        centered = coords - np.mean(coords, axis=0)
        

        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # 3. Extract and normalize axes
        axes = []
        for i in range(3):
            axis = Vt[i]
            norm = np.linalg.norm(axis)
            
            # Handle degenerate cases (e.g., planar or linear distributions)
            if norm < 1e-8 or not np.isfinite(norm):
                # Fallback to standard basis if an axis is undefined
                fallback = np.zeros(3)
                fallback[i] = 1.0
                axes.append(fallback)
            else:
                axes.append(axis / norm)
                
        self.pc1_axis = axes[0]
        self.pc2_axis = axes[1]
        self.pc3_axis = axes[2]
    
        self.singular_values = S
        return axes
    
    def __len__(self):
        return self.n_real
    
    def __repr__(self):
        return f"ReferenceFrame(N={self.n_cells}, labels={self.labels[:3]}...)"
        
        