import numpy as np
from scipy.spatial.distance import pdist

class EmbryoFrame:
    """
    Represents a single time-point of an embryo's developmental trajectory.

    This class handles the internal state of experimental cell coordinates,
    providing standardized (centered and scaled) versions of the data for 
    alignment algorithms.

    Attributes:
        embryo_id (str): Unique identifier for the embryo.
        time_idx (int): The developmental time point (e.g., minute or index).
        coords (np.ndarray): The raw (N, 3) coordinate matrix.
        normalized_coords (np.ndarray): The (N, 3) matrix after scaling and centering.
        scale_factor (float): The median pairwise distance used for scaling.
    """
    
    def __init__(self, coords: np.ndarray, embryo_id: str, time_idx: int):
        self.embryo_id = embryo_id
        self.time_idx = time_idx
        self.coords = coords
        
        # Internal state placeholders
        self.normalized_coords = None
        self.scale_factor = 1.0
    
    def __repr__(self)-> str:
        """Returns a string representation for easy debugging."""
        return f"EmbryoFrame(ID={self.embryo_id}, T={self.time_idx}, N={len(self)})"
    
    def __len__(self) -> int:
        """Allows usage of len(frame) to get cell count."""
        return self.coords.shape[0]
     
    def prepare(self) -> None:
        """
        Standardizes the frame by scaling by median distance and centering.
        """
        self.scale_factor = self._calculate_median_dist()
        # Normalization
        norm = self.coords / self.scale_factor
        self.normalized_coords = norm - norm.mean(axis=0)
    
    def _calculate_median_dist(self) -> float:
        """
        Helper to find the median pairwise distance between all cells.
        """
        if len(self) < 2: 
            return 1.0
        return np.median(pdist(self.coords))
    

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
        self.means, self.inv_covs = atlas.get_params(list(labels))
        
        # Pre compute center for coarse scan
        self.center_of_mass = self.means.mean(axis=0)
        self.centered_means = self.means - self.center_of_mass
        
        # Pre compute PC1 for coarse scan
        self.pc1_axis = self._calculate_pc1(self.centered_means)
        
    def _calculate_pc1(self, coords: np.ndarray) -> np.ndarray:
        """
        Calculates the first principal axis via SVD.
        """
        U,S,Vt = np.linalg.svd(coords, full_matrices=False)
        axis = Vt[0]
        return axis / (np.linalg.norm(axis) + 1e-12)
    
    def __repr__(self):
        return f"ReferenceFrame(N={self.n_cells}, labels={self.labels[:3]}...)"
        
        