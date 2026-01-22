import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class StaticGaussianAtlas:
    """
    Manages reference coordinate data derived from labeled embryos.
    
    This class loads a Gaussian Atlas CSV, filters by data quality, 
    and pre-computes inverse covariance matrices for fast Mahalanobis 
    distance calculations during alignment.
    """
    
    def __init__(self, atlas_path: str, min_samples: int = 5):
        """
        Initializes the Atlas from a CSV file.
        
        Args:
            atlas_path (str): Path to the atlas CSV file.
            min_samples (int): Minimum number of samples required to include a cell type.
        """
        self.means: Dict[str, np.ndarray] = {}
        self.inv_covs: Dict[str, np.ndarray] = {}
        
        df = pd.read_csv(atlas_path)
        self._build_lookup(df, min_samples)
    
    def _build_lookup(self, df: pd.DataFrame, min_samples: int) -> None:
        """
        Internal helper to populate lookup dictionaries.
        """
        valid_df = df[df['n_samples'] >= min_samples]
        
        for _, row in valid_df.iterrows():
            name = row['cell_name']
            self.means[name] = np.array([row['mu_x'], row['mu_y'], row['mu_z']])
            cov = self._assemble_cov(row)
            self.inv_covs[name] = np.linalg.inv(cov + 1e-6 * np.eye(3))
            
    def _assemble_cov(self, row) -> np.ndarray:
        """
        Constructs a symmetric 3x3 covariance matrix from row values.
        """
        return np.array([
            [row['cov_xx'], row['cov_xy'], row['cov_xz']],
            [row['cov_xy'], row['cov_yy'], row['cov_yz']],
            [row['cov_xz'], row['cov_yz'], row['cov_zz']],
        ])
        
    def get_params(self, labels: List[str]) -> Tuple[np.ndarray, list[np.ndarray]]:
        """
        Retrieves means and inverse covariances for a specific set of cell labels.
        
        Args:
            labels (List[str]): List of cell names to retrieve.
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: 
                - (N, 3) matrix of means
                - List of (3, 3) inverse covariance matrices
        """
        mus = []
        invs = []
        for l in labels:
            if l not in self.means:
                raise KeyError(f"Critical Error: Cell'{l}' missing from Atlas.")
            mus.append(self.means[l])
            invs.append(self.inv_covs[l])
        return np.vstack(mus), invs
    
