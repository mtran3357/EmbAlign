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
    
    def __init__(self, atlas_path: str, min_samples: int = 5, reg_eps: float = 1e-6):
        """
        Initializes the Atlas from a CSV file.
        """
        self.means: Dict[str, np.ndarray] = {}
        self.inv_covs: Dict[str, np.ndarray] = {}
        self.covs: Dict[str, np.ndarray] ={}
        self.reg_eps = reg_eps
        
        df = pd.read_csv(atlas_path)
        self._build_lookup(df, min_samples)
    
    def _build_lookup(self, df: pd.DataFrame, min_samples: int) -> None:
        """
        Internal helper to populate lookup dictionaries.
        """
        valid_df = df[df['n_samples'] >= min_samples].copy()
        
        for _, row in valid_df.iterrows():
            name = str(row['cell_name'])
            mu = np.array([row['mu_x'], row['mu_y'], row['mu_z']], dtype=float)
            cov = self._assemble_cov(row)
            # regularize cov mat to ensure invertibility
            cov_reg = cov + self.reg_eps * np.eye(3)
            
            try: 
                inv_cov = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                # pseudo inverse if singular
                inv_cov = np.linalg.pinv(cov_reg)
            
            self.means[name] = mu
            self.covs[name] = cov_reg
            self.inv_covs[name] = inv_cov
            
    def _assemble_cov(self, row) -> np.ndarray:
        """
        Constructs a symmetric 3x3 covariance matrix from row values.
        """
        return np.array([
            [row['cov_xx'], row['cov_xy'], row['cov_xz']],
            [row['cov_xy'], row['cov_yy'], row['cov_yz']],
            [row['cov_xz'], row['cov_yz'], row['cov_zz']],
        ], dtype = float)
        
    def get_params(self, labels: List[str]) -> Tuple[np.ndarray, list[np.ndarray]]:
        """
        Retrieves means and inverse covariances for a specific set of cell labels.
        
        Args:
            labels (List[str]): List of cell names to retrieve.
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: 
                - (N, 3) matrix of means
                - (N, 3, 3) array of inverse covariances
                - (N, 3, 3) array of covariances
        """
        mus = []
        invs = []
        covs = []
        for l in labels:
            if l not in self.means:
                raise KeyError(f"Critical Error: Cell '{l}' missing from Atlas.")
            mus.append(self.means[l])
            invs.append(self.inv_covs[l])
            covs.append(self.covs[l])
        return np.vstack(mus), np.array(invs), np.array(covs)
    
class SliceAtlas:
    """
    Manages the database of valid cell-type combinations (slices).
    """
    
    def __init__(self, slice_csv_path: str):
            self.n_to_ids: Dict[int, List[int]] = {}
            self.id_to_labels: Dict[int, Tuple[str, ...]] = {}
            self._load_slices(slice_csv_path)
    
    def _load_slices(self, path: str) -> None:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            s_id = int(row['slice_id'])
            n = int(row['n_cells_frame'])
            labels = tuple(sorted(str(row["cell_names"]).split(";")))
            if n not in self.n_to_ids:
                self.n_to_ids[n] = []
            self.n_to_ids[n].append(s_id)
            self.id_to_labels[s_id] = labels
    
    def get_candidates(self, n_cells: int) -> List[int]:
        """Returns a list of slice_ids matching the cell count."""
        return self.n_to_ids.get(n_cells, [])

    def get_labels(self, slice_id: int) -> Tuple[str, ...]:
        """Retrieves the biological labels for a specific slice ID."""
        return self.id_to_labels.get(slice_id, ())
