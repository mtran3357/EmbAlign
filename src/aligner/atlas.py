import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
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

class GPTimeAtlas:
    """Time resolved atlas from GP."""
    def __init__(self, atlas_csv_path, n_prior_csv_path = None):
        self.df = pd.read_csv(atlas_csv_path)
        self.n_prior= pd.read_csv(n_prior_csv_path) if n_prior_csv_path else None
        
        # time grid and labels
        self.time_grid = np.sort(self.df['canonical_time'].unique())
        self.labels = self.df['cell_name'].unique()
        
        # interpolators
        self._interps ={}
        self._build_interpolators()
        
    def _build_interpolators(self):
        """Build standard interpolators without pre-baked tapering."""
        for label in self.labels:
            sub = self.df[self.df['cell_name'] == label].sort_values('canonical_time')
            if len(sub) < 2: continue
            
            t = sub['canonical_time'].values
            coords = sub[['mu_x', 'mu_y', 'mu_z']].values
            base_sigma2 = sub['sigma2_label'].values + sub['sigma2_gp'].values
            
            self._interps[label] = {
                'mu': interp1d(t, coords, axis=0, kind='linear', 
                               fill_value=(coords[0], coords[-1]), bounds_error=False),
                'sigma2': interp1d(t, base_sigma2, kind='linear', 
                                   fill_value=(base_sigma2[0], base_sigma2[-1]), bounds_error=False),
                't_range': (t.min(), t.max())
            }

    def get_state(self, time, active_labels=None):
        if active_labels is None:
            active_labels = self.get_valid_labels(time)
        
        means, variances, labels = [], [], []
        for lbl in active_labels:
            if lbl in self._interps:
                data = self._interps[lbl]
                t_min, t_max = data['t_range']
                
                # 1. Get base values from interpolation
                mu = data['mu'](time)
                sig2 = data['sigma2'](time)
                
                # 2. Dynamic Tapering: Increase variance based on distance to 'living' window
                # If time is inside (t_min, t_max), dist_out is 0.
                dist_out = max(0, t_min - time, time - t_max)
                
                # Apply an exponential or linear 'swell'
                # 10.0 is the Ghost Cap; 2.0 is the growth rate
                swelled_sig2 = sig2 + (dist_out * 0.18) 
                
                # Cap the variance so the ghost doesn't become infinite
                final_sig2 = min(swelled_sig2, 3.0)
                
                means.append(mu)
                variances.append(final_sig2)
                labels.append(lbl)
                
        return {'labels': labels, 'means': np.array(means), 'variances': np.array(variances)}
        
    def p_n_given_t(self, n_cells, time):
        if self.n_prior is None:
            return 1.0
        
        # find nearest existing observed time
        idx = np.abs(self.n_prior['canonical_time'] - time).argmin()
        closest_time = self.n_prior.iloc[idx]['canonical_time']
        # extract prob of n
        prob_row = self.n_prior[(self.n_prior['canonical_time'] == closest_time) & 
                                (self.n_prior['N'] == n_cells)]
        
        return prob_row['P_N_given_t'].values[0] if not prob_row.empty else 1e-6
    
    def get_valid_labels(self, time):
        """
        Returns all labels that are biologically active OR in a ghost state.
        We add a 2.0 minute buffer to ensure the 'bridge' is visible.
        """
        valid_at_time = []
        buffer = 2.0  # Allows ghosts to persist for 2 minutes post-division
        
        for lbl, data in self._interps.items():
            t_min, t_max = data['t_range']
            # The new logic: Check range PLUS the ghost buffer
            if (t_min - buffer) <= time <= (t_max + buffer):
                valid_at_time.append(lbl)
                    
        return valid_at_time
        