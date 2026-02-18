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
    def __init__(self, atlas_csv_path, n_prior_csv_path=None):
        raw_df = pd.read_csv(atlas_csv_path)
        # Average coordinates at duplicate timepoints to prevent interpolation errors
        self.df = raw_df.groupby(['cell_name', 'canonical_time']).mean(numeric_only=True).reset_index()
        self.n_prior = pd.read_csv(n_prior_csv_path) if n_prior_csv_path else None
        self.labels = self.df['cell_name'].unique()
        self._interps = {}
        self._build_interpolators()
        
    def _build_interpolators(self):
        """Build linear interpolators for cell means and variances."""
        for label in self.labels:
            sub = self.df[self.df['cell_name'] == label].sort_values('canonical_time')
            if len(sub) < 2: continue
            t = sub['canonical_time'].values
            coords = sub[['mu_x', 'mu_y', 'mu_z']].values
            base_sigma2 = sub['sigma2_label'].values + sub['sigma2_gp'].values
            self._interps[label] = {
                'mu': interp1d(t, coords, axis=0, kind='linear', fill_value="extrapolate", bounds_error=False),
                'sigma2': interp1d(t, base_sigma2, kind='linear', fill_value="extrapolate", bounds_error=False),
                't_range': (t.min(), t.max())
            }

    def get_state(self, time, active_labels=None):
        """Retrieves coordinates and variances for a specific time."""
        means, variances, labels = [], [], []
        target_labels = active_labels if active_labels is not None else self.labels
        for lbl in target_labels:
            if lbl in self._interps:
                d = self._interps[lbl]
                means.append(d['mu'](time))
                variances.append(d['sigma2'](time))
                labels.append(lbl)
        return {'labels': labels, 'means': np.array(means), 'variances': np.array(variances)}
        
class SliceTimeAtlas:
    """Maps Slice Ids to their valid canonical time windows."""
    def __init__(self, gp_atlas, slice_db, padding: float = 1.0):
        self.gp_atlas = gp_atlas
        self.slice_db = slice_db
        self.padding = padding
        self.slice_windows = {}
        self._map_slices_to_time()
    
    def _map_slices_to_time(self):
        """Calculates the intersection of member cell lifespans for each slice."""
        for s_id, labels in self.slice_db.id_to_labels.items():
            t_starts, t_ends = [], []
            for lbl in labels:
                if lbl in self.gp_atlas._interps:
                    t_min, t_max = self.gp_atlas._interps[lbl]['t_range']
                    t_starts.append(t_min)
                    t_ends.append(t_max)
            if t_starts: 
                self.slice_windows[s_id] = {
                    't_range': (max(t_starts) - self.padding, min(t_ends) + self.padding),
                    'labels': labels
                }
    
    def get_temporal_state(self, s_id, time_offset=0.5):
        """Fetches the biological geometry at a specific point in the slice window."""
        win = self.slice_windows.get(s_id)
        if not win: return None
        t_start, t_end = win['t_range']
        target_t = t_start + (t_end - t_start) * time_offset
        return self.gp_atlas.get_state(target_t, active_labels=win['labels'])
    
class GPToStaticAdapter:
    """Makes a dynamic GP state look likea. Static GaussianAtlas for legacy engine use."""
    def __init__(self, labels, means, variances):
        self.means = {l: m for l, m in zip(labels, means)}
        self.covs = {l: np.eye(3) * v for l, v in zip(labels, variances)}
        # Pre-compute inverse for Mahalanobis scoring efficiency
        self.inv_covs = {l: np.eye(3) * (1.0 / v) for l, v in zip(labels, variances)}
        
    def get_params(self, labels_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mus = np.array([self.means[l] for l in labels_list])
        invs = np.array([self.inv_covs[l] for l in labels_list])
        covs = np.array([self.covs[l] for l in labels_list])
        return mus, invs, covs
        
    # def p_n_given_t(self, n_cells, time):
    #     if self.n_prior is None:
    #         return 1.0
        
    #     # find nearest existing observed time
    #     idx = np.abs(self.n_prior['canonical_time'] - time).argmin()
    #     closest_time = self.n_prior.iloc[idx]['canonical_time']
    #     # extract prob of n
    #     prob_row = self.n_prior[(self.n_prior['canonical_time'] == closest_time) & 
    #                             (self.n_prior['N'] == n_cells)]
        
    #     return prob_row['P_N_given_t'].values[0] if not prob_row.empty else 1e-6
    
    # def get_valid_labels(self, time):
    #     """
    #     Returns all labels that are biologically active OR in a ghost state.
    #     We add a 2.0 minute buffer to ensure the 'bridge' is visible.
    #     """
    #     valid_at_time = []
    #     buffer = 2.0  # Allows ghosts to persist for 2 minutes post-division
        
    #     for lbl, data in self._interps.items():
    #         t_min, t_max = data['t_range']
    #         # The new logic: Check range PLUS the ghost buffer
    #         if (t_min - buffer) <= time <= (t_max + buffer):
    #             valid_at_time.append(lbl)
                    
    #     return valid_at_time
        