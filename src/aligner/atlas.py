import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
from typing import List, Dict, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
from sklearn.exceptions import ConvergenceWarning

class StaticGaussianAtlas:
    """
    Manages reference coordinate data derived from labeled embryos.
    
    This class loads a Gaussian Atlas CSV, filters by data quality, 
    and pre-computes inverse covariance matrices for fast Mahalanobis 
    distance calculations during alignment.
    """
    
    def __init__(self, atlas_path: str = None, min_samples: int = 5, reg_eps: float = 1e-6):
        self.means = {}
        self.inv_covs = {}
        self.covs = {}
        self.reg_eps = reg_eps
        
        if atlas_path:
            df = pd.read_csv(atlas_path)
            self._build_lookup(df, min_samples)
            
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, min_samples: int = 5, reg_eps: float = 1e-6):
        """Allows initialization directly from memory."""
        instance = cls(min_samples=min_samples, reg_eps=reg_eps)
        instance._build_lookup(df, min_samples)
        return instance
    
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
    
    def __init__(self, slice_csv_path: str = None):
        self.n_to_ids = {}
        self.id_to_labels = {}
        self.metadata = {}
        if slice_csv_path:
            self._load_slices(slice_csv_path)

    def _populate_from_df(self, df: pd.DataFrame):
        """Centralized logic to load data from a DataFrame."""
        for _, row in df.iterrows():
            s_id = int(row['slice_id'])
            n = int(row['n_cells_frame'])
            labels = tuple(sorted(str(row["cell_names"]).split(";")))
            
            self.n_to_ids.setdefault(n, []).append(s_id)
            self.id_to_labels[s_id] = labels
            self.metadata[s_id] = {
                'is_augmented': bool(row.get('is_augmented', False)),
                'MAP_time': float(row.get('MAP_time', np.nan))
            }

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        instance = cls()
        instance._populate_from_df(df)
        return instance
    
    def _load_slices(self, path: str) -> None:
        self._populate_from_df(pd.read_csv(path))
        
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
        
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, n_prior_df: pd.DataFrame = None):
        instance = cls.__new__(cls)
        instance.df = df
        instance.n_prior = n_prior_df
        instance.labels = df['cell_name'].unique()
        instance._interps = {}
        instance._build_interpolators()
        return instance
    
    def _build_interpolators(self):
        """Build linear interpolators for cell means and variances."""
        for label in self.labels:
            sub = self.df[self.df['cell_name'] == label].sort_values('canonical_time')
            if len(sub) < 2: continue
            t = sub['canonical_time'].values
            if np.any(np.diff(t) <= 0):
                # Simple fix: select only indices where diff > 0
                mask = np.concatenate(([True], np.diff(t) > 0))
                t = t[mask]
                sub = sub.iloc[mask]
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
    def __init__(self, labels, means, variances, min_var: float = 1e-6, reg_eps: float = 1e-6):
        self.min_var = min_var
        self.reg_eps = reg_eps
        self.means = {l: m for l, m in zip(labels, means)}
        
        # 1. Apply safety guards to variance:
        # - Floor (min_var) prevents division by near-zero.
        # - Jitter (reg_eps) ensures positive definiteness.
        safe_vars = np.maximum(np.array(variances), self.min_var) + self.reg_eps
        
        # 2. Construct robust covariance and inverse matrices
        self.covs = {l: np.eye(3) * v for l, v in zip(labels, safe_vars)}
        self.inv_covs = {l: np.eye(3) * (1.0 / v) for l, v in zip(labels, safe_vars)}
        
    def get_params(self, labels_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves means and inverse covariances for a specific set of cell labels.
        Raises KeyError if a label is not found in the initialized adapter.
        """
        mus = []
        invs = []
        covs = []
        
        for l in labels_list:
            if l not in self.means:
                raise KeyError(f"Critical Error: Cell '{l}' missing from GP Adapter.")
            
            mus.append(self.means[l])
            invs.append(self.inv_covs[l])
            covs.append(self.covs[l])
            
        return np.array(mus), np.array(invs), np.array(covs)


class AnchoredAtlas:
    def __init__(self, lh_df):
        self.lh = lh_df.copy()
        self.lh['cell_name'] = self.lh['cell_name'].str.strip()
        self.lh = self.lh.set_index('cell_name')
        
        # 1. STRICTLY define the 6-cell founders
        self.roots = ['ABal', 'ABar', 'ABpl', 'ABpr', 'EMS', 'P2']
        
        # Verify these roots exist in your data
        missing = [r for r in self.roots if r not in self.lh.index]
        if missing:
            print(f"Warning: Missing roots from data: {missing}")
            
        self.tree = self._build_tree()

    def _build_tree(self):
        tree = {}
        names = self.lh.index.tolist()
        # Non-standard divisions for C. elegans
        manual = {'EMS': ['E', 'MS'], 'P2': ['C', 'P3'], 'P3': ['D', 'P4'], 'P4': ['Z2', 'Z3']}
        
        for name in names:
            if name in manual:
                tree[name] = [c for c in manual[name] if c in names]
            # Standard suffix logic (ABal -> ABala)
            children = [c for c in names if c.startswith(name) and len(c) == len(name) + 1]
            if children:
                tree.setdefault(name, []).extend(children)
        return tree

    def get_constrained_state(self, target_N, t_ref):
        # Calculate Z-score progress
        progress = {}
        for name, row in self.lh.iterrows():
            if pd.isna(row['mean_division']):
                progress[name] = -999 
            else:
                progress[name] = (t_ref - row['mean_division']) / (row['std_division'] + 1e-6)

        def generate_slice(threshold):
            ml_vector = []
            def traverse(node):
                children = self.tree.get(node, [])
                # DECISION: Keep mother if no children or threshold not met
                if not children or progress.get(node, -999) < threshold:
                    ml_vector.append(node)
                else:
                    for child in children:
                        traverse(child)
            
            # Start ONLY from the 6 anchored roots
            for r in self.roots:
                if r in self.lh.index:
                    traverse(r)
            return ml_vector

        # Binary Search for threshold
        best_vector = []
        # We use a very wide threshold range to force mothers to stay alive
        for thresh in np.linspace(10, -10, 500): 
            candidate = generate_slice(thresh)
            if len(candidate) == target_N:
                return candidate
            if not best_vector or abs(len(candidate) - target_N) < abs(len(best_vector) - target_N):
                best_vector = candidate
        return best_vector

class AtlasBuilder:
    def __init__(self, full_df: pd.DataFrame, min_points_gp: int = 4, min_count_var: int = 3):
        self.full_df = full_df
        self.min_points_gp = min_points_gp
        self.min_count_var = min_count_var
        self.life_history = None
        
    def fit(self, train_embryo_ids: List[str]) -> Tuple['GPTimeAtlas', 'SliceAtlas']:
        """
        Executes the full pipeline: Existence Matrix -> GP Spatial Atlas -> MAP Slice Atlas.
        """
        train_df = self.full_df[self.full_df['embryo_id'].isin(train_embryo_ids)].copy()
        
        # 1. Build Existence Matrix (Required for MAP estimates)
        self.life_history = self._build_existence_matrix(train_df)
        
        # 2. Fit GP Spatial Trajectories
        gp_df = self._fit_gp_smoothed_means(train_df)
        nprior_df = self._build_nprior(train_df)
        gp_atlas = GPTimeAtlas.from_dataframe(gp_df, nprior_df)
        
        # 3. Build Augmented Slice DB
        slice_df = self._build_augmented_slice_db(train_df)
        slice_db = SliceAtlas.from_dataframe(slice_df)
        
        return gp_atlas, slice_db
    
    def _build_existence_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates global mean birth/division times per cell."""
        bounds = df.groupby(['embryo_id', 'cell_name']).agg(
            t_start=('canonical_time', 'min'), t_end=('canonical_time', 'max')
        ).reset_index()

        lh = bounds.groupby('cell_name').agg(
            mean_birth=('t_start', 'mean'), std_birth=('t_start', 'std'),
            mean_division=('t_end', 'mean'), std_division=('t_end', 'std')
        ).reset_index()
        
        # Cliff time handling: avoid crashing to zero for terminal cells
        cliff_time = lh['mean_division'].max()
        lh.loc[lh['mean_division'] > (cliff_time - 0.5), 'mean_division'] = 1000.0
        return lh.set_index('cell_name')
    
    def _build_nprior(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates P(N | canonical_time)."""
        frame_df = df.groupby(['embryo_id', 'time_idx']).agg({
            'canonical_time': 'first',
            'n_cells_frame': 'first'
        }).reset_index()
        
        nprior_rows = []
        for ct, sub in frame_df.groupby("canonical_time"):
            counts = sub["n_cells_frame"].value_counts().sort_index()
            total = counts.sum()
            for N_val, cnt in counts.items():
                nprior_rows.append({
                    "canonical_time": int(ct),
                    "N": int(N_val),
                    "count": int(cnt),
                    "P_N_given_t": float(cnt / total)
                })
        return pd.DataFrame(nprior_rows)

    @staticmethod
    def _fit_gp_1d(t_train, y_train):
        """Fit a 1D GP y(t) with RBF+White kernel."""
        X = t_train.reshape(-1, 1)
        kernel = (
            1.0 * RBF(length_scale=20.0, length_scale_bounds=(1.0, 200.0))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning) # Sometimes GP fits trigger these
            gp.fit(X, y_train)
        return gp

    def _fit_gp_smoothed_means(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits GP-smoothed trajectories for all cell types.
        """
        # 1. Filter valid data
        mask = (
            (train_df["valid"] == 1) &
            np.isfinite(train_df["canonical_time"]) &
            np.isfinite(train_df[["x_aligned", "y_aligned", "z_aligned"]]).all(axis=1)
        )
        df = train_df[mask].copy()
        time_grid = np.sort(df["canonical_time"].unique().astype(int))

        # 2. Empirical means and counts
        grouped_lt = df.groupby(["cell_name", "canonical_time"])
        means = grouped_lt[["x_aligned", "y_aligned", "z_aligned"]].mean().rename(
            columns={"x_aligned": "mu_x", "y_aligned": "mu_y", "z_aligned": "mu_z"}
        )
        counts = grouped_lt.size().rename("n_obs")
        mean_table = means.join(counts).reset_index()

        atlas_rows = []
        gp_models = {}
        labels = mean_table["cell_name"].unique()

        # 3. Fit GPs
        for lbl in labels:
            sub = mean_table[mean_table["cell_name"] == lbl]
            t_train = sub["canonical_time"].to_numpy(float)
            x_train = sub["mu_x"].to_numpy(float)
            y_train = sub["mu_y"].to_numpy(float)
            z_train = sub["mu_z"].to_numpy(float)
            n_obs = sub["n_obs"].to_numpy(int)

            uniq_t = np.unique(t_train)
            if uniq_t.size < self.min_points_gp:
                for i in range(sub.shape[0]):
                    atlas_rows.append({
                        "cell_name": lbl,
                        "canonical_time": int(t_train[i]),
                        "mu_x": float(x_train[i]),
                        "mu_y": float(y_train[i]),
                        "mu_z": float(z_train[i]),
                        "n_obs": int(n_obs[i]),
                        "sigma2_gp": 0.0,
                    })
                gp_models[lbl] = None
                continue

            # Fit GPs
            gp_x = self._fit_gp_1d(uniq_t, [x_train[t_train == tt].mean() for tt in uniq_t])
            gp_y = self._fit_gp_1d(uniq_t, [y_train[t_train == tt].mean() for tt in uniq_t])
            gp_z = self._fit_gp_1d(uniq_t, [z_train[t_train == tt].mean() for tt in uniq_t])
            gp_models[lbl] = (gp_x, gp_y, gp_z)

            # Predict on grid
            t_min, t_max = uniq_t.min(), uniq_t.max()
            grid_lbl = time_grid[(time_grid >= t_min) & (time_grid <= t_max)]
            X_grid = grid_lbl.reshape(-1, 1)
            
            mu_x_pred, std_x = gp_x.predict(X_grid, return_std=True)
            mu_y_pred, std_y = gp_y.predict(X_grid, return_std=True)
            mu_z_pred, std_z = gp_z.predict(X_grid, return_std=True)

            sigma2_gp = std_x**2 + std_y**2 + std_z**2
            n_obs_dict = {int(ct): int(n) for ct, n in zip(t_train.astype(int), n_obs)}

            for t, mx, my, mz, s2gp in zip(grid_lbl, mu_x_pred, mu_y_pred, mu_z_pred, sigma2_gp):
                atlas_rows.append({
                    "cell_name": lbl,
                    "canonical_time": int(t),
                    "mu_x": float(mx),
                    "mu_y": float(my),
                    "mu_z": float(mz),
                    "n_obs": int(n_obs_dict.get(int(t), 0)),
                    "sigma2_gp": float(s2gp),
                })

        atlas_df = pd.DataFrame(atlas_rows)

        # 4. Variance Estimation (Sigma2_label)
        sigma2_by_label = {}
        for lbl in labels:
            sub_raw = df[df["cell_name"] == lbl]
            if len(sub_raw) < self.min_count_var:
                sigma2_by_label[lbl] = np.nan
                continue

            t_samp = sub_raw["canonical_time"].to_numpy(float).reshape(-1, 1)
            coords = sub_raw[["x_aligned", "y_aligned", "z_aligned"]].to_numpy(float)

            model = gp_models.get(lbl, None)
            if model is None:
                means_lbl = mean_table[mean_table["cell_name"] == lbl][["canonical_time", "mu_x", "mu_y", "mu_z"]].set_index("canonical_time")
                sub2 = sub_raw.merge(means_lbl, left_on="canonical_time", right_index=True, how="inner")
                r2 = (sub2["x_aligned"] - sub2["mu_x"])**2 + (sub2["y_aligned"] - sub2["mu_y"])**2 + (sub2["z_aligned"] - sub2["mu_z"])**2
            else:
                gp_x, gp_y, gp_z = model
                r2 = (coords[:,0] - gp_x.predict(t_samp))**2 + (coords[:,1] - gp_y.predict(t_samp))**2 + (coords[:,2] - gp_z.predict(t_samp))**2

            sigma2_by_label[lbl] = float(r2.mean() / 3.0)

        # Global fallback
        sigma_vals = np.array([v for v in sigma2_by_label.values() if np.isfinite(v) and v > 0])
        global_sigma2 = float(np.median(sigma_vals)) if sigma_vals.size > 0 else 1.0
        
        atlas_df["sigma2_label"] = atlas_df["cell_name"].map(lambda x: sigma2_by_label.get(x, global_sigma2) if np.isfinite(sigma2_by_label.get(x, np.nan)) else global_sigma2)

        return atlas_df[["cell_name", "canonical_time", "mu_x", "mu_y", "mu_z", "n_obs", "sigma2_label", "sigma2_gp"]]
    
    # def _build_augmented_slice_db(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Generates observed and MAP-augmented slice configurations."""
    #     # 1. Standardize observed data
    #     obs = df[df['valid'] == 1].groupby(['embryo_id', 'time_idx'], group_keys=False).apply(
    #         lambda x: ";".join(sorted(x['cell_name'].str.strip().unique())),
    #         include_groups=False
    #     ).to_frame(name='cell_names').reset_index()
        
    #     obs['n_cells_frame'] = obs['cell_names'].apply(lambda x: len(x.split(';')))
    #     obs['is_augmented'] = False
        
    #     # 2. Generate MAP-augmented slices
    #     map_rows = []
    #     n_min, n_max = int(df['n_cells_frame'].min()), int(df['n_cells_frame'].max())
    #     for n in range(n_min, n_max + 1):
    #         t_map, labels, conf, sigma = self._predict_map_state(n)
    #         map_rows.append({
    #             'n_cells_frame': n,
    #             'cell_names': ";".join(sorted(labels)),
    #             'is_augmented': True,
    #             'MAP_time': t_map,
    #             'MAP_confidence': conf
    #         })
        
    #     # 3. CONCAT AND DEDUPLICATE
    #     master_df = pd.concat([obs, pd.DataFrame(map_rows)], ignore_index=True)
        
    #     # Ensure label consistency: strip and sort again just in case
    #     master_df['cell_names'] = master_df['cell_names'].apply(
    #         lambda x: ";".join(sorted([n.strip() for n in x.split(";")]))
    #     )
        
    #     # Drop duplicates, prioritizing observed (non-augmented) slices
    #     master_df = master_df.sort_values('is_augmented').drop_duplicates(subset=['cell_names'])
        
    #     # Assign unique IDs
    #     master_df = master_df.reset_index(drop=True)
    #     master_df['slice_id'] = master_df.index
        
    #     return master_df
    
    def _build_augmented_slice_db(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Create a clean slice-level summary
        valid_df = df[df['valid'] == 1].copy()
        
        # Aggregate to get cell_names and n_cells_frame per frame
        slice_summary = valid_df.groupby(['embryo_id', 'time_idx']).agg(
            cell_names=('cell_name', lambda x: ";".join(sorted(x.unique()))),
            n_cells_frame=('cell_name', 'count'),
            canonical_time=('canonical_time', 'mean')
        ).reset_index()
        
        master_rows = []
        
        # 2. Observed Slices: Group the summary by 'cell_names'
        # This guarantees 'n_cells_frame' and 'cell_names' are in the groups
        obs_groups = slice_summary.groupby('cell_names')
        for config, sub in obs_groups:
            master_rows.append({
                'n_cells_frame': sub['n_cells_frame'].iloc[0],
                'cell_names': config,
                'is_augmented': False,
                'MAP_time': sub['canonical_time'].mean()
            })
            
        # 3. Augmented Slices
        n_min = int(slice_summary['n_cells_frame'].min())
        n_max = int(slice_summary['n_cells_frame'].max())
        
        for n in range(n_min, n_max + 1):
            t_map, labels, _, _ = self._predict_map_state(n)
            master_rows.append({
                'n_cells_frame': n,
                'cell_names': ";".join(sorted(labels)),
                'is_augmented': True,
                'MAP_time': t_map
            })
            
        # 4. Finalize
        master_df = pd.DataFrame(master_rows)
        master_df = master_df.sort_values('is_augmented').drop_duplicates(subset=['cell_names'])
        master_df['slice_id'] = master_df.index
        return master_df
    
    def _predict_map_state(self, target_N, t_max=250):
        """Calculates P(N|t) and derives AnchoredAtlas state."""
        t_grid = np.linspace(0, t_max, 500)
        # Prob(Exists at t)
        p_exists = norm.cdf(t_grid[:, None], self.life_history['mean_birth'].values, self.life_history['std_birth'].values + 1e-6) - \
                   norm.cdf(t_grid[:, None], self.life_history['mean_division'].values, self.life_history['std_division'].values + 1e-6)
        
        mu_n, var_n = np.sum(p_exists, axis=1), np.sum(p_exists * (1 - p_exists), axis=1)
        var_n = np.maximum(var_n, 1e-9)
        posteriors = norm.pdf(target_N, mu_n, np.sqrt(var_n))
        posteriors /= (posteriors.sum() + 1e-12)
        
        map_t = t_grid[np.argmax(posteriors)]
        
        # Traverse AnchoredAtlas lineage tree
        anchored = AnchoredAtlas(self.life_history.reset_index())
        labels = anchored.get_constrained_state(target_N, map_t)
        
        std_t = np.sqrt(np.sum(posteriors * (t_grid - np.sum(posteriors * t_grid))**2))
        return map_t, labels, np.exp(-std_t / 5.0), std_t

class LegacyAtlasBuilder:
    """
    Constructs Legacy Atlas objects directly from clean_df in-memory.
    No CSV file writing is required for the pipeline integration.
    """
    def __init__(self, full_df: pd.DataFrame, min_samples: int = 10):
        self.full_df = full_df
        self.min_samples = min_samples

    def fit(self, train_embryo_ids: List[str]) -> Tuple[StaticGaussianAtlas, SliceAtlas, GPTimeAtlas]:
        """
        Builds the 3 core atlas objects in-memory for legacy consumption.
        """
        train_df = self.full_df[self.full_df['embryo_id'].isin(train_embryo_ids)].copy()
        
        # 1. Build Static Gaussian Atlas
        gauss_df = self._build_gaussian_atlas(train_df)
        gauss_atlas = StaticGaussianAtlas.from_dataframe(gauss_df, min_samples=self.min_samples)
        
        # 2. Build Slice Atlas
        # We assume your SliceAtlas.from_dataframe uses the 'slice_frequencies' format
        _, slice_freq = self._build_slice_stats(train_df)
        slice_atlas = SliceAtlas.from_dataframe(slice_freq)
        
        return gauss_atlas, slice_atlas

    def _build_gaussian_atlas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes lifetime-aggregated 3D Gaussian stats per cell."""
        df = df[(df["canonical_time"] > -1) & df[["x_aligned", "y_aligned", "z_aligned"]].notna().all(axis=1)].copy()
        centroid = df[["x_aligned", "y_aligned", "z_aligned"]].mean().values
        rows = []
        
        for cell_name, g in df.groupby("cell_name"):
            coords = g[["x_aligned", "y_aligned", "z_aligned"]].values
            n = coords.shape[0]
            if n < self.min_samples: continue
            
            mu = coords.mean(axis=0)
            cov = np.cov(coords.T) + 1e-6 * np.eye(3)
            
            # Diagnostic radial/tangential projection
            v = mu - centroid
            norm_v = np.linalg.norm(v) or 1e-8
            radial_dir = v / norm_v
            r = (coords - mu) @ radial_dir
            
            rows.append({
                "cell_name": cell_name, "n_samples": n,
                "mu_x": mu[0], "mu_y": mu[1], "mu_z": mu[2],
                "cov_xx": cov[0,0], "cov_xy": cov[0,1], "cov_xz": cov[0,2],
                "cov_yy": cov[1,1], "cov_yz": cov[1,2], "cov_zz": cov[2,2],
                "radial_mean": r.mean(), "radial_var": r.var(ddof=1) if n > 1 else 0,
                "tangential_var": np.linalg.norm(coords - mu - np.outer(r, radial_dir), axis=1).var(ddof=1) if n > 1 else 0,
                "canonical_time_mean": g["canonical_time"].mean(),
                "canonical_time_std": g["canonical_time"].std() if n > 1 else 0,
                "canonical_time_min": g["canonical_time"].min(),
                "canonical_time_max": g["canonical_time"].max()
            })
        return pd.DataFrame(rows)

    def _build_slice_stats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate frames to create slice catalog and frequency table."""
        slice_inst = df.groupby(["embryo_id", "time_idx"]).agg(
            cell_names=("cell_name", lambda s: ";".join(sorted(map(str, s)))),
            n_cells_frame=("cell_name", "count")
        ).reset_index()
        
        freq = slice_inst.groupby(["n_cells_frame", "cell_names"]).size().reset_index(name="count")
        freq["p_slice_given_N"] = freq["count"] / freq.groupby("n_cells_frame")["count"].transform("sum")
        freq["slice_id"] = freq.index
        return slice_inst, freq

