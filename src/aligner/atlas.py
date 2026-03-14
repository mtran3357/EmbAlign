import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm
from typing import List, Tuple, Dict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
from sklearn.exceptions import ConvergenceWarning

# Import your new configuration schema
from aligner.config import PipelineConfig, AtlasStrategy, SliceStrategy

class StaticGaussianAtlas:
    """Manages reference coordinate data derived from labeled embryos."""
    
    def __init__(self, atlas_path: str = None, min_samples: int = 5, reg_eps: float = 1e-6):
        self.means = {}
        self.inv_covs = {}
        self.covs = {}
        self.reg_eps = reg_eps
        
        if atlas_path:
            self._build_lookup(pd.read_csv(atlas_path), min_samples)
            
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, min_samples: int = 5, reg_eps: float = 1e-6):
        instance = cls(min_samples=min_samples, reg_eps=reg_eps)
        instance._build_lookup(df, min_samples)
        return instance
    
    def _build_lookup(self, df: pd.DataFrame, min_samples: int) -> None:
        valid_df = df[df['n_samples'] >= min_samples].copy()
        for _, row in valid_df.iterrows():
            name = str(row['cell_name'])
            mu = np.array([row['mu_x'], row['mu_y'], row['mu_z']], dtype=float)
            cov = np.array([
                [row['cov_xx'], row['cov_xy'], row['cov_xz']],
                [row['cov_xy'], row['cov_yy'], row['cov_yz']],
                [row['cov_xz'], row['cov_yz'], row['cov_zz']],
            ], dtype=float)
            
            cov_reg = cov + self.reg_eps * np.eye(3)
            try: 
                inv_cov = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_reg)
            
            self.means[name] = mu
            self.covs[name] = cov_reg
            self.inv_covs[name] = inv_cov
        
    def get_params(self, labels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mus, invs, covs = [], [], []
        for l in labels:
            if l not in self.means:
                raise KeyError(f"Critical Error: Cell '{l}' missing from Atlas.")
            mus.append(self.means[l])
            invs.append(self.inv_covs[l])
            covs.append(self.covs[l])
        return np.vstack(mus), np.array(invs), np.array(covs)
    
class SliceAtlas:
    """Manages the database of valid cell-type combinations (slices)."""
    
    def __init__(self, slice_csv_path: str = None):
        self.n_to_ids = {}
        self.id_to_labels = {}
        self.metadata = {}
        if slice_csv_path:
            self._populate_from_df(pd.read_csv(slice_csv_path))

    def _populate_from_df(self, df: pd.DataFrame):
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
        
    def get_candidates(self, n_cells: int) -> List[int]:
        return self.n_to_ids.get(n_cells, [])

    def get_labels(self, slice_id: int) -> Tuple[str, ...]:
        return self.id_to_labels.get(slice_id, ())

class GPTimeAtlas:
    """Time resolved atlas from GP."""
    def __init__(self, atlas_csv_path: str = None):
        self._interps = {}
        if atlas_csv_path:
            raw_df = pd.read_csv(atlas_csv_path)
            self.df = raw_df.groupby(['cell_name', 'canonical_time']).mean(numeric_only=True).reset_index()
            self.labels = self.df['cell_name'].unique()
            self._build_interpolators()
        
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        instance = cls()
        instance.df = df
        instance.labels = df['cell_name'].unique()
        instance._build_interpolators()
        return instance
    
    def _build_interpolators(self):
        for label in self.labels:
            sub = self.df[self.df['cell_name'] == label].sort_values('canonical_time')
            if len(sub) < 2: continue
            t = sub['canonical_time'].values
            if np.any(np.diff(t) <= 0):
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

    def get_state(self, time: float, active_labels: List[str] = None) -> Dict:
        means, variances, labels = [], [], []
        target_labels = active_labels if active_labels is not None else self.labels
        
        for lbl in target_labels:
            if lbl in self._interps:
                d = self._interps[lbl]
                means.append(d['mu'](time))
                variances.append(d['sigma2'](time))
                labels.append(lbl)
            else:
                raise KeyError(f"Critical Error: Cell '{lbl}' missing from GP Atlas.")
                
        return {'labels': labels, 'means': np.array(means), 'variances': np.array(variances)}
        
class SliceTimeAtlas:
    """Maps Slice Ids to their valid canonical time windows."""
    def __init__(self, gp_atlas: GPTimeAtlas, slice_db: SliceAtlas, padding: float = 1.0):
        self.gp_atlas = gp_atlas
        self.slice_db = slice_db
        self.padding = padding
        self.slice_windows = {}
        self._map_slices_to_time()
    
    def _map_slices_to_time(self):
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
    
    def get_temporal_state(self, s_id: int, time_offset: float = 0.5):
        win = self.slice_windows.get(s_id)
        if not win: return None
        t_start, t_end = win['t_range']
        target_t = t_start + (t_end - t_start) * time_offset
        return self.gp_atlas.get_state(target_t, active_labels=win['labels'])
    
class GPToStaticAdapter:
    """Makes a dynamic GP state look like a Static GaussianAtlas for legacy engine use."""
    def __init__(self, labels: List[str], means: np.ndarray, variances: np.ndarray, min_var: float = 1e-6, reg_eps: float = 1e-6):
        self.min_var = min_var
        self.reg_eps = reg_eps
        self.means = {l: m for l, m in zip(labels, means)}
        safe_vars = np.maximum(np.array(variances), self.min_var) + self.reg_eps
        self.covs = {l: np.eye(3) * v for l, v in zip(labels, safe_vars)}
        self.inv_covs = {l: np.eye(3) * (1.0 / v) for l, v in zip(labels, safe_vars)}
        
    def get_params(self, labels_list: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mus, invs, covs = [], [], []
        for l in labels_list:
            if l not in self.means:
                raise KeyError(f"Critical Error: Cell '{l}' missing from GP Adapter.")
            mus.append(self.means[l])
            invs.append(self.inv_covs[l])
            covs.append(self.covs[l])
        return np.array(mus), np.array(invs), np.array(covs)

class AnchoredAtlas:
    # Explicit Biological Constants 
    ROOTS = ['ABal', 'ABar', 'ABpl', 'ABpr', 'EMS', 'P2']
    MANUAL_DIVISIONS = {'EMS': ['E', 'MS'], 'P2': ['C', 'P3'], 'P3': ['D', 'P4'], 'P4': ['Z2', 'Z3']}

    def __init__(self, lh_df: pd.DataFrame):
        self.lh = lh_df.copy()
        self.lh['cell_name'] = self.lh['cell_name'].str.strip()
        self.lh = self.lh.set_index('cell_name')
        
        missing = [r for r in self.ROOTS if r not in self.lh.index]
        if missing:
            print(f"Warning: Missing roots from data: {missing}")
            
        self.tree = self._build_tree()

    def _build_tree(self):
        tree = {}
        names = self.lh.index.tolist()
        for name in names:
            if name in self.MANUAL_DIVISIONS:
                tree[name] = [c for c in self.MANUAL_DIVISIONS[name] if c in names]
            children = [c for c in names if c.startswith(name) and len(c) == len(name) + 1]
            if children:
                tree.setdefault(name, []).extend(children)
        return tree

    def get_constrained_state(self, target_N: int, t_ref: float) -> List[str]:
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
                if not children or progress.get(node, -999) < threshold:
                    ml_vector.append(node)
                else:
                    for child in children:
                        traverse(child)
            
            for r in self.ROOTS:
                if r in self.lh.index:
                    traverse(r)
            return ml_vector

        best_vector = []
        for thresh in np.linspace(10, -10, 500): 
            candidate = generate_slice(thresh)
            if len(candidate) == target_N:
                return candidate
            if not best_vector or abs(len(candidate) - target_N) < abs(len(best_vector) - target_N):
                best_vector = candidate
        return best_vector
    
class AnchoredAtlas:
    # Explicit Biological Constants 
    ROOTS = ['ABal', 'ABar', 'ABpl', 'ABpr', 'EMS', 'P2']
    MANUAL_DIVISIONS = {'EMS': ['E', 'MS'], 'P2': ['C', 'P3'], 'P3': ['D', 'P4'], 'P4': ['Z2', 'Z3']}

    def __init__(self, lh_df: pd.DataFrame):
        self.lh = lh_df.copy()
        self.lh['cell_name'] = self.lh['cell_name'].str.strip()
        self.lh = self.lh.set_index('cell_name')
        
        missing = [r for r in self.ROOTS if r not in self.lh.index]
        if missing:
            print(f"Warning: Missing roots from data: {missing}")
            
        self.tree = self._build_tree()

    def _build_tree(self):
        tree = {}
        names = self.lh.index.tolist()
        for name in names:
            if name in self.MANUAL_DIVISIONS:
                tree[name] = [c for c in self.MANUAL_DIVISIONS[name] if c in names]
            children = [c for c in names if c.startswith(name) and len(c) == len(name) + 1]
            if children:
                tree.setdefault(name, []).extend(children)
        return tree

    def get_constrained_state(self, target_N: int, t_ref: float) -> List[str]:
        """
        Strictly generates a valid lineage state with EXACTLY target_N cells.
        Divides cells one-by-one based on their life progress to mathematically 
        prevent gaps from synchronous divisions.
        """
        # 1. Calculate division readiness for all cells
        progress = {}
        for name, row in self.lh.iterrows():
            if pd.isna(row['mean_division']):
                progress[name] = -999 # Terminal cells never divide
            else:
                progress[name] = (t_ref - row['mean_division']) / (row['std_division'] + 1e-6)

        # 2. Start at the biological roots
        active_cells = [r for r in self.ROOTS if r in self.lh.index]
        
        # 3. FORWARD TRAVERSAL: Divide cells one by one until we hit target_N
        while len(active_cells) < target_N:
            # Find all cells currently active that are capable of dividing
            dividable = [c for c in active_cells if c in self.tree and len(self.tree[c]) > 0]
            
            if not dividable:
                print(f"[Atlas Warning] Hit terminal lineage state at {len(active_cells)} cells. Cannot reach {target_N}.")
                break
                
            # Pick the cell most "overdue" for division (highest life progress)
            to_divide = max(dividable, key=lambda c: progress.get(c, -999))
            
            # Perform the division (removes parent, adds children)
            active_cells.remove(to_divide)
            active_cells.extend(self.tree[to_divide])
            
        # 4. BACKWARD TRAVERSAL (Safety Net): If target_N < len(ROOTS)
        while len(active_cells) > target_N:
            # Find all sets of active children that share a parent
            parents_of_active = {}
            for parent, children in self.tree.items():
                if all(c in active_cells for c in children):
                    parents_of_active[parent] = children
                    
            if not parents_of_active:
                break
                
            # Pick the parent that divided MOST RECENTLY (lowest progress) to merge back
            to_merge = min(parents_of_active.keys(), key=lambda p: progress.get(p, 999))
            
            # Perform the merge (remove children, add parent)
            for child in parents_of_active[to_merge]:
                active_cells.remove(child)
            active_cells.append(to_merge)

        return active_cells

class AtlasFactory:
    """Master factory that orchestrates atlas construction based on the active PipelineConfig."""
    def __init__(self, full_df: pd.DataFrame, config: PipelineConfig):
        self.full_df = full_df
        self.config = config
        # self.life_history = self._build_existence_matrix(full_df)
        self.min_samples = config.min_samples_static
        self.min_points_gp = config.min_points_gp
        self.min_count_var = config.min_count_var
        
    def build(self, train_embryo_ids: List[str]) -> Tuple:
        """Constructs the exact Spatial and Slice atlases required by the config."""
        train_df = self.full_df[self.full_df['embryo_id'].isin(train_embryo_ids)].copy()
        self.life_history = self._build_existence_matrix(train_df)
        # 1. Build Spatial Geometry (Static vs GP)
        if self.config.atlas_strategy == AtlasStrategy.STATIC:
            gauss_df = self._build_static_gaussians(train_df)
            spatial_atlas = StaticGaussianAtlas.from_dataframe(gauss_df, min_samples=self.min_samples)
        else:
            self.life_history = self._build_existence_matrix(train_df)
            gp_df = self._fit_gp_smoothed_means(train_df)
            spatial_atlas = GPTimeAtlas.from_dataframe(gp_df)
            
        # 2. Build Slice Constraints (Observed vs Augmented)
        if self.config.slice_strategy == SliceStrategy.OBSERVED:
            slice_df = self._build_observed_slices(train_df)
        else:
            if self.life_history is None:
                self.life_history = self._build_existence_matrix(train_df)
            slice_df = self._build_augmented_slice_db(train_df)
            
        slice_atlas = SliceAtlas.from_dataframe(slice_df)
        return spatial_atlas, slice_atlas

    def _build_static_gaussians(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stripped down, highly efficient 3D Gaussian calculation."""
        df = df[(df["valid"] == 1) & df[["x_aligned", "y_aligned", "z_aligned"]].notna().all(axis=1)].copy()
        rows = []
        for cell_name, g in df.groupby("cell_name"):
            coords = g[["x_aligned", "y_aligned", "z_aligned"]].values
            n = coords.shape[0]
            if n < self.min_samples: continue
            
            mu = coords.mean(axis=0)
            cov = np.cov(coords.T) + 1e-6 * np.eye(3)
            rows.append({
                "cell_name": cell_name, "n_samples": n,
                "mu_x": mu[0], "mu_y": mu[1], "mu_z": mu[2],
                "cov_xx": cov[0,0], "cov_xy": cov[0,1], "cov_xz": cov[0,2],
                "cov_yy": cov[1,1], "cov_yz": cov[1,2], "cov_zz": cov[2,2],
            })
        return pd.DataFrame(rows)

    def _build_existence_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        bounds = df.groupby(['embryo_id', 'cell_name']).agg(
            t_start=('canonical_time', 'min'), t_end=('canonical_time', 'max')
        ).reset_index()

        lh = bounds.groupby('cell_name').agg(
            mean_birth=('t_start', 'mean'), std_birth=('t_start', 'std'),
            mean_division=('t_end', 'mean'), std_division=('t_end', 'std')
        ).reset_index()
        
        cliff_time = lh['mean_division'].max()
        lh.loc[lh['mean_division'] > (cliff_time - 0.5), 'mean_division'] = 1000.0
        return lh.set_index('cell_name')

    def _fit_gp_smoothed_means(self, train_df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (train_df["valid"] == 1) &
            np.isfinite(train_df["canonical_time"]) &
            np.isfinite(train_df[["x_aligned", "y_aligned", "z_aligned"]]).all(axis=1)
        )
        df = train_df[mask].copy()
        time_grid = np.sort(df["canonical_time"].unique().astype(int))

        grouped_lt = df.groupby(["cell_name", "canonical_time"])
        means = grouped_lt[["x_aligned", "y_aligned", "z_aligned"]].mean().rename(
            columns={"x_aligned": "mu_x", "y_aligned": "mu_y", "z_aligned": "mu_z"}
        )
        counts = grouped_lt.size().rename("n_obs")
        mean_table = means.join(counts).reset_index()

        atlas_rows = []
        gp_models = {}
        labels = mean_table["cell_name"].unique()

        def fit_gp_1d(t_train, y_train):
            X = t_train.reshape(-1, 1)
            kernel = (1.0 * RBF(length_scale=20.0, length_scale_bounds=(1.0, 200.0))
                      + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                gp.fit(X, y_train)
            return gp

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
                        "cell_name": lbl, "canonical_time": int(t_train[i]),
                        "mu_x": float(x_train[i]), "mu_y": float(y_train[i]), "mu_z": float(z_train[i]),
                        "n_obs": int(n_obs[i]), "sigma2_gp": 0.0,
                    })
                gp_models[lbl] = None
                continue

            gp_x = fit_gp_1d(uniq_t, [x_train[t_train == tt].mean() for tt in uniq_t])
            gp_y = fit_gp_1d(uniq_t, [y_train[t_train == tt].mean() for tt in uniq_t])
            gp_z = fit_gp_1d(uniq_t, [z_train[t_train == tt].mean() for tt in uniq_t])
            gp_models[lbl] = (gp_x, gp_y, gp_z)

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
                    "cell_name": lbl, "canonical_time": int(t),
                    "mu_x": float(mx), "mu_y": float(my), "mu_z": float(mz),
                    "n_obs": int(n_obs_dict.get(int(t), 0)), "sigma2_gp": float(s2gp),
                })

        atlas_df = pd.DataFrame(atlas_rows)

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

        sigma_vals = np.array([v for v in sigma2_by_label.values() if np.isfinite(v) and v > 0])
        global_sigma2 = float(np.median(sigma_vals)) if sigma_vals.size > 0 else 1.0
        
        atlas_df["sigma2_label"] = atlas_df["cell_name"].map(lambda x: sigma2_by_label.get(x, global_sigma2) if np.isfinite(sigma2_by_label.get(x, np.nan)) else global_sigma2)

        return atlas_df[["cell_name", "canonical_time", "mu_x", "mu_y", "mu_z", "n_obs", "sigma2_label", "sigma2_gp"]]
    
    def _build_observed_slices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Isolates purely empirical slice data without MAP estimation."""
        slice_summary = df[df['valid'] == 1].groupby(['embryo_id', 'time_idx']).agg(
            cell_names=('cell_name', lambda x: ";".join(sorted(x.unique()))),
            n_cells_frame=('cell_name', 'count'),
            canonical_time=('canonical_time', 'mean')
        ).reset_index()
        
        master_rows = []
        for config, sub in slice_summary.groupby('cell_names'):
            master_rows.append({
                'n_cells_frame': sub['n_cells_frame'].iloc[0],
                'cell_names': config,
                'is_augmented': False,
                'MAP_time': sub['canonical_time'].mean()
            })
            
        master_df = pd.DataFrame(master_rows).drop_duplicates(subset=['cell_names']).reset_index(drop=True)
        master_df['slice_id'] = master_df.index
        return master_df

    def _build_augmented_slice_db(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combines observed slices with MAP-estimated gap fillers."""
        observed_df = self._build_observed_slices(df)
        master_rows = observed_df.to_dict('records')
        
        n_min = int(observed_df['n_cells_frame'].min())
        n_max = int(observed_df['n_cells_frame'].max())
        
        for n in range(n_min, n_max + 1):
            t_map, labels, _, _ = self._predict_map_state(n)
            master_rows.append({
                'n_cells_frame': n,
                'cell_names': ";".join(sorted(labels)),
                'is_augmented': True,
                'MAP_time': t_map
            })
            
        master_df = pd.DataFrame(master_rows)
        master_df = master_df.sort_values('is_augmented').drop_duplicates(subset=['cell_names']).reset_index(drop=True)
        master_df['slice_id'] = master_df.index
        return master_df
    
    def _predict_map_state(self, target_N: int):
        t_max = self.config.map_t_max
        t_grid = np.linspace(0, t_max, 500)
        p_exists = norm.cdf(t_grid[:, None], self.life_history['mean_birth'].values, self.life_history['std_birth'].values + 1e-6) - \
                   norm.cdf(t_grid[:, None], self.life_history['mean_division'].values, self.life_history['std_division'].values + 1e-6)
        
        mu_n, var_n = np.sum(p_exists, axis=1), np.sum(p_exists * (1 - p_exists), axis=1)
        var_n = np.maximum(var_n, 1e-9)
        posteriors = norm.pdf(target_N, mu_n, np.sqrt(var_n))
        posteriors /= (posteriors.sum() + 1e-12)
        
        map_t = t_grid[np.argmax(posteriors)]
        
        anchored = AnchoredAtlas(self.life_history.reset_index())
        labels = anchored.get_constrained_state(target_N, map_t)
        
        std_t = np.sqrt(np.sum(posteriors * (t_grid - np.sum(posteriors * t_grid))**2))
        return map_t, labels, np.exp(-std_t / 5.0), std_t